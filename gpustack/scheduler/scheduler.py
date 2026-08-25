import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import queue
import time
import uuid
from typing import List, Tuple, Optional
from sqlalchemy import func, select
from sqlmodel.ext.asyncio.session import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.config.config import Config
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    WorkerFilterChain,
)
from gpustack.policies.candidate_selectors import (
    AscendMindIEResourceFitSelector,
    GGUFResourceFitSelector,
    VLLMResourceFitSelector,
    VoxBoxResourceFitSelector,
)
from gpustack.policies.utils import ListMessageBuilder
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.scheduler.model_registry import (
    vllm_supported_embedding_architectures,
    vllm_supported_llm_architectures,
    vllm_supported_reranker_architectures,
)
from gpustack.scheduler.placement_override import get_model_for_instance_scheduling
from gpustack.scheduler.aggregation import (
    candidate_snapshot,
    filter_by_aggregation_rate,
)
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack.schemas.workers import Worker
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
    get_backend,
    is_gguf_model,
    is_gguf_backend,
    is_audio_model,
    DistributedServerCoordinateModeEnum,
    PlacementStrategyEnum,
)
from gpustack.schemas.scheduler import (
    SchedulerPolicy,
    SchedulingAttemptEvent,
    SchedulingOutcome,
)
from gpustack.server.bus import EventType
from gpustack.server.db import get_engine
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    calculate_model_resource_claim,
)
from gpustack.server.services import ModelInstanceService, ModelService
from gpustack.utils.command import find_parameter
from gpustack.utils.gpu import parse_gpu_ids_by_worker
from gpustack.utils.hub import get_pretrained_config
from gpustack.utils.task import run_in_thread

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, cfg: Config, check_interval: int = 180):
        """
        Init the scheduler with queue and interval.
        """

        self._id = "model-instance-scheduler"
        self._config = cfg
        self._check_interval = check_interval
        self._engine = get_engine()
        self._queue = AsyncUniqueQueue()
        self._cache_dir = None

        if self._config.cache_dir is not None:
            self._cache_dir = os.path.join(self._config.cache_dir, "gguf-parser")
            os.makedirs(self._cache_dir, exist_ok=True)

            self._vox_box_cache_dir = os.path.join(self._config.cache_dir, "vox-box")
            os.makedirs(self._vox_box_cache_dir, exist_ok=True)

    async def start(self):
        """
        Start the scheduler.
        """

        try:
            # scheduler queue.
            asyncio.create_task(self._run_schedule_cycle())

            # scheduler job trigger by time interval.
            trigger = IntervalTrigger(
                seconds=self._check_interval, timezone=timezone.utc
            )
            scheduler = AsyncIOScheduler(timezone=timezone.utc)
            scheduler.add_job(
                self._enqueue_pending_instances,
                trigger=trigger,
                id=self._id,
                max_instances=1,
            )
            scheduler.start()
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

        logger.info("Scheduler started.")

        await self._run_event_trigger()

    async def _run_schedule_cycle(self):
        while True:
            try:
                await self._schedule_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scheduler cycle failed: {e}")
                await asyncio.sleep(5)

    async def _run_event_trigger(self):
        while True:
            try:
                async for event in ModelInstance.subscribe(self._engine):
                    if event.type != EventType.CREATED:
                        continue

                    await self._enqueue_pending_instances()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Scheduler event trigger failed: {e}")
                await asyncio.sleep(5)

    async def _enqueue_pending_instances(self):
        """
        Get the pending model instances.
        """
        try:
            async with AsyncSession(self._engine) as session:
                instances = await ModelInstance.all(session)
                tasks = []
                for instance in instances:
                    if self._should_schedule(instance):
                        task = asyncio.create_task(self._evaluate(instance))
                        tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")

    async def _evaluate(self, instance: ModelInstance):  # noqa: C901
        """
        Evaluate the model instance's metadata.
        """
        async with AsyncSession(self._engine) as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)

                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")

                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements"
                    await ModelInstanceService(session).update(instance)

                if model.source == SourceEnum.LOCAL_PATH and not os.path.exists(
                    model.local_path
                ):
                    # The local path model is not accessible from the server, skip evaluation.
                    await self._queue.put(instance)
                    return

                should_update_model = False
                try:
                    evaluation_model = get_model_for_instance_scheduling(
                        model, instance
                    )
                    if is_gguf_model(evaluation_model):
                        should_update_model = await evaluate_gguf_model(
                            self._config, evaluation_model
                        )
                        if await self.check_model_distributability(
                            session, evaluation_model, instance
                        ):
                            return
                    elif is_audio_model(evaluation_model):
                        should_update_model = await evaluate_audio_model(
                            self._config, evaluation_model
                        )
                    elif get_backend(evaluation_model) == BackendEnum.VLLM_OMNI:
                        # vLLM-Omni can serve models that do not expose a standard
                        # transformers config, such as diffusion-format models.
                        evaluation_model.categories = evaluation_model.categories or [
                            CategoryEnum.LLM
                        ]
                    else:
                        should_update_model = await evaluate_pretrained_config(
                            evaluation_model, raise_raw=True
                        )
                except Exception as e:
                    # Even if the evaluation failed, we still want to proceed to deployment.
                    # Cases can be:
                    # 1. Model config is not valid, but is overridable by backend parameters.
                    # 2. It may not be required to be transformer-compatible for certain backends.
                    logger.error(
                        f"Failed to evaluate model {model.name or model.readable_source}: {e}"
                    )

                if should_update_model:
                    await ModelService(session).update(model)

                await self._queue.put(instance)
            except Exception as e:
                try:
                    instance.state = ModelInstanceStateEnum.ERROR
                    instance.state_message = str(e)
                    await ModelInstanceService(session).update(instance)
                except Exception as ue:
                    logger.error(
                        f"Failed to update model instance: {ue}. Original error: {e}"
                    )

    async def check_model_distributability(
        self, session: AsyncSession, model: Model, instance: ModelInstance
    ):
        if (
            not model.distributable
            and model.gpu_selector
            and model.gpu_selector.gpu_ids
        ):
            worker_gpu_ids = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if len(worker_gpu_ids) > 1:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = (
                    "The model is not distributable to multiple workers."
                )
                await ModelInstanceService(session).update(instance)
                return True
        return False

    def _should_schedule(self, instance: ModelInstance) -> bool:
        """
        Check if the model instance should be scheduled.
        Args:
            instance: ModelInstance to check.
        """

        return (
            (
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.PENDING
            )
            or (
                # Reschedule while it stays in anayzing state for too long,
                # maybe the server is restarted.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.ANALYZING
                and datetime.now(timezone.utc)
                - instance.updated_at.replace(tzinfo=timezone.utc)
                > timedelta(minutes=3)
            )
            or (
                # Reschedule while it stays in scheduled state for too long,
                # maybe the worker is down.
                instance.worker_id is not None
                and instance.state == ModelInstanceStateEnum.SCHEDULED
                and datetime.now(timezone.utc)
                - instance.updated_at.replace(tzinfo=timezone.utc)
                > timedelta(minutes=3)
            )
        )

    async def _schedule_cycle(self):
        while True:
            try:
                item = await self._queue.get()
                try:
                    await self._schedule_one(item)
                    self._queue.task_done()
                except Exception as e:
                    logger.error(f"Failed to schedule model instance: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Failed to get item from schedule queue: {e}")

    async def _schedule_one(self, instance: ModelInstance):  # noqa: C901
        """
        Schedule a model instance by picking one candidate.
        Args:
            item: Model instance to schedule.
        """
        logger.debug(f"Scheduling model instance {instance.name}")

        state_message = ""
        started_at = time.monotonic()

        async with AsyncSession(self._engine) as session:
            workers = await Worker.all(session)
            if len(workers) == 0:
                state_message = "No available workers"

            model = await Model.one_by_id(session, instance.model_id)
            if model is None:
                state_message = "Model not found"

            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if model_instance is None:
                logger.debug(
                    f"Model instance(ID: {instance.id}) was deleted before scheduling due"
                )
                return

            candidate = None
            candidates = []
            projected_loads = {}
            messages = []
            policy = (
                await SchedulerPolicy.one_by_field(session, "code", "aggregation")
                if model
                else None
            )
            if model and policy is None:
                state_message = "Aggregation scheduler policy not found"
            if workers and model and policy:
                try:
                    scheduling_model = get_model_for_instance_scheduling(
                        model, model_instance
                    )
                    aggregation_rate = None
                    if (
                        policy.enabled
                        and scheduling_model.placement_strategy
                        == PlacementStrategyEnum.BINPACK
                    ):
                        aggregation_rate = policy.aggregation_rate
                    elif (
                        not policy.enabled
                        and scheduling_model.placement_strategy
                        == PlacementStrategyEnum.BINPACK
                    ):
                        scheduling_model.placement_strategy = (
                            PlacementStrategyEnum.SPREAD
                        )
                    candidate, messages, candidates, projected_loads = (
                        await find_candidate_detailed(
                            self._config,
                            scheduling_model,
                            workers,
                            aggregation_rate=aggregation_rate,
                        )
                    )
                except Exception as e:
                    state_message = f"Failed to find candidate: {e}"

            if candidate is None:
                # update model instance.
                if model_instance.state in (
                    ModelInstanceStateEnum.SCHEDULED,
                    ModelInstanceStateEnum.ANALYZING,
                ):
                    model_instance.state = ModelInstanceStateEnum.PENDING
                    model_instance.state_message = (
                        "No suitable workers.\nDetails:\n" + "".join(messages)
                    )
                if state_message != "":
                    model_instance.state_message = state_message

                if model and policy:
                    event = await _build_scheduling_event(
                        session=session,
                        model=model,
                        model_instance=model_instance,
                        policy=policy,
                        candidates=candidates,
                        selected=None,
                        projected_loads=projected_loads,
                        outcome=SchedulingOutcome.FAILED,
                        reason_code="no_suitable_target",
                        reason=model_instance.state_message or "No suitable workers",
                        started_at=started_at,
                    )
                    session.add(event)
                    session.add(model_instance)
                    await session.commit()
                    await ModelInstance._publish_event(
                        EventType.UPDATED, model_instance
                    )
                else:
                    await ModelInstanceService(session).update(model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}, state: {model_instance.state}"
                )
            else:
                # update model instance.
                model_instance.state = ModelInstanceStateEnum.SCHEDULED
                model_instance.state_message = ""
                model_instance.worker_id = candidate.worker.id
                model_instance.worker_name = candidate.worker.name
                model_instance.worker_ip = candidate.worker.ip
                model_instance.computed_resource_claim = (
                    candidate.computed_resource_claim
                )
                model_instance.gpu_indexes = candidate.gpu_indexes
                model_instance.gpu_addresses = candidate.gpu_addresses
                model_instance.distributed_servers = DistributedServers(
                    subordinate_workers=candidate.subordinate_workers,
                )
                if get_backend(model) == BackendEnum.ASCEND_MINDIE:
                    model_instance.distributed_servers.mode = (
                        DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    )
                elif is_gguf_backend(get_backend(model)):
                    model_instance.distributed_servers.download_model_files = False

                event = await _build_scheduling_event(
                    session=session,
                    model=model,
                    model_instance=model_instance,
                    policy=policy,
                    candidates=candidates,
                    selected=candidate,
                    projected_loads=projected_loads,
                    outcome=SchedulingOutcome.SUCCESS,
                    reason_code="target_selected",
                    reason="Scheduler selected a target that satisfies the active policy",
                    started_at=started_at,
                )
                session.add(event)
                session.add(model_instance)
                await session.commit()
                await ModelInstance._publish_event(EventType.UPDATED, model_instance)

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )


async def find_candidate(
    config: Config,
    model: Model,
    workers: List[Worker],
) -> Tuple[Optional[ModelInstanceScheduleCandidate], List[str]]:
    """
    Find a schedule candidate for the model instance.
    :param config: GPUStack configuration.
    :param model: Model to schedule.
    :param workers: List of workers to consider.
    :return: A tuple containing:
                - The schedule candidate.
                - A list of messages for the scheduling process.
    """
    candidate, messages, _, _ = await find_candidate_detailed(config, model, workers)
    return candidate, messages


async def find_candidate_detailed(
    config: Config,
    model: Model,
    workers: List[Worker],
    aggregation_rate: Optional[float] = None,
) -> Tuple[
    Optional[ModelInstanceScheduleCandidate],
    List[str],
    List[ModelInstanceScheduleCandidate],
    dict,
]:
    filters = [
        GPUMatchingFilter(model),
        LabelMatchingFilter(model),
        StatusFilter(model),
    ]

    worker_filter_chain = WorkerFilterChain(filters)
    workers, filter_messages = await worker_filter_chain.filter(workers)
    messages = []
    if filter_messages:
        messages.append(str(ListMessageBuilder(filter_messages)) + "\n")

    try:
        if is_gguf_model(model):
            candidates_selector = GGUFResourceFitSelector(model, config.cache_dir)
        elif is_audio_model(model):
            candidates_selector = VoxBoxResourceFitSelector(
                config, model, config.cache_dir
            )
        elif model.backend == BackendEnum.ASCEND_MINDIE:
            candidates_selector = AscendMindIEResourceFitSelector(config, model)
        else:
            candidates_selector = VLLMResourceFitSelector(config, model)
    except Exception as e:
        return (
            None,
            [f"Failed to initialize {model.backend} candidates selector: {e}"],
            [],
            {},
        )

    candidates = await candidates_selector.select_candidates(workers)

    projected_loads = {}
    evaluated_candidates = candidates
    if aggregation_rate is not None:
        candidates, projected_loads = await filter_by_aggregation_rate(
            get_engine(), candidates, workers, aggregation_rate
        )
        if not candidates:
            messages.append(
                f"No candidates remain below aggregation rate {aggregation_rate:.2f}%."
            )

    placement_scorer = PlacementScorer(model)
    candidates = await placement_scorer.score(candidates)

    candidate = pick_highest_score_candidate(candidates)

    if candidate is None and len(workers) > 0:
        resource_fit_messages = candidates_selector.get_messages() or [
            "No workers meet the resource requirements."
        ]
        messages.extend(resource_fit_messages)
    elif candidate and candidate.overcommit:
        messages.extend(candidates_selector.get_messages())

    return candidate, messages, evaluated_candidates, projected_loads


async def _build_scheduling_event(
    session,
    model: Model,
    model_instance: ModelInstance,
    policy: SchedulerPolicy,
    candidates: List[ModelInstanceScheduleCandidate],
    selected: Optional[ModelInstanceScheduleCandidate],
    projected_loads: dict,
    outcome: SchedulingOutcome,
    reason_code: str,
    reason: str,
    started_at: float,
) -> SchedulingAttemptEvent:
    workload_id = str(
        (model.meta or {}).get("model_deploy_id") or f"gpustack-model-{model.id}"
    )
    attempt_statement = select(func.max(SchedulingAttemptEvent.attempt_no)).where(
        SchedulingAttemptEvent.workload_id == workload_id
    )
    previous_attempt = (await session.exec(attempt_statement)).scalar_one_or_none() or 0
    candidate_targets = [
        candidate_snapshot(item, projected_loads.get(id(item))) for item in candidates
    ]
    selected_targets = (
        [candidate_snapshot(selected, projected_loads.get(id(selected)))]
        if selected
        else []
    )
    resource_candidate = selected or (candidates[0] if candidates else None)
    requested_resources = (
        resource_candidate.computed_resource_claim.model_dump(mode="json")
        if resource_candidate
        else {}
    )
    return SchedulingAttemptEvent(
        event_id=str(uuid.uuid4()),
        workload_id=workload_id,
        attempt_no=previous_attempt + 1,
        policy_code=(
            "aggregation"
            if policy.enabled
            and model.placement_strategy == PlacementStrategyEnum.BINPACK
            else str(model.placement_strategy.value)
        ),
        policy_revision=policy.runtime_revision,
        requested_replicas=model.replicas,
        requested_resources=requested_resources,
        candidate_targets=candidate_targets,
        selected_targets=selected_targets,
        outcome=outcome,
        reason_code=reason_code,
        reason=reason,
        latency_ms=max(int((time.monotonic() - started_at) * 1000), 0),
        trace_id=None,
        occurred_at=datetime.now(timezone.utc),
    )


def pick_highest_score_candidate(candidates: List[ModelInstanceScheduleCandidate]):
    """
    Pick the most offload layers from candidates.
    Args:
        candidates: List of ModelInstanceScheduleCandidate.
    """

    logger.debug(f"Pick highest score candidate from {len(candidates)} candidates")

    if len(candidates) == 0:
        return None

    candidate = candidates[0]
    for i in range(1, len(candidates)):
        if candidates[i].score > candidate.score:
            candidate = candidates[i]

    return candidate


async def evaluate_gguf_model(
    config: Config,
    model: Model,
) -> bool:
    task_output = await calculate_model_resource_claim(
        model,
        offload=GPUOffloadEnum.Full,
        cache_dir=config.cache_dir,
        ollama_library_base_url=config.ollama_library_base_url,
    )
    if (
        task_output.resource_architecture
        and not task_output.resource_architecture.is_deployable()
    ):
        raise ValueError("Not a supported model.")

    should_update = False
    if task_output.resource_claim_estimate.reranking and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.RERANKER]
        model.reranker = True

    if task_output.resource_claim_estimate.embeddingOnly and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.EMBEDDING]
        model.embedding_only = True

    if task_output.resource_claim_estimate.imageOnly and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.IMAGE]
        model.image_only = True

    if not model.categories:
        should_update = True
        model.categories = [CategoryEnum.LLM]

    if task_output.resource_claim_estimate.distributable and not model.distributable:
        should_update = True
        model.distributable = True

    if model.gpu_selector and model.gpu_selector.gpu_ids:
        worker_gpu_ids = parse_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
        if (
            len(worker_gpu_ids) > 1
            and model.distributable
            and not model.distributed_inference_across_workers
        ):
            should_update = True
            model.distributed_inference_across_workers = True

    return should_update


async def evaluate_audio_model(
    config: Config,
    model: Model,
) -> bool:
    try:
        from vox_box.estimator.estimate import estimate_model
        from vox_box.config import Config as VoxBoxConfig
    except ImportError:
        raise Exception("vox_box is not installed.")

    cfg = VoxBoxConfig()
    cfg.cache_dir = os.path.join(config.cache_dir, "vox-box")
    cfg.model = model.local_path
    cfg.huggingface_repo_id = model.huggingface_repo_id
    cfg.model_scope_model_id = model.model_scope_model_id

    try:
        timeout_in_seconds = 15
        model_dict = await asyncio.wait_for(
            asyncio.to_thread(estimate_model, cfg),
            timeout=timeout_in_seconds,
        )
    except Exception as e:
        raise Exception(
            f"Failed to estimate model {model.name or model.readable_source}: {e}"
        )

    supported = model_dict.get("supported", False)
    if not supported:
        raise ValueError("Not a supported model.")

    should_update = False
    task_type = model_dict.get("task_type")
    if task_type == "tts" and not model.categories:
        model.categories = [CategoryEnum.TEXT_TO_SPEECH]
        model.text_to_speech = True
        should_update = True
    elif task_type == "stt" and not model.categories:
        model.categories = [CategoryEnum.SPEECH_TO_TEXT]
        model.speech_to_text = True
        should_update = True

    return should_update


async def evaluate_pretrained_config(model: Model, raise_raw: bool = False) -> bool:
    """
    evaluate the model's pretrained config to determine its type.
    Args:
        model: Model to evaluate.
        raise_raw: If True, raise the raw exception.
    Returns:
        True if the model's categories are updated, False otherwise.
    """
    # Check overrided architectures if specified in backend parameters.
    architectures = get_vllm_override_architectures(model)
    if not architectures:
        try:
            pretrained_config = await run_in_thread(
                get_pretrained_config, timeout=30, model=model
            )
        except ValueError as e:
            # Skip value error exceptions and defaults to LLM catagory for certain cases.
            if should_skip_architecture_check(model):
                model.categories = model.categories or [CategoryEnum.LLM]
                return True

            if raise_raw:
                raise

            logger.debug(
                f"Failed to get config for model {model.name or model.readable_source}: {e}"
            )
            raise simplify_auto_config_value_error(e)
        except TimeoutError:
            raise Exception(
                f"Timeout while getting config for model {model.name or model.readable_source}."
            )
        except Exception as e:
            raise Exception(
                f"Failed to get config for model {model.name or model.readable_source}: {e}"
            )

        architectures = getattr(pretrained_config, "architectures", []) or []
        if not architectures and not model.backend_version:
            raise ValueError("Not a supported model. Unrecognized architecture.")

    model_type = detect_model_type(architectures)

    if model_type == CategoryEnum.UNKNOWN and not model.backend_version:
        raise ValueError(
            f"Not a supported model. Detected architectures: {architectures}."
        )

    return set_model_categories(model, model_type)


def get_vllm_override_architectures(model: Model) -> List[str]:
    """
    Get the vLLM override architectures from the model's backend parameters.
    Args:
        model: Model to check.
    Returns:
        List of override architectures.
    """
    backend = get_backend(model)
    if backend != BackendEnum.VLLM:
        return []

    hf_overrides = find_parameter(model.backend_parameters, ["hf-overrides"])
    if hf_overrides:
        overrides_dict = json.loads(hf_overrides)
        return overrides_dict.get("architectures", [])
    return []


def should_skip_architecture_check(model: Model) -> bool:
    """
    Check if the model should skip architecture check.
    Args:
        model: Model to check.
    Returns:
        True if the model should skip architecture check, False otherwise.
    """

    if model.backend_version:
        # New model architectures may be added with custom backend version.
        return True

    if model.backend == BackendEnum.VLLM_OMNI:
        # vllm-omni supports diffusion models (Flux, Z-Image, etc.) that use diffusers-format
        # configs without standard transformers architecture fields.
        return True

    if model.backend_parameters and find_parameter(
        model.backend_parameters, ["tokenizer-mode"]
    ):
        # Models like Pixtral may not provide compatible config but still work with custom parameters.
        return True

    return False


def simplify_auto_config_value_error(e: ValueError) -> ValueError:
    """
    Simplify the error message for ValueError exceptions.
    """
    message = str(e)
    if "trust_remote_code=True" in message:
        return ValueError(
            "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run."
        )
    return ValueError("Not a supported model.")


def set_model_categories(model: Model, model_type: CategoryEnum) -> bool:
    if model.categories:
        return False

    if model_type == CategoryEnum.EMBEDDING:
        model.categories = [CategoryEnum.EMBEDDING]
        model.embedding_only = True
        return True
    elif model_type == CategoryEnum.RERANKER:
        model.categories = [CategoryEnum.RERANKER]
        model.reranker = True
        return True
    elif model_type == CategoryEnum.LLM:
        model.categories = [CategoryEnum.LLM]
        return True
    elif model_type == CategoryEnum.UNKNOWN:
        # Default to LLM for unknown architectures
        model.categories = [CategoryEnum.LLM]
        return True

    return False


def detect_model_type(architectures: List[str]) -> CategoryEnum:
    """
    Detect the model type based on the architectures.
    """
    for architecture in architectures:
        if architecture in vllm_supported_embedding_architectures:
            return CategoryEnum.EMBEDDING
        if architecture in vllm_supported_reranker_architectures:
            return CategoryEnum.RERANKER
        if architecture in vllm_supported_llm_architectures:
            return CategoryEnum.LLM
    return CategoryEnum.UNKNOWN
