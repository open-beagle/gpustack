from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import MissingGreenlet
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from tests.utils.model import new_model
from tests.fixtures.workers.fixtures import macos_metal_1_m1pro_21g
from gpustack.scheduler.evaluator import (
    evaluate_environment,
    set_default_worker_selector,
)
from gpustack.scheduler import scheduler as scheduler_module
from gpustack.scheduler.scheduler import Scheduler, evaluate_pretrained_config
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.schemas.scheduler import SchedulingOutcome


class ExpiringModelInstance(SimpleNamespace):
    """模拟 SQLAlchemy 提交后需要异步刷新的实例。"""

    def __getattribute__(self, name):
        if name == "name" and object.__getattribute__(self, "expired"):
            raise RuntimeError("提交后的实例属性已过期")
        return object.__getattribute__(self, name)


class ExpiringSession:
    def __init__(self, instance):
        self.instance = instance
        self.added = []
        self.refreshed = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    def add(self, record):
        self.added.append(record)

    async def commit(self):
        self.committed = True
        self.instance.expired = True

    async def refresh(self, record):
        assert self.committed
        assert record is self.instance
        self.refreshed.append(record)
        record.expired = False


@pytest.mark.asyncio
async def test_expire_on_commit_requires_refresh_before_instance_attribute_access(
    tmp_path,
):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'scheduler-expire-on-commit.db'}"
    )
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    Model.__table__,
                    ModelInstance.__table__,
                    ModelFile.__table__,
                    ModelInstanceModelFileLink.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            model = Model(
                name="scheduler-model",
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/scheduler-model",
            )
            instance = ModelInstance(
                name="scheduler-model-1",
                model_id=1,
                model_name="scheduler-model",
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/scheduler-model",
            )
            session.add(model)
            await session.flush()
            instance.model_id = model.id
            session.add(instance)
            await session.commit()
            instance_id = instance.id

        async with AsyncSession(engine, expire_on_commit=True) as session:
            instance = await ModelInstance.one_by_id(session, instance_id)
            instance.state = ModelInstanceStateEnum.SCHEDULED
            await session.commit()

            with pytest.raises(MissingGreenlet):
                _ = instance.name

            await session.refresh(instance)
            assert instance.name == "scheduler-model-1"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_candidate", [False, True])
async def test_schedule_one_refreshes_expired_instance_before_publishing_event(
    has_candidate,
):
    model = new_model(101, "model-a")
    instance = ExpiringModelInstance(
        id=202,
        name="model-a-1",
        model_id=model.id,
        state=ModelInstanceStateEnum.SCHEDULED,
        state_message="",
        expired=False,
    )
    session = ExpiringSession(instance)
    event = SimpleNamespace(name="scheduling-attempt")
    policy = SimpleNamespace(enabled=True, runtime_revision=1, aggregation_rate=80)
    worker = SimpleNamespace(id=7, name="worker-a", ip="10.0.0.7")
    candidate = SimpleNamespace(
        worker=worker,
        computed_resource_claim=SimpleNamespace(),
        gpu_indexes=[0],
        gpu_addresses=["GPU-0"],
        subordinate_workers=[],
    )
    published = []

    async def publish_event(event_type, record):
        assert session.refreshed == [instance]
        published.append((event_type, record.name))

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine = object()
    scheduler._config = SimpleNamespace()

    with (
        patch.object(
            scheduler_module,
            "AsyncSession",
            return_value=session,
        ),
        patch.object(
            scheduler_module.Worker,
            "all",
            AsyncMock(return_value=[worker] if has_candidate else []),
        ),
        patch.object(
            scheduler_module.Model,
            "one_by_id",
            AsyncMock(return_value=model),
        ),
        patch.object(
            scheduler_module.ModelInstance,
            "one_by_id",
            AsyncMock(return_value=instance),
        ),
        patch.object(
            scheduler_module.SchedulerPolicy,
            "one_by_field",
            AsyncMock(return_value=policy),
        ),
        patch.object(
            scheduler_module,
            "_build_scheduling_event",
            AsyncMock(return_value=event),
        ) as build_event,
        patch.object(
            scheduler_module.ModelInstance,
            "_publish_event",
            new=publish_event,
        ),
        patch.object(
            scheduler_module,
            "find_candidate_detailed",
            AsyncMock(
                return_value=(
                    (candidate, [], [candidate], {})
                    if has_candidate
                    else (None, [], [], {})
                )
            ),
        ),
        patch.object(
            scheduler_module,
            "get_model_for_instance_scheduling",
            return_value=model,
        ),
        patch.object(
            scheduler_module,
            "get_backend",
            return_value=BackendEnum.VLLM,
        ),
        patch.object(scheduler_module, "is_gguf_backend", return_value=False),
    ):
        await scheduler._schedule_one(
            SimpleNamespace(id=instance.id, name="queued-model-a-1", model_id=model.id)
        )

    assert session.added == [event, instance]
    assert session.refreshed == [instance]
    assert published == [(scheduler_module.EventType.UPDATED, "model-a-1")]
    assert build_event.await_args.kwargs["outcome"] == (
        SchedulingOutcome.SUCCESS if has_candidate else SchedulingOutcome.FAILED
    )
    assert instance.state == (
        ModelInstanceStateEnum.SCHEDULED
        if has_candidate
        else ModelInstanceStateEnum.PENDING
    )


@pytest.mark.asyncio
async def test_vllm_omni_requires_linux_workers(config):
    model = new_model(
        1,
        "qwen-image",
        1,
        model_scope_model_id="Qwen/Qwen-Image",
        backend_parameters=[],
    )
    model.backend = BackendEnum.VLLM_OMNI

    is_compatible, messages = await evaluate_environment(
        model, [macos_metal_1_m1pro_21g()]
    )

    assert is_compatible is False
    assert "requires Linux workers" in messages[0]


def test_set_default_worker_selector_for_vllm_omni():
    model = new_model(
        1,
        "qwen-image",
        1,
        model_scope_model_id="Qwen/Qwen-Image",
        backend_parameters=[],
    )
    model.backend = BackendEnum.VLLM_OMNI

    set_default_worker_selector(model)

    assert model.worker_selector == {"os": "linux"}


@pytest.mark.asyncio
async def test_evaluate_pretrained_config(config):
    Phi_4_multimodal = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="microsoft/Phi-4-multimodal-instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # Custom code without --trust-remote-code, should raise ValueError
    with pytest.raises(
        ValueError,
        match="The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run.",
    ):
        await evaluate_pretrained_config(Phi_4_multimodal)

    # Custom code with --trust-remote-code, should not raise ValueError
    Phi_4_multimodal.backend_parameters = ["--trust-remote-code"]
    await evaluate_pretrained_config(Phi_4_multimodal)
    assert Phi_4_multimodal.categories == [CategoryEnum.LLM]

    t5 = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="google-t5/t5-base",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # Model architecture not supported, should raise ValueError
    with pytest.raises(ValueError, match="Not a supported model"):
        await evaluate_pretrained_config(t5)

    qwen = new_model(
        1,
        "test_name",
        1,
        huggingface_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend=BackendEnum.VLLM,
        backend_parameters=[],
    )

    # Model architecture supported, should not raise ValueError
    await evaluate_pretrained_config(qwen)
    assert qwen.categories == [CategoryEnum.LLM]
