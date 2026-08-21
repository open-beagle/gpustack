import ntpath
import posixpath
from dataclasses import dataclass
from glob import has_magic

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentityError,
    normalize_source,
)


@dataclass(frozen=True)
class TrustedLocalCandidateRecord:
    worker_uuid: str
    worker_id: int
    source: str
    root: str
    paths: tuple[str, ...]
    repository_complete: bool


@dataclass(frozen=True)
class ProductionLocalInventoryProbeResult:
    worker_uuid: str
    state: str
    error_code: str | None = None
    source: str | None = None
    root: str | None = None
    paths: tuple[str, ...] = ()
    repository_complete: bool = False


class ProductionLocalInventoryProbe:
    def __init__(self, engine):
        self._engine = engine

    async def probe(self, task, worker_uuids):
        async with AsyncSession(self._engine) as session:
            workers = (
                await session.exec(
                    select(Worker).where(Worker.worker_uuid.in_(worker_uuids))
                )
            ).all()
            by_uuid = {worker.worker_uuid: worker for worker in workers}
            results = {}
            for worker_uuid in worker_uuids:
                worker = by_uuid.get(worker_uuid)
                candidate = (
                    await trusted_local_candidate_for_worker(
                        session, task, worker_uuid, worker.id
                    )
                    if worker is not None
                    else None
                )
                results[worker_uuid] = ProductionLocalInventoryProbeResult(
                    worker_uuid=worker_uuid,
                    state="candidate" if candidate is not None else "missing",
                    source=candidate.source if candidate is not None else None,
                    root=candidate.root if candidate is not None else None,
                    paths=candidate.paths if candidate is not None else (),
                    repository_complete=(
                        candidate.repository_complete
                        if candidate is not None
                        else False
                    ),
                )
            return results


async def trusted_local_candidate_for_worker(
    session, task, worker_uuid: str, worker_id: int
) -> TrustedLocalCandidateRecord | None:
    # READY 是模型文件/归档流程建立的本地业务信任边界；Worker 仍会重新扫描并
    # 校验实际文件，它不代表对远端 revision 的额外证明。
    model_files = (
        await session.exec(
            select(ModelFile)
            .where(
                ModelFile.worker_id == worker_id,
                ModelFile.state == ModelFileStateEnum.READY,
            )
            .order_by(ModelFile.id.desc())
        )
    ).all()
    matching_files = [
        model_file
        for model_file in model_files
        if model_file.resolved_paths and _matches_task(model_file, task)
    ]
    if not matching_files:
        return None

    for model_file in matching_files:
        candidate = _record(
            worker_uuid,
            worker_id,
            "model_file",
            model_file,
            model_file.resolved_paths,
        )
        if candidate is not None:
            return candidate
    return None


def _matches_task(model_file, task) -> bool:
    source = (
        model_file.source.value
        if hasattr(model_file.source, "value")
        else str(model_file.source)
    )
    if source not in {
        SourceEnum.HUGGING_FACE.value,
        SourceEnum.MODEL_SCOPE.value,
    }:
        return False
    try:
        normalized_source = normalize_source(source)
        normalized_task_source = normalize_source(task.source)
    except ModelPreheatIdentityError:
        return False
    if normalized_source != normalized_task_source:
        return False
    if source == SourceEnum.HUGGING_FACE.value:
        return model_file.huggingface_repo_id == task.model_id
    if source == SourceEnum.MODEL_SCOPE.value:
        return model_file.model_scope_model_id == task.model_id
    return False


def _record(worker_uuid, worker_id, source, model_file, raw_paths):
    paths = tuple(
        dict.fromkeys(path for path in raw_paths if isinstance(path, str) and path)
    )
    if not paths:
        return None
    root = _candidate_root(model_file, paths)
    if not root:
        return None
    repository_complete = (
        model_file.huggingface_filename is None
        and model_file.model_scope_file_path is None
    )
    return TrustedLocalCandidateRecord(
        worker_uuid,
        worker_id,
        source,
        root,
        paths,
        repository_complete,
    )


def _candidate_root(model_file, paths):
    path_module = ntpath if any("\\" in path for path in paths) else posixpath
    if model_file.local_dir:
        return path_module.normpath(model_file.local_dir)
    if len(paths) == 1:
        selected_path = (
            model_file.huggingface_filename or model_file.model_scope_file_path
        )
        if selected_path:
            if not has_magic(selected_path):
                return _repository_root_for_exact_path(
                    path_module, paths[0], selected_path
                )
            return path_module.dirname(path_module.normpath(paths[0]))
        return path_module.normpath(paths[0])
    try:
        return path_module.commonpath(paths)
    except ValueError:
        return None


def _repository_root_for_exact_path(path_module, resolved_path, selected_path):
    normalized_selected = path_module.normpath(selected_path)
    if (
        path_module.isabs(normalized_selected)
        or normalized_selected in {"", ".", ".."}
        or normalized_selected.startswith(f"..{path_module.sep}")
    ):
        return None
    remaining = path_module.normpath(resolved_path)
    for component in reversed(normalized_selected.split(path_module.sep)):
        if path_module.basename(remaining) != component:
            return None
        remaining = path_module.dirname(remaining)
    return remaining or None
