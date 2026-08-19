import ntpath
import posixpath
from dataclasses import dataclass

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_cache import ModelCacheTask, ModelCacheTaskStateEnum
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


@dataclass(frozen=True)
class ProductionLocalInventoryProbeResult:
    worker_uuid: str
    state: str
    error_code: str | None = None
    source: str | None = None
    root: str | None = None
    paths: tuple[str, ...] = ()


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

    model_file_ids = [model_file.id for model_file in matching_files]
    archives = (
        await session.exec(
            select(ModelCacheTask)
            .where(
                ModelCacheTask.worker_id == worker_id,
                ModelCacheTask.model_file_id.in_(model_file_ids),
                ModelCacheTask.state == ModelCacheTaskStateEnum.READY,
            )
            .order_by(ModelCacheTask.id.desc())
        )
    ).all()
    for archive in archives:
        model_file = next(
            item for item in matching_files if item.id == archive.model_file_id
        )
        candidate = _record(
            worker_uuid,
            worker_id,
            "model_archive",
            model_file,
            archive.source_paths,
        )
        if candidate is not None:
            return candidate

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
    return TrustedLocalCandidateRecord(worker_uuid, worker_id, source, root, paths)


def _candidate_root(model_file, paths):
    path_module = ntpath if any("\\" in path for path in paths) else posixpath
    if model_file.local_dir:
        return path_module.normpath(model_file.local_dir)
    if len(paths) == 1:
        if model_file.huggingface_filename or model_file.model_scope_file_path:
            return path_module.dirname(path_module.normpath(paths[0]))
        return path_module.normpath(paths[0])
    try:
        return path_module.commonpath(paths)
    except ValueError:
        return None
