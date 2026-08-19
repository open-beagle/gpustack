import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_cache import ModelCacheTask, ModelCacheTaskStateEnum
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.model_preheat_trusted_local import (
    ProductionLocalInventoryProbe,
    trusted_local_candidate_for_worker,
)


async def _database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'trusted.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed(session):
    workers = []
    for index in range(10):
        worker = Worker(
            name=f"worker-{index}",
            hostname=f"worker-{index}",
            ip=f"127.0.0.{index + 1}",
            port=10150,
            worker_uuid=f"worker-{index}",
            state=WorkerStateEnum.READY,
        )
        session.add(worker)
        workers.append(worker)
    await session.flush()
    task = ModelPreheatTask(
        source="modelscope",
        model_id="Qwen/Test",
        resolved_revision="commit-1",
        include_patterns=[],
        exclude_patterns=[],
        selection_digest="selection",
        cache_key="cache-key",
        generation_id="preheat-00000000-0000-4000-8000-000000000001",
        target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
        target_worker_uuids=[worker.worker_uuid for worker in workers],
        target_worker_snapshot=[],
        s3_profile_id=1,
        s3_profile_config_version=1,
        s3_profile_snapshot_encrypted={"ciphertext": "encrypted"},
        encryption_key_version="v1",
        s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
    )
    session.add(task)
    await session.flush()
    return task, workers


def test_probe_finds_ready_model_file_on_any_of_ten_workers(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine) as session:
            task, workers = await _seed(session)
            session.add(
                ModelFile(
                    source=SourceEnum.MODEL_SCOPE,
                    model_scope_model_id="Qwen/Test",
                    worker_id=workers[7].id,
                    resolved_paths=["/models/Qwen/Test"],
                    state=ModelFileStateEnum.READY,
                )
            )
            task_id = task.id
            await session.commit()
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
        results = await ProductionLocalInventoryProbe(engine).probe(
            task, [f"worker-{index}" for index in range(10)]
        )
        await engine.dispose()
        return results

    results = asyncio.run(run())
    assert results["worker-7"].state == "candidate"
    assert results["worker-7"].source == "model_file"
    assert results["worker-7"].paths == ("/models/Qwen/Test",)
    assert results["worker-7"].repository_complete is True
    assert sum(result.state == "candidate" for result in results.values()) == 1


def test_archive_candidate_is_worker_scoped_and_preserves_source_paths(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine) as session:
            task, workers = await _seed(session)
            model_file = ModelFile(
                source=SourceEnum.MODEL_SCOPE,
                model_scope_model_id="Qwen/Test",
                worker_id=workers[3].id,
                resolved_paths=["/models/Qwen/Test"],
                state=ModelFileStateEnum.READY,
            )
            session.add(model_file)
            await session.flush()
            session.add(
                ModelCacheTask(
                    model_file_id=model_file.id,
                    worker_id=workers[3].id,
                    model_id="Qwen/Test",
                    target_path="archive/Qwen/Test",
                    source_paths=["/archive-source/Qwen/Test"],
                    state=ModelCacheTaskStateEnum.READY,
                )
            )
            task_id = task.id
            worker_ids = [worker.id for worker in workers]
            await session.commit()
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
            own = await trusted_local_candidate_for_worker(
                session, task, "worker-3", worker_ids[3]
            )
            other = await trusted_local_candidate_for_worker(
                session, task, "worker-4", worker_ids[4]
            )
        await engine.dispose()
        return own, other

    own, other = asyncio.run(run())
    assert own.source == "model_archive"
    assert own.paths == ("/archive-source/Qwen/Test",)
    assert own.repository_complete is True
    assert other is None


def test_single_file_candidate_is_not_repository_complete(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine) as session:
            task, workers = await _seed(session)
            session.add(
                ModelFile(
                    source=SourceEnum.MODEL_SCOPE,
                    model_scope_model_id="Qwen/Test",
                    model_scope_file_path="weights/model.bin",
                    worker_id=workers[0].id,
                    resolved_paths=["/models/Qwen/Test/weights/model.bin"],
                    state=ModelFileStateEnum.READY,
                )
            )
            task_id = task.id
            worker_id = workers[0].id
            await session.commit()
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
            candidate = await trusted_local_candidate_for_worker(
                session, task, "worker-0", worker_id
            )
        await engine.dispose()
        return candidate

    candidate = asyncio.run(run())
    assert candidate.repository_complete is False
    assert candidate.root == "/models/Qwen/Test"


def test_probe_ignores_unsupported_ready_model_sources(tmp_path):
    async def run():
        engine = await _database(tmp_path)
        async with AsyncSession(engine) as session:
            task, workers = await _seed(session)
            session.add(
                ModelFile(
                    source=SourceEnum.OLLAMA_LIBRARY,
                    ollama_library_model_name="Qwen/Test",
                    worker_id=workers[0].id,
                    resolved_paths=["/models/ollama/Qwen/Test"],
                    state=ModelFileStateEnum.READY,
                )
            )
            session.add(
                ModelFile(
                    source=SourceEnum.LOCAL_PATH,
                    local_path="Qwen/Test",
                    worker_id=workers[0].id,
                    resolved_paths=["/models/local/Qwen/Test"],
                    state=ModelFileStateEnum.READY,
                )
            )
            task_id = task.id
            await session.commit()
        async with AsyncSession(engine) as session:
            task = await session.get(ModelPreheatTask, task_id)
        result = await ProductionLocalInventoryProbe(engine).probe(task, ["worker-0"])
        await engine.dispose()
        return result

    assert asyncio.run(run())["worker-0"].state == "missing"
