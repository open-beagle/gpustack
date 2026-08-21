import asyncio

from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes.model_files import (
    create_model_file,
    delete_model_file,
    get_model_file,
    update_model_file,
)
from gpustack.schemas.links import ModelInstanceModelFileLink
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_storage_sync import (
    ModelFileTransferSourceEnum,
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.model_files import (
    ModelFile,
    ModelFileCreate,
    ModelFileStateEnum,
    ModelFileUpdate,
)
from gpustack.schemas.models import Model, ModelInstance, SourceEnum
from gpustack.schemas.users import User
from gpustack.schemas.workers import Worker


def test_model_storage_sync_task_is_owned_by_model_file():
    foreign_key = next(
        foreign_key
        for foreign_key in ModelStorageSyncTask.__table__.foreign_keys
        if foreign_key.parent.name == "model_file_id"
    )

    assert foreign_key.target_fullname == "model_files.id"
    assert foreign_key.ondelete == "CASCADE"


def test_model_file_crud_cascades_sync_tasks(tmp_path):
    asyncio.run(_run_model_file_crud(tmp_path))


async def _run_model_file_crud(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'model-files.db'}",
        poolclass=NullPool,
    )
    event.listen(
        engine.sync_engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )

    tables = [
        Worker.__table__,
        User.__table__,
        Model.__table__,
        ModelInstance.__table__,
        ModelFile.__table__,
        ModelInstanceModelFileLink.__table__,
        ModelPreheatS3Profile.__table__,
        ModelStorageSyncTask.__table__,
    ]
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all, tables=tables)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-a-uuid",
        )
        worker = await Worker.create(session, worker)

        created = await create_model_file(
            session,
            ModelFileCreate(
                source=SourceEnum.LOCAL_PATH,
                local_path="/models/original.gguf",
                worker_id=worker.id,
            ),
        )
        fetched = await get_model_file(session, created.id)
        assert fetched.local_path == "/models/original.gguf"
        assert fetched.state == ModelFileStateEnum.DOWNLOADING

        update = ModelFileUpdate(**fetched.model_dump())
        update.state = ModelFileStateEnum.READY
        update.resolved_paths = ["/models/original.gguf"]
        updated = await update_model_file(session, fetched.id, update)
        assert updated.state == ModelFileStateEnum.READY
        assert updated.resolved_paths == ["/models/original.gguf"]

        profile = ModelPreheatS3Profile(
            name="center-cache",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={},
            secret_key_encrypted={},
            encryption_key_version="v1",
        )
        profile = await ModelPreheatS3Profile.create(session, profile)
        sync_task = ModelStorageSyncTask(
            model_file_id=created.id,
            worker_id=worker.id,
            worker_uuid=worker.worker_uuid,
            profile_id=profile.id,
            profile_config_version=1,
            request_identity={},
            request_digest="d" * 64,
            source="modelscope",
            model_id="example/model",
            resolved_revision="sha",
            credential_snapshot_encrypted={},
            encryption_key_version="v1",
            state=ModelStorageSyncTaskStateEnum.READY,
            transfer_source=ModelFileTransferSourceEnum.S3,
            transfer_profile_id=profile.id,
            source_worker_id=worker.id,
        )
        sync_task = await ModelStorageSyncTask.create(session, sync_task)

        await delete_model_file(session, created.id)

    async with AsyncSession(engine) as session:
        assert await ModelFile.one_by_id(session, created.id) is None
        assert await ModelStorageSyncTask.one_by_id(session, sync_task.id) is None

    await engine.dispose()
