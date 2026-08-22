import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_file_download_executions
from gpustack.schemas.model_file_download_executions import ModelFileDownloadExecution
from gpustack.schemas.model_files import ModelFile, ModelFilePublic
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3Profile,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_session
from gpustack.server.model_file_download_execution_service import (
    create_model_file_with_download_execution,
)
from gpustack.server.model_preheat_worker_identity import (
    get_model_preheat_worker_identity,
)


SHA = "a" * 40


def _app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'download-executions.db'}",
        poolclass=NullPool,
    )
    asyncio.run(_create_tables(engine))
    key = generate_model_preheat_credential_key()
    app = FastAPI()
    app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=key,
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        huggingface_token=None,
    )
    app.state.model_file_download_revision_resolver = (
        lambda source, model_id, revision, token=None: SHA
    )

    async def session_override():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    async def identity_override():
        return SimpleNamespace(worker_id=1, worker_uuid="worker-uuid")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_model_preheat_worker_identity] = identity_override
    app.include_router(model_file_download_executions.router, prefix="/v1/model-files")
    exceptions.register_handlers(app)
    return app, engine, key


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _seed(engine, key, *, with_profile=True):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        worker = Worker(
            name="worker-a",
            hostname="worker-a",
            ip="127.0.0.1",
            port=10150,
            worker_uuid="worker-uuid",
            model_storage_protocol_version=1,
        )
        session.add(worker)
        await session.commit()
        await session.refresh(worker)
        if with_profile:
            cipher = ModelPreheatCredentialCipher(key, "v1")
            profile = ModelPreheatS3Profile(
                name="default",
                endpoint="https://s3.example.com",
                bucket="models",
                prefix="storage",
                access_key_encrypted=cipher.encrypt("access-value"),
                secret_key_encrypted=cipher.encrypt("secret-value"),
                encryption_key_version="v1",
                default_slot=DEFAULT_SLOT_GLOBAL,
                source_fallback_enabled=False,
            )
            session.add(profile)
            await session.commit()
        model_file = ModelFile(
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="org/model",
            requested_revision="main",
            worker_id=worker.id,
            source_index="hf:org/model",
        )
        config = SimpleNamespace(
            model_preheat_credential_key=key,
            model_preheat_credential_key_version="v1",
            model_preheat_credential_old_keys=None,
        )
        model_file = await create_model_file_with_download_execution(
            session, model_file, config
        )
        execution = (
            await session.exec(
                select(ModelFileDownloadExecution).where(
                    ModelFileDownloadExecution.model_file_id == model_file.id
                )
            )
        ).one()
        return worker.id, model_file.id, execution.id


def test_claim_is_private_fixed_and_retryable(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))
    client = TestClient(app)

    first = client.post(f"/v1/model-files/{model_file_id}/download-executions/claim")
    second = client.post(f"/v1/model-files/{model_file_id}/download-executions/claim")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["cache-control"] == "no-store"
    payload = first.json()
    assert payload["resolved_revision"] == SHA
    assert payload["source_fallback_enabled"] is False
    assert payload["profile"]["access_key"] == "access-value"
    assert payload["profile"]["secret_key"] == "secret-value"

    async def inspect():
        async with AsyncSession(engine) as session:
            model_file = await session.get(ModelFile, model_file_id)
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            public = ModelFilePublic.model_validate(model_file).model_dump()
            return execution, public

    execution, public = asyncio.run(inspect())
    assert execution.credential_snapshot_encrypted is not None
    assert "credential" not in public
    assert "access-value" not in str(public)
    assert "secret-value" not in str(public)
    asyncio.run(engine.dispose())


def test_claim_rejects_wrong_worker_and_stale_registration(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))

    async def wrong_identity():
        return SimpleNamespace(worker_id=999, worker_uuid="other-worker")

    app.dependency_overrides[get_model_preheat_worker_identity] = wrong_identity
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 403

    async def add_replacement():
        async with AsyncSession(engine) as session:
            session.add(
                Worker(
                    name="replacement",
                    hostname="replacement",
                    ip="127.0.0.2",
                    port=10150,
                    worker_uuid="worker-uuid",
                    model_storage_protocol_version=1,
                )
            )
            await session.commit()

    asyncio.run(add_replacement())

    async def old_identity():
        return SimpleNamespace(worker_id=1, worker_uuid="worker-uuid")

    app.dependency_overrides[get_model_preheat_worker_identity] = old_identity
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 403
    assert response.json()["message"] == "worker_not_current"
    asyncio.run(engine.dispose())


def test_claim_rejects_model_file_no_longer_owned_by_worker(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))

    async def detach():
        async with AsyncSession(engine) as session:
            model_file = await session.get(ModelFile, model_file_id)
            model_file.worker_id = None
            session.add(model_file)
            await session.commit()

    asyncio.run(detach())
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 403
    asyncio.run(engine.dispose())


def test_no_default_profile_creates_explicit_source_fallback_execution(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key, with_profile=False))
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200
    assert response.json()["profile"] is None
    assert response.json()["source_fallback_enabled"] is True
    asyncio.run(engine.dispose())
