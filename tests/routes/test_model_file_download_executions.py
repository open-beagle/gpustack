import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import update
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
from gpustack.routes.model_files import reset_model_file
from gpustack.routes.model_preheat_s3_profiles import (
    _detach_unclaimed_download_executions,
    delete_profile,
)
from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_file_download_executions import (
    ModelFileDownloadExecution,
    ModelFileDownloadExecutionProfilePin,
    ModelFileDownloadExecutionStateEnum,
)
from gpustack.schemas.model_files import ModelFile, ModelFilePublic
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.model_storage_sync import (
    ModelStorageSyncTask,
    ModelStorageSyncTaskStateEnum,
)
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.workers import Worker
from gpustack.server.db import get_session
from gpustack.server import model_file_download_execution_service
from gpustack.server.model_file_download_execution_service import (
    create_model_file_with_download_execution,
)
from gpustack.server.model_preheat_s3_profile_lifecycle import (
    ModelPreheatS3ProfileNotActive,
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
    app.state.model_file_download_file_listing_resolver = (
        lambda source, model_id, revision, token=None: ["model.gguf"]
    )

    async def session_override():
        async with AsyncSession(engine) as session:
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


async def _seed(
    engine, key, *, with_profile=True, filename=None, requested_revision="main"
):
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
            huggingface_filename=filename,
            requested_revision=requested_revision,
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


def test_claim_matches_task3_concrete_file_selection(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key, filename="model.gguf"))

    async def seed_artifact():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            session.add(
                ModelPreheatArtifact(
                    profile_id=execution.default_profile_id,
                    profile_config_version=execution.default_profile_config_version,
                    artifact_id="b" * 64,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    include_patterns=["model.gguf", "model.gguf/**"],
                    exclude_patterns=[],
                    manifest_path="storage/huggingface/org/model/"
                    + "b" * 64
                    + "/manifest.json",
                    manifest_digest="c" * 64,
                    file_count=1,
                    total_size=12,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    asyncio.run(seed_artifact())
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] == "b" * 64
    asyncio.run(engine.dispose())


def test_immutable_revision_claims_exact_artifact_without_hub_metadata(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(
        _seed(engine, key, filename="model.bin", requested_revision=SHA)
    )
    metadata_calls = []

    def unavailable(*args, **kwargs):
        metadata_calls.append((args, kwargs))
        raise RuntimeError("hub unavailable")

    app.state.model_file_download_revision_resolver = unavailable
    app.state.model_file_download_file_listing_resolver = unavailable

    async def seed_artifact():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            session.add(
                ModelPreheatArtifact(
                    profile_id=execution.default_profile_id,
                    profile_config_version=execution.default_profile_config_version,
                    artifact_id="f" * 64,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    include_patterns=["model.bin", "model.bin/**"],
                    exclude_patterns=[],
                    manifest_path="storage/huggingface/org/model/"
                    + "f" * 64
                    + "/manifest.json",
                    manifest_digest="e" * 64,
                    file_count=1,
                    total_size=12,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    asyncio.run(seed_artifact())
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] == "f" * 64
    assert metadata_calls == []


def test_s3_completion_uses_provenance_and_publishes_transfer_snapshot(
    tmp_path, monkeypatch
):
    app, engine, key = _app(tmp_path)
    worker_id, model_file_id, _ = asyncio.run(
        _seed(engine, key, filename="model.bin", requested_revision=SHA)
    )

    async def seed_artifact_and_provenance():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            profile_id = execution.default_profile_id
            profile_config_version = execution.default_profile_config_version
            artifact_id = "f" * 64
            session.add(
                ModelPreheatArtifact(
                    profile_id=profile_id,
                    profile_config_version=profile_config_version,
                    artifact_id=artifact_id,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    include_patterns=["model.bin", "model.bin/**"],
                    exclude_patterns=[],
                    manifest_path=f"storage/huggingface/org/model/{artifact_id}/manifest.json",
                    manifest_digest="e" * 64,
                    file_count=1,
                    total_size=12,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            session.add(
                ModelStorageSyncTask(
                    model_file_id=model_file_id,
                    worker_id=worker_id,
                    worker_uuid="worker-uuid",
                    profile_id=profile_id,
                    profile_config_version=profile_config_version,
                    request_identity={},
                    request_digest="d" * 64,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    credential_snapshot_encrypted={},
                    encryption_key_version="v1",
                    artifact_id=artifact_id,
                    state=ModelStorageSyncTaskStateEnum.READY,
                    finished_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            return profile_id

    profile_id = asyncio.run(seed_artifact_and_provenance())
    claimed = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert claimed.status_code == 200, claimed.text
    published = []

    async def capture(cls, event_type, data):
        del cls, event_type
        published.append(data)

    monkeypatch.setattr(ModelFile, "_publish_event", classmethod(capture))
    completed = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/complete",
        json={"transfer_source": "s3", "transfer_profile_id": profile_id},
    )

    async def inspect():
        async with AsyncSession(engine) as session:
            return (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()

    execution = asyncio.run(inspect())

    async def reject_provenance_reselection(*args, **kwargs):
        del args, kwargs
        raise AssertionError("READY replay must use the fixed transfer result")

    monkeypatch.setattr(
        model_file_download_executions,
        "_normalized_transfer_result",
        reject_provenance_reselection,
    )
    replayed = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/complete",
        json={"transfer_source": "s3", "transfer_profile_id": profile_id},
    )
    assert completed.status_code == 200, completed.text
    assert replayed.status_code == 200, replayed.text
    assert execution.transfer_source.value == "peer_via_s3"
    assert execution.source_worker_id == worker_id
    assert published[-1].source == SourceEnum.HUGGING_FACE
    assert published[-1].transfer_source.value == "peer_via_s3"
    assert published[-1].transfer_profile_id == profile_id
    assert published[-1].source_worker_id == worker_id
    asyncio.run(engine.dispose())
    asyncio.run(engine.dispose())


def test_immutable_revision_does_not_claim_partial_glob_artifact_offline(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(
        _seed(engine, key, filename="*.gguf", requested_revision=SHA)
    )
    metadata_calls = []

    def unavailable(*args, **kwargs):
        metadata_calls.append((args, kwargs))
        raise RuntimeError("hub unavailable")

    app.state.model_file_download_revision_resolver = unavailable
    app.state.model_file_download_file_listing_resolver = unavailable

    async def seed_partial_artifact():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            session.add(
                ModelPreheatArtifact(
                    profile_id=execution.default_profile_id,
                    profile_config_version=execution.default_profile_config_version,
                    artifact_id="d" * 64,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    include_patterns=["model-1.gguf", "model-1.gguf/**"],
                    exclude_patterns=[],
                    manifest_path="storage/huggingface/org/model/"
                    + "d" * 64
                    + "/manifest.json",
                    manifest_digest="e" * 64,
                    file_count=1,
                    total_size=12,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    asyncio.run(seed_partial_artifact())
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] is None
    assert metadata_calls == []
    asyncio.run(engine.dispose())


@pytest.mark.parametrize(
    ("artifact_patterns", "expected_artifact_id"),
    [
        (["model-1.gguf", "model-1.gguf/**"], None),
        (
            [
                "model-1.gguf",
                "model-1.gguf/**",
                "model-2.gguf",
                "model-2.gguf/**",
            ],
            "d" * 64,
        ),
    ],
)
def test_claim_requires_complete_artifact_for_glob_selection(
    tmp_path, artifact_patterns, expected_artifact_id
):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key, filename="*.gguf"))
    app.state.model_file_download_file_listing_resolver = (
        lambda source, model_id, revision, token=None: [
            "model-1.gguf",
            "model-2.gguf",
        ]
    )

    async def seed_incomplete_artifact():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            session.add(
                ModelPreheatArtifact(
                    profile_id=execution.default_profile_id,
                    profile_config_version=execution.default_profile_config_version,
                    artifact_id="d" * 64,
                    source="huggingface",
                    model_id="org/model",
                    resolved_revision=SHA,
                    include_patterns=artifact_patterns,
                    exclude_patterns=[],
                    manifest_path="storage/huggingface/org/model/"
                    + "d" * 64
                    + "/manifest.json",
                    manifest_digest="e" * 64,
                    file_count=len(artifact_patterns) // 2,
                    total_size=12,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

    asyncio.run(seed_incomplete_artifact())
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200
    assert response.json()["artifact_id"] == expected_artifact_id
    asyncio.run(engine.dispose())


def test_concurrent_first_claim_returns_one_pinned_revision(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    revisions = iter(("a" * 40, "b" * 40))

    def resolver(source, model_id, revision, token=None):
        with lock:
            resolved = next(revisions)
        barrier.wait(timeout=5)
        return resolved

    app.state.model_file_download_revision_resolver = resolver

    def claim():
        return TestClient(app).post(
            f"/v1/model-files/{model_file_id}/download-executions/claim"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: claim(), range(2)))
    assert [response.status_code for response in responses] == [200, 200]
    assert len({response.json()["resolved_revision"] for response in responses}) == 1
    asyncio.run(engine.dispose())


def test_ready_execution_claim_is_idempotent_and_does_not_reopen(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key, with_profile=False))
    client = TestClient(app)
    assert (
        client.post(
            f"/v1/model-files/{model_file_id}/download-executions/claim"
        ).status_code
        == 200
    )
    completed = client.post(
        f"/v1/model-files/{model_file_id}/download-executions/complete",
        json={"transfer_source": "huggingface"},
    )
    assert completed.status_code == 200
    claimed_again = client.post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert claimed_again.status_code == 200

    async def inspect():
        async with AsyncSession(engine) as session:
            return (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()

    execution = asyncio.run(inspect())
    assert execution.state.value == "ready"
    assert execution.transfer_source.value == "huggingface"
    assert execution.finished_at is not None
    asyncio.run(engine.dispose())


def test_error_execution_requires_explicit_reset_before_reclaim(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key, with_profile=False))
    client = TestClient(app)
    first = client.post(f"/v1/model-files/{model_file_id}/download-executions/claim")
    assert first.status_code == 200
    failed = client.post(
        f"/v1/model-files/{model_file_id}/download-executions/fail",
        json={"error_code": "worker_execution_failed"},
    )
    assert failed.status_code == 200
    rejected = client.post(f"/v1/model-files/{model_file_id}/download-executions/claim")
    assert rejected.status_code == 409
    assert rejected.json()["message"] == "execution_not_claimable"

    async def reset():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await reset_model_file(session, model_file_id)

    asyncio.run(reset())
    reclaimed = client.post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert reclaimed.status_code == 200
    assert reclaimed.json()["resolved_revision"] == first.json()["resolved_revision"]
    asyncio.run(engine.dispose())


def test_profile_referenced_by_unclaimed_download_execution_can_be_deleted(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))

    async def attempt_delete():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            profile = await session.get(
                ModelPreheatS3Profile, execution.default_profile_id
            )
            profile_id = profile.id
            execution_id = execution.id
            profile.default_slot = None
            session.add(profile)
            await session.commit()
            request = SimpleNamespace(app=app)
            await delete_profile(request, session, profile_id)
            refreshed_execution = await session.get(
                ModelFileDownloadExecution, execution_id
            )
            refreshed_profile = await session.get(ModelPreheatS3Profile, profile_id)
            return refreshed_execution, refreshed_profile

    execution, profile = asyncio.run(attempt_delete())
    assert profile is None
    assert execution.default_profile_id is None
    assert execution.default_profile_config_version is None
    assert execution.credential_snapshot_encrypted is None
    assert execution.encryption_key_version is None
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200, response.text
    assert response.json()["profile"] is None
    assert "access-value" not in response.text
    assert "secret-value" not in response.text
    asyncio.run(engine.dispose())


def test_download_claim_wins_delete_detach_interleave(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))

    async def interleave():
        async with AsyncSession(engine) as delete_session:
            execution = (
                await delete_session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            execution_id = execution.id
            profile_id = execution.default_profile_id

            async with AsyncSession(engine) as claim_session:
                claimed_at = datetime.now(timezone.utc)
                claimed = await claim_session.exec(
                    update(ModelFileDownloadExecution)
                    .where(
                        ModelFileDownloadExecution.id == execution_id,
                        ModelFileDownloadExecution.state
                        == ModelFileDownloadExecutionStateEnum.PENDING,
                    )
                    .values(
                        state=ModelFileDownloadExecutionStateEnum.RUNNING,
                        claimed_by_worker_uuid="worker-uuid",
                        claimed_at=claimed_at,
                    )
                )
                assert claimed.rowcount == 1
                await claim_session.exec(
                    update(ModelPreheatS3Profile)
                    .where(ModelPreheatS3Profile.id == profile_id)
                    .values(ever_used_at=claimed_at)
                )
                await claim_session.commit()

            detached = await _detach_unclaimed_download_executions(
                delete_session, profile_id, [execution_id]
            )
            assert detached is False

        async with AsyncSession(engine) as session:
            stored_execution = await session.get(
                ModelFileDownloadExecution, execution_id
            )
            stored_profile = await session.get(ModelPreheatS3Profile, profile_id)
            pin = await session.get(ModelFileDownloadExecutionProfilePin, execution_id)
            return stored_execution, stored_profile, pin

    execution, profile, pin = asyncio.run(interleave())
    assert execution.state == ModelFileDownloadExecutionStateEnum.RUNNING
    assert execution.default_profile_id == profile.id
    assert execution.credential_snapshot_encrypted is not None
    assert profile.ever_used_at is not None
    assert pin.profile_id == profile.id
    asyncio.run(engine.dispose())


def test_claim_marks_profile_used_and_prevents_delete(tmp_path):
    app, engine, key = _app(tmp_path)
    _, model_file_id, _ = asyncio.run(_seed(engine, key))
    response = TestClient(app).post(
        f"/v1/model-files/{model_file_id}/download-executions/claim"
    )
    assert response.status_code == 200, response.text

    async def attempt_delete():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            profile = await session.get(
                ModelPreheatS3Profile, execution.default_profile_id
            )
            assert profile.ever_used_at is not None
            profile_id = profile.id
            profile.default_slot = None
            session.add(profile)
            await session.commit()
            with pytest.raises(HTTPException) as exc_info:
                await delete_profile(SimpleNamespace(app=app), session, profile_id)
            assert exc_info.value.status_code == 409
            assert exc_info.value.message == "profile_has_been_used"

    asyncio.run(attempt_delete())
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


def test_default_profile_maintenance_race_falls_back_without_credentials(
    tmp_path, monkeypatch
):
    app, engine, key = _app(tmp_path)

    async def reject_stale_active_profile(*args, **kwargs):
        del args, kwargs
        raise ModelPreheatS3ProfileNotActive

    monkeypatch.setattr(
        model_file_download_execution_service,
        "lock_active_profile_for_new_work",
        reject_stale_active_profile,
    )
    _, model_file_id, _ = asyncio.run(_seed(engine, key))

    async def inspect():
        async with AsyncSession(engine) as session:
            execution = (
                await session.exec(
                    select(ModelFileDownloadExecution).where(
                        ModelFileDownloadExecution.model_file_id == model_file_id
                    )
                )
            ).one()
            pins = (
                await session.exec(select(ModelFileDownloadExecutionProfilePin))
            ).all()
            return execution, pins

    execution, pins = asyncio.run(inspect())
    assert execution.default_profile_id is None
    assert execution.default_profile_config_version is None
    assert execution.credential_snapshot_encrypted is None
    assert execution.encryption_key_version is None
    assert pins == []
    asyncio.run(engine.dispose())
