import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.routes import model_preheat_distribution_policies
from gpustack.schemas.model_preheat_distribution_policies import (
    ModelPreheatDistributionPolicy,
    ModelPreheatWorkerObservation,
)
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import ModelPreheatTargetScopeEnum
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.schemas.users import User
from gpustack.server.db import get_session


def _test_app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'policies.db'}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))
    app = FastAPI()
    app.state.model_preheat_worker_reconciler = SimpleNamespace(
        reconcile_policy=lambda policy_id: _done(policy_id)
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(
        model_preheat_distribution_policies.router,
        prefix="/model-preheat-distribution-policies",
    )
    app.include_router(router, prefix="/v1")
    exceptions.register_handlers(app)
    return app, engine


async def _create_tables(engine):
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)


async def _done(value):
    return value


async def _seed(engine):
    async with AsyncSession(engine) as session:
        profile = ModelPreheatS3Profile(
            name="profile",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "access-secret"},
            secret_key_encrypted={"ciphertext": "secret-secret"},
            encryption_key_version="v1",
        )
        session.add(profile)
        await session.flush()
        policy = ModelPreheatDistributionPolicy(
            name="模型同步",
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            request_identity={
                "source": "huggingface",
                "model_id": "org/model",
                "requested_revision": "main",
                "include_patterns": [],
                "exclude_patterns": [],
            },
            request_digest="c" * 64,
            target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
            worker_selector={"worker_uuids": ["worker-a"]},
            gpu_selector={},
            selector_digest="d" * 64,
            created_by_task_id=None,
            last_reconciled_at=datetime.now(timezone.utc),
        )
        session.add(policy)
        await session.commit()
        await session.refresh(policy)
        return policy.id


async def _seed_artifact(engine):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        profile = ModelPreheatS3Profile(
            name="artifact-profile",
            endpoint="https://s3.example.com",
            bucket="models",
            access_key_encrypted={"ciphertext": "access-secret"},
            secret_key_encrypted={"ciphertext": "secret-secret"},
            encryption_key_version="v1",
        )
        session.add(profile)
        await session.flush()
        artifact = ModelPreheatArtifact(
            profile_id=profile.id,
            profile_config_version=profile.config_version,
            artifact_id="a" * 64,
            source="huggingface",
            model_id="org/model",
            resolved_revision="commit-1",
            include_patterns=[],
            exclude_patterns=[],
            manifest_path="models/huggingface/org/model/manifest.json",
            manifest_digest="b" * 64,
            file_count=1,
            total_size=10,
            manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
            last_verified_at=datetime.now(timezone.utc),
        )
        session.add(artifact)
        await session.commit()
        return profile.id, artifact.artifact_id


def test_policy_routes_list_get_disable_and_reconcile_without_credentials(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))
    with TestClient(app) as client:
        listed = client.get("/v1/model-preheat-distribution-policies")
        fetched = client.get(f"/v1/model-preheat-distribution-policies/{policy_id}")
        disabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": False},
        )
        reconciled = client.post(
            f"/v1/model-preheat-distribution-policies/{policy_id}/reconcile"
        )
    asyncio.run(engine.dispose())

    assert listed.status_code == 200
    assert fetched.status_code == 200
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False
    assert fetched.json()["profile_config_version"] == 1
    assert reconciled.status_code == 200
    payload = str([listed.json(), fetched.json(), disabled.json()])
    assert "access-secret" not in payload
    assert "secret-secret" not in payload
    assert "snapshot_encrypted" not in payload


def test_policy_patch_rejects_selector_or_credential_mutation(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))
    with TestClient(app) as client:
        selector = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"worker_selector": {"worker_uuids": ["other"]}},
        )
        credential = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"access_key": "credential-shaped-value"},
        )
    asyncio.run(engine.dispose())

    assert selector.status_code == 422
    assert credential.status_code == 422


def test_policy_can_be_created_from_existing_s3_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "artifact-distribution",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    asyncio.run(engine.dispose())

    assert created.status_code == 200, created.text
    assert created.json()["source_artifact"] == artifact_id
    assert created.json()["source_artifact_id"] is not None
    assert created.json()["source_sync_task_id"] is None
    assert created.json()["request_identity"]["source"] == "huggingface"
    assert created.json()["trigger_mode"] == "manual"


def test_policy_exposes_blocked_reason_for_stale_fixed_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))

    async def mark_stale():
        async with AsyncSession(engine) as session:
            artifact = (
                await session.exec(
                    select(ModelPreheatArtifact).where(
                        ModelPreheatArtifact.profile_id == profile_id,
                        ModelPreheatArtifact.artifact_id == artifact_id,
                    )
                )
            ).one()
            artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
            session.add(artifact)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "stale-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
    asyncio.run(engine.dispose())

    assert created.status_code == 409
    assert created.json()["message"] == "artifact_stale"


def test_reenable_rejects_stale_fixed_artifact(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "fixed-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
        policy_id = created.json()["id"]
        assert (
            client.patch(
                f"/v1/model-preheat-distribution-policies/{policy_id}",
                json={"enabled": False},
            ).status_code
            == 200
        )

    async def mark_stale():
        async with AsyncSession(engine) as session:
            artifact = (
                await session.exec(
                    select(ModelPreheatArtifact).where(
                        ModelPreheatArtifact.profile_id == profile_id,
                        ModelPreheatArtifact.artifact_id == artifact_id,
                    )
                )
            ).one()
            artifact.manifest_state = ModelPreheatInventoryManifestStateEnum.STALE
            session.add(artifact)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        enabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": True},
        )
    asyncio.run(engine.dispose())

    assert enabled.status_code == 409
    assert enabled.json()["message"] == "artifact_stale"


def test_reenable_rejects_profile_config_version_drift(tmp_path):
    app, engine = _test_app(tmp_path)
    profile_id, artifact_id = asyncio.run(_seed_artifact(engine))
    with TestClient(app) as client:
        created = client.post(
            "/v1/model-preheat-distribution-policies",
            json={
                "name": "version-bound-artifact",
                "profile_id": profile_id,
                "artifact_id": artifact_id,
                "target_scope": "selected_workers",
                "worker_selector": {"worker_uuids": ["worker-a"]},
                "gpu_selector": {},
            },
        )
        policy_id = created.json()["id"]
        assert (
            client.patch(
                f"/v1/model-preheat-distribution-policies/{policy_id}",
                json={"enabled": False},
            ).status_code
            == 200
        )

    async def rotate_profile():
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
            profile.config_version += 1
            session.add(profile)
            await session.commit()

    asyncio.run(rotate_profile())
    with TestClient(app) as client:
        enabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": True},
        )
    asyncio.run(engine.dispose())

    assert enabled.status_code == 409
    assert enabled.json()["message"] == "distribution_profile_version_stale"


def test_repeated_disable_preserves_profile_version_stale(tmp_path):
    app, engine = _test_app(tmp_path)
    policy_id = asyncio.run(_seed(engine))

    async def mark_stale():
        async with AsyncSession(engine) as session:
            policy = await session.get(ModelPreheatDistributionPolicy, policy_id)
            policy.enabled = False
            policy.profile_version_stale = True
            session.add(policy)
            await session.commit()

    asyncio.run(mark_stale())
    with TestClient(app) as client:
        disabled = client.patch(
            f"/v1/model-preheat-distribution-policies/{policy_id}",
            json={"enabled": False},
        )

    async def read_policy():
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatDistributionPolicy, policy_id)

    policy = asyncio.run(read_policy())
    asyncio.run(engine.dispose())

    assert disabled.status_code == 200
    assert policy.profile_version_stale is True


def test_policy_schema_and_successor_migration_are_portable():
    migration = (
        "gpustack/migrations/versions/"
        "2026_08_11_1500-b8c9d0e1f2a3_add_preheat_distribution_policies.py"
    )
    source = Path(migration).read_text(encoding="utf-8")
    assert 'down_revision: Union[str, None] = "a7b8c9d0e1f2"' in source
    assert "postgresql_where" not in source
    assert "CREATE UNIQUE INDEX" not in source
    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        for table in (
            ModelPreheatDistributionPolicy.__table__,
            ModelPreheatWorkerObservation.__table__,
        ):
            ddl = str(CreateTable(table).compile(dialect=dialect))
            assert table.name in ddl
