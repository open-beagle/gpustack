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
from sqlmodel import SQLModel
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
