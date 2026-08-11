import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.routes import model_preheat_s3_profiles
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import ModelPreheatCachedModel
from gpustack.schemas.users import User
from gpustack.server.db import get_session


@pytest.fixture
def app(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'cached.db'}", poolclass=NullPool
    )

    async def seed():
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(engine) as session:
            for profile_id in (1, 2):
                session.add(
                    ModelPreheatS3Profile(
                        id=profile_id,
                        name=f"profile-{profile_id}",
                        endpoint="https://s3.example.com",
                        bucket="models",
                        access_key_encrypted={"ciphertext": "x"},
                        secret_key_encrypted={"ciphertext": "y"},
                        encryption_key_version="v1",
                    )
                )
            for index in range(3):
                session.add(
                    ModelPreheatCachedModel(
                        profile_id=1,
                        profile_config_version=1,
                        cache_key=f"cache-{index}",
                        source="huggingface",
                        model_id=f"org/model-{index}",
                        resolved_revision="a" * 40,
                        include_patterns=[],
                        exclude_patterns=[],
                        generation_id="preheat-11111111-1111-1111-1111-111111111111",
                        ready_path=f"model-cache/v1/{index}/ready.json",
                        manifest_path=f"model-cache/v1/{index}/manifest.json",
                        manifest_digest="b" * 64,
                        file_count=1,
                        total_size=10,
                        manifest_state="valid" if index < 2 else "invalid",
                        last_verified_at=datetime.now(timezone.utc),
                    )
                )
            await session.commit()

    asyncio.run(seed())
    value = FastAPI()
    value.state.server_config = SimpleNamespace(
        model_preheat_inventory_cursor_key="inventory-test-key"
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    value.dependency_overrides[get_session] = session_override
    value.dependency_overrides[get_admin_user] = admin_override
    router = APIRouter(dependencies=[Depends(get_admin_user)])
    router.include_router(model_preheat_s3_profiles.router, prefix="/profiles")
    value.include_router(router, prefix="/v1")
    exceptions.register_handlers(value)
    yield value
    asyncio.run(engine.dispose())


def test_cached_models_get_uses_database_only_and_stable_cursor(app, monkeypatch):
    def forbidden_scan(*args, **kwargs):
        raise AssertionError("GET must not scan S3")

    monkeypatch.setattr(
        model_preheat_s3_profiles,
        "scan_model_preheat_s3",
        forbidden_scan,
        raising=False,
    )
    with TestClient(app) as client:
        first = client.get("/v1/profiles/1/cached-models?limit=1&manifest_state=valid")
        assert first.status_code == 200, first.text
        payload = first.json()
        assert [item["cache_key"] for item in payload["items"]] == ["cache-0"]
        assert payload["next_cursor"]
        second = client.get(
            "/v1/profiles/1/cached-models",
            params={
                "limit": 1,
                "manifest_state": "valid",
                "cursor": payload["next_cursor"],
            },
        )
        assert [item["cache_key"] for item in second.json()["items"]] == ["cache-1"]


def test_cached_models_cursor_rejects_tamper_and_cross_query_reuse(app):
    with TestClient(app) as client:
        cursor = client.get("/v1/profiles/1/cached-models?limit=1").json()[
            "next_cursor"
        ]
        tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
        assert (
            client.get(
                "/v1/profiles/1/cached-models", params={"limit": 1, "cursor": tampered}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/v1/profiles/2/cached-models", params={"limit": 1, "cursor": cursor}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/v1/profiles/1/cached-models",
                params={"limit": 2, "cursor": cursor},
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/v1/profiles/1/cached-models",
                params={"limit": 1, "manifest_state": "invalid", "cursor": cursor},
            ).status_code
            == 422
        )


def test_inventory_job_routes_create_async_job_and_return_persisted_status(app):
    with TestClient(app) as client:
        created = client.post("/v1/profiles/1/inventory-jobs")
        assert created.status_code == 202, created.text
        body = created.json()
        assert body["state"] == "pending"
        fetched = client.get(f"/v1/profiles/1/inventory-jobs/{body['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == body["id"]
        assert "access" not in fetched.text.lower()
        assert "secret" not in fetched.text.lower()

        gc_created = client.post("/v1/profiles/1/inventory-jobs?kind=gc")
        assert gc_created.status_code == 202
        assert gc_created.json()["kind"] == "gc"
