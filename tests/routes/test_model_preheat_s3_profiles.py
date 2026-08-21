import asyncio
import ast
import importlib
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user, get_current_user
from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.routes import model_preheat_s3_profiles
from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
)
from gpustack.schemas.model_preheat_schedules import ModelPreheatSchedule
from gpustack.schemas.model_preheats import (
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatTargetScopeEnum,
)
from gpustack.schemas.users import User
from gpustack.server.db import get_session


API_PREFIX = "/v1/model-preheat-s3-profiles"
ACCESS_KEY = "plain-access-key"
SECRET_KEY = "plain-secret-key"


async def _create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def _drop_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "profiles.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))

    test_app = FastAPI()
    test_app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        force_auth_localhost=True,
    )
    test_app.state.test_engine = engine

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def admin_user_override():
        return User(id=1, username="admin", is_admin=True, hashed_password="")

    test_app.dependency_overrides[get_session] = session_override
    test_app.dependency_overrides[get_admin_user] = admin_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(
        model_preheat_s3_profiles.router,
        prefix="/model-preheat-s3-profiles",
    )
    test_app.include_router(admin_router, prefix="/v1")
    exceptions.register_handlers(test_app)

    yield test_app

    test_app.dependency_overrides.clear()
    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def profile_payload(**overrides):
    payload = {
        "name": "center-cache",
        "description": "central cache",
        "endpoint": "https://s3.example.com",
        "bucket": "models",
        "prefix": "datamodel//team-a/",
        "access_key": ACCESS_KEY,
        "secret_key": SECRET_KEY,
        "tls_enabled": True,
        "region": "us-east-1",
        "use_virtual_hosted_style": True,
        "default_slot": None,
    }
    payload.update(overrides)
    return payload


def create_profile(client, **overrides):
    response = client.post(API_PREFIX, json=profile_payload(**overrides))
    assert response.status_code == 200, response.text
    return response.json()


def test_profile_delete_returns_conflict_when_schedule_references_it(app, client):
    created = create_profile(client)

    async def seed_schedule():
        async with AsyncSession(app.state.test_engine) as session:
            session.add(
                ModelPreheatSchedule(
                    name="profile-reference",
                    cron_expression="0 1 * * *",
                    timezone="UTC",
                    window_duration_minutes=60,
                    source="huggingface",
                    model_id="org/model",
                    target_scope=ModelPreheatTargetScopeEnum.SELECTED_WORKERS,
                    target_worker_uuids=["worker-a"],
                    s3_profile_id=created["id"],
                    s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                )
            )
            await session.commit()

    asyncio.run(seed_schedule())
    response = client.delete(f"{API_PREFIX}/{created['id']}")
    assert response.status_code == 409
    assert response.json()["reason"] == "model_preheat_schedule_uses_profile"


def test_profile_route_is_registered_on_v1_admin_router():
    route_file = Path(__file__).parents[2] / "gpustack" / "routes" / "routes.py"
    tree = ast.parse(route_file.read_text())

    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "gpustack.routes"
        and any(alias.name == "model_preheat_s3_profiles" for alias in node.names)
        for node in ast.walk(tree)
    )
    registered = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "v1_admin_router"
        and node.func.attr == "include_router"
        and any(
            keyword.arg == "prefix"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "/model-preheat-s3-profiles"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )

    assert imported is True
    assert registered is True


def test_profile_create_is_admin_only(tmp_path):
    db_path = tmp_path / "non_admin.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))

    test_app = FastAPI()
    test_app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        force_auth_localhost=True,
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def current_user_override():
        return User(id=2, username="viewer", is_admin=False, hashed_password="")

    test_app.dependency_overrides[get_session] = session_override
    test_app.dependency_overrides[get_current_user] = current_user_override
    admin_router = APIRouter(dependencies=[Depends(get_admin_user)])
    admin_router.include_router(
        model_preheat_s3_profiles.router,
        prefix="/model-preheat-s3-profiles",
    )
    test_app.include_router(admin_router, prefix="/v1")
    exceptions.register_handlers(test_app)

    with TestClient(test_app) as test_client:
        response = test_client.post(API_PREFIX, json=profile_payload())

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 403


def test_profile_create_is_admin_only_on_real_api_router(tmp_path, monkeypatch):
    db_path = tmp_path / "real_router_non_admin.db"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={"check_same_thread": False},
    )
    asyncio.run(_create_tables(engine))

    test_app = FastAPI()
    test_app.state.server_config = SimpleNamespace(
        model_preheat_credential_key=generate_model_preheat_credential_key(),
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
        force_auth_localhost=True,
    )

    async def session_override():
        async with AsyncSession(engine) as session:
            yield session

    async def current_user_override():
        return User(id=2, username="viewer", is_admin=False, hashed_password="")

    onelogin = types.ModuleType("onelogin")
    saml2 = types.ModuleType("onelogin.saml2")
    auth = types.ModuleType("onelogin.saml2.auth")
    auth.OneLogin_Saml2_Auth = object
    python_multipart = types.ModuleType("python_multipart")
    python_multipart.__version__ = "0.0.20"
    cachetools = types.ModuleType("cachetools")

    class TTLCache(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()

    cachetools.TTLCache = TTLCache
    aiolimiter = types.ModuleType("aiolimiter")

    class AsyncLimiter:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    aiolimiter.AsyncLimiter = AsyncLimiter
    monkeypatch.setitem(sys.modules, "onelogin", onelogin)
    monkeypatch.setitem(sys.modules, "onelogin.saml2", saml2)
    monkeypatch.setitem(sys.modules, "onelogin.saml2.auth", auth)
    monkeypatch.setitem(sys.modules, "python_multipart", python_multipart)
    monkeypatch.setitem(sys.modules, "cachetools", cachetools)
    monkeypatch.setitem(sys.modules, "aiolimiter", aiolimiter)
    real_routes = importlib.import_module("gpustack.routes.routes")

    test_app.dependency_overrides[get_session] = session_override
    test_app.dependency_overrides[get_current_user] = current_user_override
    test_app.include_router(real_routes.api_router)
    exceptions.register_handlers(test_app)

    with TestClient(test_app) as test_client:
        response = test_client.post(API_PREFIX, json=profile_payload())

    asyncio.run(_drop_tables(engine))
    asyncio.run(engine.dispose())

    assert response.status_code == 403


def test_create_requires_configured_credential_key(app):
    app.state.server_config.model_preheat_credential_key = None

    response = TestClient(app).post(API_PREFIX, json=profile_payload())

    assert response.status_code == 503
    body = response.json()
    assert body["reason"] == "ServiceUnavailable"
    assert "credential_encryption_unavailable" in body["message"]


def test_validation_error_scrubs_plain_credentials(client):
    payload = profile_payload()
    payload.pop("endpoint")

    response = client.post(
        API_PREFIX,
        json=payload,
    )

    assert response.status_code == 422
    assert ACCESS_KEY not in response.text
    assert SECRET_KEY not in response.text


@pytest.mark.parametrize(
    ("field", "plain_value", "nested_plain_value"),
    [
        ("access_key", ACCESS_KEY, "nested-plain-access-key"),
        ("secret_key", SECRET_KEY, "nested-plain-secret-key"),
    ],
)
def test_validation_error_scrubs_sensitive_field_input(
    client, field, plain_value, nested_plain_value
):
    payload = profile_payload(
        **{field: {"v": plain_value, "nested": {"raw": nested_plain_value}}}
    )

    response = client.post(API_PREFIX, json=payload)

    assert response.status_code == 422
    assert ACCESS_KEY not in response.text
    assert SECRET_KEY not in response.text
    assert plain_value not in response.text
    assert nested_plain_value not in response.text


def test_create_returns_public_profile_and_never_exposes_plain_credentials(client):
    created = create_profile(client, default_slot="global")

    assert created["name"] == "center-cache"
    assert created["prefix"] == "datamodel/team-a"
    assert created["credential_configured"] is True
    assert created["tls_verify"] is True
    assert created["config_version"] == 1
    assert created["connectivity_state"] == "no_workers"
    assert created["last_connectivity_check_id"] is None
    assert ACCESS_KEY not in repr(created)
    assert SECRET_KEY not in repr(created)
    assert "access_key" not in created
    assert "secret_key" not in created


def test_profile_detail_persists_expired_connectivity_as_stale(client, app):
    created = create_profile(client)
    app.state.server_config.model_preheat_connectivity_ttl_seconds = 30
    asyncio.run(
        _update_stored_profile(
            app,
            created["id"],
            {
                "connectivity_state": ModelPreheatS3ConnectivityStateEnum.AVAILABLE,
                "last_connectivity_checked_at": datetime.now(timezone.utc)
                - timedelta(seconds=31),
            },
        )
    )

    response = client.get(f"{API_PREFIX}/{created['id']}")

    assert response.status_code == 200
    assert response.json()["connectivity_state"] == "stale"
    assert (
        asyncio.run(_stored_profile(app, created["id"])).connectivity_state
        == ModelPreheatS3ConnectivityStateEnum.STALE
    )


def test_profile_list_persists_expired_connectivity_as_stale(client, app):
    created = create_profile(client)
    app.state.server_config.model_preheat_connectivity_ttl_seconds = 30
    asyncio.run(
        _update_stored_profile(
            app,
            created["id"],
            {
                "connectivity_state": ModelPreheatS3ConnectivityStateEnum.PARTIAL,
                "last_connectivity_checked_at": datetime.now(timezone.utc)
                - timedelta(seconds=31),
            },
        )
    )

    response = client.get(API_PREFIX)

    assert response.status_code == 200
    assert response.json()["items"][0]["connectivity_state"] == "stale"
    assert (
        asyncio.run(_stored_profile(app, created["id"])).connectivity_state
        == ModelPreheatS3ConnectivityStateEnum.STALE
    )


def test_profile_name_is_unique(client):
    create_profile(client)

    response = client.post(API_PREFIX, json=profile_payload())

    assert response.status_code == 409


@pytest.mark.parametrize(
    "prefix",
    ["../escape", "/absolute/../escape", r"models\\bad", "models/\x01bad"],
)
def test_prefix_rejects_unsafe_paths(client, prefix):
    response = client.post(API_PREFIX, json=profile_payload(prefix=prefix))

    assert response.status_code == 422


@pytest.mark.parametrize("endpoint", ["ftp://s3.example.com", "s3.example.com"])
def test_endpoint_allows_only_http_or_https(client, endpoint):
    response = client.post(API_PREFIX, json=profile_payload(endpoint=endpoint))

    assert response.status_code == 422


def test_setting_default_profile_unsets_other_defaults(client):
    first = create_profile(client, name="first", default_slot="global")
    second = create_profile(client, name="second", default_slot="global")

    first_response = client.get(f"{API_PREFIX}/{first['id']}")
    second_response = client.get(f"{API_PREFIX}/{second['id']}")

    assert first_response.json()["is_default"] is False
    assert second_response.json()["is_default"] is True


def test_failed_default_create_does_not_clear_existing_default(
    client, app, monkeypatch
):
    first = create_profile(client, name="first", default_slot="global")

    async def fail_create(*args, **kwargs):
        raise RuntimeError("forced create failure")

    monkeypatch.setattr(ModelPreheatS3Profile, "create", fail_create)

    response = client.post(
        API_PREFIX,
        json=profile_payload(name="second", default_slot="global"),
    )

    assert response.status_code == 500
    stored = asyncio.run(_stored_profile(app, first["id"]))
    assert stored.default_slot == "global"


def test_update_without_credentials_preserves_encrypted_values_and_config_version(
    client, app
):
    created = create_profile(client)
    before = asyncio.run(_stored_profile(app, created["id"]))

    response = client.patch(
        f"{API_PREFIX}/{created['id']}",
        json={"description": "updated", "prefix": "datamodel/updated"},
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    after = asyncio.run(_stored_profile(app, created["id"]))
    assert updated["config_version"] == 2
    assert updated["connectivity_state"] == "no_workers"
    assert after.last_connectivity_check_id is None
    assert after.last_connectivity_checked_at is None
    assert after.access_key_encrypted == before.access_key_encrypted
    assert after.secret_key_encrypted == before.secret_key_encrypted


def test_update_with_credentials_reencrypts_and_increments_config_version(client, app):
    created = create_profile(client)
    before = asyncio.run(_stored_profile(app, created["id"]))

    response = client.patch(
        f"{API_PREFIX}/{created['id']}",
        json={"access_key": "new-access", "secret_key": "new-secret"},
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    after = asyncio.run(_stored_profile(app, created["id"]))
    assert updated["config_version"] == 2
    assert after.access_key_encrypted != before.access_key_encrypted
    assert after.secret_key_encrypted != before.secret_key_encrypted
    assert "new-access" not in repr(updated)
    assert "new-secret" not in repr(updated)


def test_update_connection_config_increments_version_and_resets_connectivity(
    client, app
):
    created = create_profile(client)
    asyncio.run(
        _update_stored_profile(
            app,
            created["id"],
            {
                "connectivity_state": ModelPreheatS3ConnectivityStateEnum.AVAILABLE,
                "last_connectivity_check_id": 42,
                "last_connectivity_checked_at": datetime(2026, 8, 10, 10, 0, 0),
            },
        )
    )

    response = client.patch(
        f"{API_PREFIX}/{created['id']}",
        json={
            "endpoint": "https://s3-new.example.com",
            "prefix": "datamodel/team-b",
            "tls_enabled": False,
        },
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    after = asyncio.run(_stored_profile(app, created["id"]))
    assert updated["config_version"] == 2
    assert updated["connectivity_state"] == "no_workers"
    assert after.last_connectivity_check_id is None
    assert after.last_connectivity_checked_at is None


def test_concurrent_connection_config_updates_use_database_cas(app, monkeypatch):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            created_response = await async_client.post(
                API_PREFIX, json=profile_payload()
            )
            assert created_response.status_code == 200
            profile_id = created_response.json()["id"]
            original_get_profile = model_preheat_s3_profiles._get_profile
            loaded = 0
            loaded_lock = asyncio.Lock()
            both_loaded = asyncio.Event()

            async def get_profile_with_barrier(*args, **kwargs):
                nonlocal loaded
                profile = await original_get_profile(*args, **kwargs)
                async with loaded_lock:
                    loaded += 1
                    if loaded == 2:
                        both_loaded.set()
                await asyncio.wait_for(both_loaded.wait(), timeout=5)
                return profile

            monkeypatch.setattr(
                model_preheat_s3_profiles,
                "_get_profile",
                get_profile_with_barrier,
            )
            first, second = await asyncio.gather(
                async_client.patch(
                    f"{API_PREFIX}/{profile_id}",
                    json={"endpoint": "https://s3-a.example.com"},
                ),
                async_client.patch(
                    f"{API_PREFIX}/{profile_id}",
                    json={"endpoint": "https://s3-b.example.com"},
                ),
            )

        assert sorted([first.status_code, second.status_code]) == [200, 409]
        conflict = first if first.status_code == 409 else second
        assert conflict.json()["reason"] == "profile_config_conflict"
        assert conflict.json()["message"] == "profile_config_conflict"
        stored = await _stored_profile(app, profile_id)
        assert stored.config_version == 2
        assert stored.endpoint in {
            "https://s3-a.example.com",
            "https://s3-b.example.com",
        }

    asyncio.run(run())


def test_default_profile_changes_roll_back_when_config_cas_loses(app, monkeypatch):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            default_response = await async_client.post(
                API_PREFIX,
                json=profile_payload(name="default", default_slot="global"),
            )
            target_response = await async_client.post(
                API_PREFIX,
                json=profile_payload(name="target", default_slot=None),
            )
            default_id = default_response.json()["id"]
            target_id = target_response.json()["id"]
            original_unset = model_preheat_s3_profiles._unset_other_defaults
            defaults_unset = asyncio.Event()
            allow_loser_cas = asyncio.Event()

            async def unset_with_barrier(*args, **kwargs):
                await original_unset(*args, **kwargs)
                defaults_unset.set()
                await asyncio.wait_for(allow_loser_cas.wait(), timeout=5)

            monkeypatch.setattr(
                model_preheat_s3_profiles,
                "_unset_other_defaults",
                unset_with_barrier,
            )
            loser_request = asyncio.create_task(
                async_client.patch(
                    f"{API_PREFIX}/{target_id}",
                    json={
                        "endpoint": "https://loser.example.com",
                        "default_slot": "global",
                    },
                )
            )
            await asyncio.wait_for(defaults_unset.wait(), timeout=5)
            winner_response = await async_client.patch(
                f"{API_PREFIX}/{target_id}",
                json={"endpoint": "https://winner.example.com"},
            )
            allow_loser_cas.set()
            loser_response = await loser_request

        assert winner_response.status_code == 200
        assert loser_response.status_code == 409
        assert loser_response.json()["message"] == "profile_config_conflict"
        stored_default = await _stored_profile(app, default_id)
        stored_target = await _stored_profile(app, target_id)
        assert stored_default.default_slot == "global"
        assert stored_target.default_slot is None
        assert stored_target.endpoint == "https://winner.example.com"
        assert stored_target.config_version == 2

    asyncio.run(run())


def test_update_rotates_old_key_credentials_without_plaintext_leak(client, app):
    old_key = app.state.server_config.model_preheat_credential_key
    created = create_profile(client)
    before = asyncio.run(_stored_profile(app, created["id"]))
    assert before.access_key_encrypted["key_version"] == "v1"
    assert before.secret_key_encrypted["key_version"] == "v1"

    new_key = generate_model_preheat_credential_key()
    app.state.server_config.model_preheat_credential_key = new_key
    app.state.server_config.model_preheat_credential_key_version = "v2"
    app.state.server_config.model_preheat_credential_old_keys = {"v1": old_key}

    response = client.patch(
        f"{API_PREFIX}/{created['id']}",
        json={"description": "rotated"},
    )

    assert response.status_code == 200, response.text
    assert ACCESS_KEY not in response.text
    assert SECRET_KEY not in response.text
    after = asyncio.run(_stored_profile(app, created["id"]))
    cipher = ModelPreheatCredentialCipher(
        current_key=new_key,
        current_key_version="v2",
        old_keys={"v1": old_key},
    )
    assert after.access_key_encrypted["key_version"] == "v2"
    assert after.secret_key_encrypted["key_version"] == "v2"
    assert cipher.decrypt(after.access_key_encrypted) == ACCESS_KEY
    assert cipher.decrypt(after.secret_key_encrypted) == SECRET_KEY


def test_update_single_credential_rotates_other_old_key_credential(client, app):
    old_key = app.state.server_config.model_preheat_credential_key
    created = create_profile(client)
    new_key = generate_model_preheat_credential_key()
    app.state.server_config.model_preheat_credential_key = new_key
    app.state.server_config.model_preheat_credential_key_version = "v2"
    app.state.server_config.model_preheat_credential_old_keys = {"v1": old_key}

    response = client.patch(
        f"{API_PREFIX}/{created['id']}",
        json={"access_key": "rotated-access"},
    )

    assert response.status_code == 200, response.text
    assert "rotated-access" not in response.text
    assert SECRET_KEY not in response.text
    after = asyncio.run(_stored_profile(app, created["id"]))
    cipher = ModelPreheatCredentialCipher(
        current_key=new_key,
        current_key_version="v2",
        old_keys={"v1": old_key},
    )
    assert after.access_key_encrypted["key_version"] == "v2"
    assert after.secret_key_encrypted["key_version"] == "v2"
    assert cipher.decrypt(after.access_key_encrypted) == "rotated-access"
    assert cipher.decrypt(after.secret_key_encrypted) == SECRET_KEY


def test_update_requires_configured_credential_key_even_without_credentials(app):
    with TestClient(app) as test_client:
        created = create_profile(test_client)
        app.state.server_config.model_preheat_credential_key = None

        response = test_client.patch(
            f"{API_PREFIX}/{created['id']}",
            json={"description": "blocked"},
        )

    assert response.status_code == 503
    assert "credential_encryption_unavailable" in response.json()["message"]


def test_delete_requires_configured_credential_key(app):
    with TestClient(app) as test_client:
        created = create_profile(test_client)
        app.state.server_config.model_preheat_credential_key = None

        response = test_client.delete(f"{API_PREFIX}/{created['id']}")

    assert response.status_code == 503
    assert "credential_encryption_unavailable" in response.json()["message"]


def test_update_system_managed_profile_is_forbidden(client, app):
    created = create_profile(client)
    asyncio.run(
        _update_stored_profile(
            app, created["id"], {"system_managed": True}
        )
    )
    response = client.patch(
        f"{API_PREFIX}/{created['id']}", json={"description": "blocked"}
    )
    assert response.status_code == 403
    assert response.json()["reason"] == "system_profile_read_only"
    # 未改动：description 仍为创建时的值。
    stored = asyncio.run(_stored_profile(app, created["id"]))
    assert stored.description == "central cache"


def test_delete_system_managed_profile_is_forbidden(client, app):
    created = create_profile(client)
    asyncio.run(_update_stored_profile(app, created["id"], {"system_managed": True}))
    response = client.delete(f"{API_PREFIX}/{created['id']}")
    assert response.status_code == 409
    assert response.json()["reason"] == "system_profile_not_deletable"


def test_delete_current_default_profile_is_forbidden(client, app):
    created = create_profile(client, default_slot=DEFAULT_SLOT_GLOBAL)
    response = client.delete(f"{API_PREFIX}/{created['id']}")
    assert response.status_code == 409
    assert response.json()["reason"] == "default_profile_not_deletable"


def test_update_system_managed_profile_allows_only_default_slot_reselect(client, app):
    """system_managed Profile 允许仅 PATCH default_slot 重新选择默认，
    但禁止连接/凭据及其他任何字段变化。"""
    system = create_profile(client, name="system")
    asyncio.run(
        _update_stored_profile(
            app, system["id"], {"system_managed": True, "default_slot": None}
        )
    )
    other = create_profile(client, name="other")

    response = client.patch(
        f"{API_PREFIX}/{other['id']}", json={"default_slot": "global"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["default_slot"] == "global"

    # 仅重新选择默认：允许。
    response = client.patch(
        f"{API_PREFIX}/{system['id']}", json={"default_slot": "global"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["default_slot"] == "global"
    assert response.json()["is_default"] is True
    stored_other = asyncio.run(_stored_profile(app, other["id"]))
    assert stored_other.default_slot is None
    stored_system = asyncio.run(_stored_profile(app, system["id"]))
    # 连接/凭据/其他字段必须保持不变。
    assert stored_system.endpoint == "https://s3.example.com"
    assert stored_system.config_version == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"description": "blocked"},
        {"endpoint": "https://s3-new.example.com"},
        {"access_key": "new-access"},
        {"secret_key": "new-secret"},
        {"prefix": "datamodel/other"},
        {"tls_enabled": False},
        {"source_fallback_enabled": False},
        {"name": "renamed"},
    ],
    ids=[
        "description",
        "endpoint",
        "access_key",
        "secret_key",
        "prefix",
        "tls_enabled",
        "source_fallback_enabled",
        "name",
    ],
)
def test_update_system_managed_profile_rejects_non_default_slot_fields(
    client, app, payload
):
    created = create_profile(client)
    asyncio.run(_update_stored_profile(app, created["id"], {"system_managed": True}))
    before = asyncio.run(_stored_profile(app, created["id"]))

    response = client.patch(f"{API_PREFIX}/{created['id']}", json=payload)

    assert response.status_code == 403
    assert response.json()["reason"] == "system_profile_read_only"
    after = asyncio.run(_stored_profile(app, created["id"]))
    assert after.model_dump(exclude={"updated_at"}) == before.model_dump(
        exclude={"updated_at"}
    )


def test_delete_non_default_non_system_profile_allowed(client, app):
    created = create_profile(client)
    response = client.delete(f"{API_PREFIX}/{created['id']}")
    assert response.status_code == 200


async def _stored_profile(app, profile_id):
    session_override = app.dependency_overrides[get_session]
    async for session in session_override():
        result = await session.exec(
            select(ModelPreheatS3Profile).where(ModelPreheatS3Profile.id == profile_id)
        )
        return result.one()


async def _update_stored_profile(app, profile_id, values):
    session_override = app.dependency_overrides[get_session]
    async for session in session_override():
        profile = await session.get(ModelPreheatS3Profile, profile_id)
        for key, value in values.items():
            setattr(profile, key, value)
        session.add(profile)
        await session.commit()
