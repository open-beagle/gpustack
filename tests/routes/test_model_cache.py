from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from gpustack.api import exceptions
from gpustack.api.auth import get_admin_user
from gpustack.api.exceptions import ConflictException
from gpustack.routes import model_cache, model_files
from gpustack.schemas.model_cache import ModelCacheTaskStateEnum
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.schemas.users import User
from gpustack.server.db import get_session


def _app():
    app = FastAPI()

    async def session_override():
        yield SimpleNamespace()

    async def admin_override():
        return User(id=7, username="admin", is_admin=True, hashed_password="")

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_admin_user] = admin_override
    app.include_router(model_files.router, prefix="/v1/model-files")
    exceptions.register_handlers(app)
    return app


def test_cache_post_has_no_request_body_and_ignores_client_model_id(monkeypatch):
    captured = {}

    async def create_task(request, session, current_user, model_file_id):
        captured.update(model_file_id=model_file_id, user_id=current_user.id)
        now = datetime.now(timezone.utc)
        return {
            "id": 13,
            "model_file_id": model_file_id,
            "worker_id": 2,
            "model_id": "Qwen/Qwen3",
            "target_path": "cache/modelscope/Qwen/Qwen3/",
            "source_paths": ["/models/Qwen3"],
            "state": ModelCacheTaskStateEnum.PENDING,
            "progress": 0,
            "uploaded_size": 0,
            "total_size": 10,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
        }

    monkeypatch.setattr(model_files, "create_model_cache_task", create_task)
    app = _app()
    operation = app.openapi()["paths"]["/v1/model-files/{id}/cache"]["post"]
    assert "requestBody" not in operation

    with TestClient(app) as client:
        response = client.post(
            "/v1/model-files/42/cache",
            json={"model_id": "attacker/override"},
        )

    assert response.status_code == 200, response.text
    assert captured == {"model_file_id": 42, "user_id": 7}
    assert response.json()["model_id"] == "Qwen/Qwen3"


@pytest.mark.asyncio
async def test_cache_preview_derives_model_id_and_source_namespaced_path(monkeypatch):
    model_file = ModelFile(
        id=42,
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        state=ModelFileStateEnum.READY,
        resolved_paths=["/models/config.json", "/models/model.gguf"],
        size=123,
    )

    async def one_by_id(session, model_file_id):
        assert model_file_id == 42
        return model_file

    monkeypatch.setattr(ModelFile, "one_by_id", one_by_id)
    monkeypatch.setattr(
        model_cache,
        "_service",
        lambda request: SimpleNamespace(
            s3_path=lambda model_id: f"s3://bd-wind/cache/modelscope/{model_id}/"
        ),
    )

    preview = await model_cache.get_model_cache_preview(
        SimpleNamespace(), SimpleNamespace(), 42
    )

    assert preview.model_id == "unsloth/Qwen3.6-27B-MTP-GGUF"
    assert preview.s3_path == (
        "s3://bd-wind/cache/modelscope/unsloth/Qwen3.6-27B-MTP-GGUF/"
    )
    assert preview.file_count == 2
    assert preview.total_size == 123


@pytest.mark.asyncio
async def test_cache_preview_rejects_non_modelscope_source(monkeypatch):
    model_file = ModelFile(
        id=42,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        state=ModelFileStateEnum.READY,
        resolved_paths=["/models/model.gguf"],
    )

    async def one_by_id(session, model_file_id):
        return model_file

    monkeypatch.setattr(ModelFile, "one_by_id", one_by_id)

    with pytest.raises(ConflictException) as exc_info:
        await model_cache.get_model_cache_preview(
            SimpleNamespace(), SimpleNamespace(), 42
        )

    assert exc_info.value.message == "model_cache_requires_modelscope_source"
