import pytest
from pydantic import ValidationError

from datetime import datetime, timezone

from gpustack.schemas.models import BackendEnum, ModelCreate, ModelPublic, SourceEnum


def test_model_scope_accepts_file_path_alias_for_gguf_model():
    model = ModelCreate(
        name="qwen-gguf",
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        file_path="Qwen3.6-27B-UD-Q4_K_XL.gguf",
        backend=BackendEnum.LLAMA_BOX,
    )

    assert model.model_scope_file_path == "Qwen3.6-27B-UD-Q4_K_XL.gguf"


def test_model_scope_gguf_requires_file_path():
    with pytest.raises(ValidationError, match="file_path must be provided"):
        ModelCreate(
            name="qwen-gguf",
            source=SourceEnum.MODEL_SCOPE,
            model_scope_model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
            backend=BackendEnum.LLAMA_BOX,
        )


def test_model_public_exposes_file_path_alias():
    model = ModelPublic(
        id=1,
        name="qwen-gguf",
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        model_scope_file_path="Qwen3.6-27B-UD-Q4_K_XL.gguf",
        backend=BackendEnum.LLAMA_BOX,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    assert model.file_path == "Qwen3.6-27B-UD-Q4_K_XL.gguf"
