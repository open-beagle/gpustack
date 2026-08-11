import pytest

from gpustack.schemas.models import SourceEnum
from gpustack.worker.model_preheat import identity as identity_module
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    encode_path,
)

MAX_FILE_PATTERNS = getattr(identity_module, "MAX_FILE_PATTERNS", 128)
MAX_PATTERN_LENGTH = getattr(identity_module, "MAX_PATTERN_LENGTH", 1024)


def test_path_segments_are_rfc3986_encoded_without_collapsing_boundaries():
    assert encode_path("Qwen/Qwen 2.5#Chat") == "Qwen/Qwen%202.5%23Chat"
    assert encode_path("中文/模型.bin") == "%E4%B8%AD%E6%96%87/%E6%A8%A1%E5%9E%8B.bin"


def test_identity_exposes_pattern_limits():
    assert identity_module.MAX_FILE_PATTERNS == MAX_FILE_PATTERNS
    assert identity_module.MAX_PATTERN_LENGTH == MAX_PATTERN_LENGTH


def test_identity_digest_is_stable_after_pattern_normalization():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision="main",
        file_patterns=["weights/*.safetensors", "config.json"],
    )
    same_identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision="main",
        file_patterns=["config.json", "weights/*.safetensors"],
    )

    assert identity.model_path == "Qwen/Qwen%202.5"
    assert identity.file_patterns == ("config.json", "weights/*.safetensors")
    assert identity.digest == same_identity.digest


def test_modelscope_source_aliases_share_canonical_digest_and_prefix():
    base = {
        "model_id": "Qwen/Qwen 2.5",
        "revision": "main",
        "file_patterns": ["config.json"],
    }

    from_enum = ModelPreheatIdentity(source=SourceEnum.MODEL_SCOPE, **base)
    with_underscore = ModelPreheatIdentity(source="model_scope", **base)
    canonical = ModelPreheatIdentity(source="modelscope", **base)

    assert SourceEnum.MODEL_SCOPE.value == "model_scope"
    assert from_enum.source == "modelscope"
    assert with_underscore.digest == canonical.digest == from_enum.digest
    assert with_underscore.storage_prefix == canonical.storage_prefix


def test_identity_rejects_unknown_source():
    with pytest.raises(ModelPreheatIdentityError, match="invalid_source"):
        ModelPreheatIdentity(
            source="ollama_library",
            model_id="org/model",
            revision="main",
            file_patterns=["config.json"],
        )


def test_revision_is_part_of_identity_and_storage_prefix():
    main = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="main",
        file_patterns=["*.gguf"],
    )
    pinned = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="refs/pr/1",
        file_patterns=["*.gguf"],
    )

    assert main.digest != pinned.digest
    assert main.storage_prefix != pinned.storage_prefix
    assert "main" in main.storage_prefix
    assert "refs/pr/1" in pinned.storage_prefix


@pytest.mark.parametrize(
    "bad_path",
    ["", "/abs", "a//b", "a/../b", "..", "a/\x1fb"],
)
def test_path_validation_rejects_unsafe_segments(bad_path):
    with pytest.raises(ModelPreheatIdentityError):
        encode_path(bad_path)


def test_identity_rejects_duplicate_normalized_patterns():
    with pytest.raises(ModelPreheatIdentityError, match="duplicate_path"):
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=["config.json", "config.json"],
        )


def test_identity_rejects_too_many_patterns():
    with pytest.raises(ModelPreheatIdentityError, match="too_many_patterns"):
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=[
                f"file-{index}.bin" for index in range(MAX_FILE_PATTERNS + 1)
            ],
        )


def test_identity_rejects_too_long_pattern():
    with pytest.raises(ModelPreheatIdentityError, match="pattern_too_long"):
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=["a" * (MAX_PATTERN_LENGTH + 1)],
        )


@pytest.mark.parametrize(
    "bad_pattern",
    [
        "/absolute/*.bin",
        "../secret.bin",
        "weights\\model.bin",
        "weights/\x1f.bin",
        "weights/**.bin",
        "weights/[abc.bin",
    ],
)
def test_identity_rejects_unsafe_or_invalid_glob_patterns(bad_pattern):
    with pytest.raises(ModelPreheatIdentityError):
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=[bad_pattern],
        )
