import hashlib
import json

import pytest

from gpustack.schemas.models import SourceEnum
from gpustack.worker.model_preheat import identity as identity_module
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    encode_path,
)
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    ModelPreheatManifest,
    compute_artifact_id,
    compute_request_digest,
)

MAX_FILE_PATTERNS = getattr(identity_module, "MAX_FILE_PATTERNS", 128)
MAX_PATTERN_LENGTH = getattr(identity_module, "MAX_PATTERN_LENGTH", 1024)


def test_path_segments_are_rfc3986_encoded_without_collapsing_boundaries():
    assert encode_path("Qwen/Qwen 2.5#Chat") == "Qwen/Qwen%202.5%23Chat"
    assert encode_path("中文/模型.bin") == "%E4%B8%AD%E6%96%87/%E6%A8%A1%E5%9E%8B.bin"


def test_identity_exposes_pattern_limits():
    assert identity_module.MAX_FILE_PATTERNS == MAX_FILE_PATTERNS
    assert identity_module.MAX_PATTERN_LENGTH == MAX_PATTERN_LENGTH


def test_identity_request_digest_is_stable_after_pattern_normalization():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision="8f73c6a9",
        file_patterns=["weights/*.safetensors", "config.json"],
    )
    same_identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision="8f73c6a9",
        file_patterns=["config.json", "weights/*.safetensors"],
    )

    assert identity.model_path == "Qwen/Qwen%202.5"
    assert identity.file_patterns == ("config.json", "weights/*.safetensors")
    assert identity.request_digest == same_identity.request_digest


def test_modelscope_source_aliases_share_canonical_identity_and_prefix():
    base = {
        "model_id": "Qwen/Qwen 2.5",
        "revision": "8f73c6a9",
        "file_patterns": ["config.json"],
    }

    from_enum = ModelPreheatIdentity(source=SourceEnum.MODEL_SCOPE, **base)
    with_underscore = ModelPreheatIdentity(source="model_scope", **base)
    canonical = ModelPreheatIdentity(source="modelscope", **base)

    assert SourceEnum.MODEL_SCOPE.value == "model_scope"
    assert from_enum.source == "modelscope"
    assert from_enum.model_path == canonical.model_path
    assert (
        from_enum.request_digest
        == canonical.request_digest
        == with_underscore.request_digest
    )
    assert (
        from_enum.artifact_prefix("storage")
        == canonical.artifact_prefix("storage")
        == with_underscore.artifact_prefix("storage")
    )


def test_identity_rejects_unknown_source():
    with pytest.raises(ModelPreheatIdentityError, match="invalid_source"):
        ModelPreheatIdentity(
            source="ollama_library",
            model_id="org/model",
            revision="main",
            file_patterns=["config.json"],
        )


def test_resolved_revision_is_encoded_into_revision_path():
    main = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["*.gguf"],
    )
    pinned = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="refs/pr/1",
        file_patterns=["*.gguf"],
    )

    assert main.revision_path == "8f73c6a9"
    # 多段 revision 保留 `/` 作为路径分隔符，段内特殊字符才编码。
    assert pinned.revision_path == "refs/pr/1"
    assert main.revision_path != pinned.revision_path
    # Artifact 前缀不直接包含 revision 段（revision 已编码进 artifact_id）。
    assert "8f73c6a9" not in main.artifact_prefix("pfx")
    assert "refs" not in pinned.artifact_prefix("pfx")


def test_request_digest_matches_canonical_request_payload():
    # model_id 含空格，编码后为 org/model%20one，验证 request_digest
    # 使用编码后的规范值而不是原始字符串。
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model one",
        revision="8f73c6a9",
        file_patterns=["weights/*.bin", "config.json"],
        requested_revision="master",
        exclude_patterns=["*.tmp"],
    )
    assert identity.model_path == "org/model%20one"

    expected = hashlib.sha256(
        json.dumps(
            {
                "source": "modelscope",
                "model_id": "org/model%20one",
                "requested_revision": "master",
                "include_patterns": ["config.json", "weights/*.bin"],
                "exclude_patterns": ["*.tmp"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert identity.request_digest == expected
    assert identity.request_digest == compute_request_digest(
        "modelscope",
        "org/model%20one",
        "master",
        ["weights/*.bin", "config.json"],
        ["*.tmp"],
    )
    # 使用未编码的原始 model_id 会得到不同摘要，证明契约要求规范编码值。
    assert (
        compute_request_digest(
            "modelscope",
            "org/model one",
            "master",
            ["config.json", "weights/*.bin"],
            ["*.tmp"],
        )
        != identity.request_digest
    )


def test_moving_revision_only_affects_request_identity_not_artifact_identity():
    via_branch = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
        requested_revision="master",
    )
    via_commit = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
        requested_revision="8f73c6a9",
    )
    no_requested = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
    )

    # 移动 revision 只进入请求身份，不同 requested_revision 的请求摘要不同。
    assert via_branch.request_digest != via_commit.request_digest
    # 但 Artifact 身份（前缀）与请求摘要完全一致。
    assert via_branch.artifact_prefix("pfx") == via_commit.artifact_prefix("pfx")
    assert via_branch.artifact_prefix("pfx") == no_requested.artifact_prefix("pfx")
    # main/master 不进入 Artifact 身份前缀。
    assert "master" not in via_branch.artifact_prefix("pfx")
    assert "main" not in no_requested.artifact_prefix("pfx")


def test_artifact_id_changes_with_resolved_revision_file_selection_and_digest():
    base_identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["*.bin"],
    )

    def build(identity, sha: str) -> ModelPreheatManifest:
        return ModelPreheatManifest(
            identity=identity,
            files=(ManifestFile(path="model.bin", size=10, sha256=sha),),
        )

    reference = build(base_identity, "a" * 64)
    different_revision = build(
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="9d1e2f30",
            file_patterns=["*.bin"],
        ),
        "a" * 64,
    )
    different_selection = build(
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="8f73c6a9",
            file_patterns=["config.json"],
        ),
        "a" * 64,
    )
    different_file_digest = build(base_identity, "b" * 64)

    assert (
        reference.artifact_id
        != different_revision.artifact_id
        != different_selection.artifact_id
        != different_file_digest.artifact_id
    )
    # artifact_id 是 64 位小写十六进制。
    assert len(reference.artifact_id) == 64
    int(reference.artifact_id, 16)

    # 与规范 payload 的 SHA-256 一致（设计文档第 7.3 节）。
    expected = hashlib.sha256(
        json.dumps(
            {
                "source": "modelscope",
                "model_id": "org/model",
                "resolved_revision": "8f73c6a9",
                "include_patterns": ["*.bin"],
                "exclude_patterns": [],
                "files": [{"path": "model.bin", "size": 10, "sha256": "a" * 64}],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert reference.artifact_id == expected
    assert reference.artifact_id == compute_artifact_id(
        "modelscope", "org/model", "8f73c6a9", ["*.bin"], [], reference.files
    )

    # requested_revision 不参与 Artifact ID。
    via_master = build(
        ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="8f73c6a9",
            file_patterns=["*.bin"],
            requested_revision="master",
        ),
        "a" * 64,
    )
    assert via_master.artifact_id == reference.artifact_id


def test_artifact_prefix_requires_explicit_prefix_argument():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
    )
    # 空串是合法的“无前缀”，结果不含任何前导斜杠。
    bare = identity.artifact_prefix("")
    assert bare == "modelscope/org/model"
    assert not bare.startswith("/")


@pytest.mark.parametrize(
    "profile_prefix,expected_prefix",
    [
        ("storage", "storage/modelscope/org/model"),
        ("team/a/b", "team/a/b/modelscope/org/model"),
        ("/storage/", "storage/modelscope/org/model"),
    ],
)
def test_artifact_prefix_transmits_profile_prefix_safely(
    profile_prefix, expected_prefix
):
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
    )
    assert identity.artifact_prefix(profile_prefix) == expected_prefix


@pytest.mark.parametrize(
    "bad_prefix",
    [
        "a/../b",
        "a/..",
        "..",
        "a//b",
        "pre\\fix",
        "pre\x1fix",
    ],
)
def test_artifact_prefix_rejects_unsafe_profile_prefix(bad_prefix):
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
    )
    with pytest.raises(ModelPreheatIdentityError):
        identity.artifact_prefix(bad_prefix)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/abs",
        "a//b",
        "a/../b",
        "..",
        "a/\x1fb",
    ],
)
def test_path_validation_rejects_unsafe_segments(bad_path):
    with pytest.raises(ModelPreheatIdentityError):
        encode_path(bad_path)


@pytest.mark.parametrize(
    "bad_segment",
    ["prefix/..", "..", "prefix//model", "pre\\fix/model", "pre\x1fix/model"],
)
def test_artifact_prefix_rejects_unsafe_profile_segments(bad_segment):
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a9",
        file_patterns=["config.json"],
    )
    with pytest.raises(ModelPreheatIdentityError):
        identity.artifact_prefix(bad_segment)


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
