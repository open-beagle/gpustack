import hashlib
import json

import pytest

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    decode_path,
)
from gpustack.worker.model_preheat.manifest import (
    ARTIFACT_MANIFEST_FIELDS,
    ManifestFile,
    ModelPreheatManifest,
    ModelPreheatManifestError,
    build_model_preheat_manifest,
    parse_artifact_manifest,
)


def _identity(revision: str = "8f73c6a91b") -> ModelPreheatIdentity:
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision=revision,
        file_patterns=["weights/*.bin", "config.json"],
        requested_revision="master",
    )


def _manifest(root, identity=None):
    return build_model_preheat_manifest(
        root,
        identity or _identity(),
        exclude_patterns=["*.tmp"],
    )


def _file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_manifest_records_file_sha256_size_and_encoded_paths(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model 1.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b'{"model":"qwen"}')

    manifest = _manifest(tmp_path)

    assert [file.path for file in manifest.files] == [
        "config.json",
        "weights/model%201.bin",
    ]
    assert manifest.total_size == len(b'{"model":"qwen"}') + len(b"weights")
    assert manifest.files[1].sha256 == _file_sha256(b"weights")


def test_empty_include_selects_all_files_then_applies_excludes(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "weights" / "ignored.tmp").write_bytes(b"ignored")
    (tmp_path / "config.json").write_bytes(b"config")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=[],
    )

    manifest = build_model_preheat_manifest(
        tmp_path,
        identity,
        exclude_patterns=["**/*.tmp"],
    )

    assert identity.file_patterns == ()
    assert manifest.to_artifact_dict()["include_patterns"] == []
    assert [file.path for file in manifest.files] == [
        "config.json",
        "weights/model.bin",
    ]


@pytest.mark.parametrize(
    ("include", "exclude", "expected"),
    [
        (["*.json"], [], ["config.json", "nested/config.json"]),
        ([], ["*.tmp"], ["config.json", "nested/config.json"]),
        (["*.json"], ["nested/*"], ["config.json"]),
    ],
)
def test_manifest_uses_hub_glob_semantics_for_root_and_nested_paths(
    tmp_path, include, exclude, expected
):
    (tmp_path / "nested").mkdir()
    (tmp_path / "config.json").write_text("root")
    (tmp_path / "nested" / "config.json").write_text("nested")
    (tmp_path / "nested" / "ignored.tmp").write_text("ignored")
    identity = ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="a" * 40,
        file_patterns=include,
    )

    manifest = build_model_preheat_manifest(
        tmp_path,
        identity,
        exclude_patterns=exclude,
    )

    assert [decode_path(file.path) for file in manifest.files] == expected


def test_artifact_manifest_contains_only_design_section_8_fields(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    manifest = _manifest(tmp_path)
    payload = manifest.to_artifact_dict()

    assert set(payload) == ARTIFACT_MANIFEST_FIELDS
    assert payload["schema_version"] == 1
    assert payload["source"] == "modelscope"
    assert payload["model_id"] == "Qwen/Qwen%202.5"
    assert payload["resolved_revision"] == "8f73c6a91b"
    assert payload["include_patterns"] == ["config.json", "weights/*.bin"]
    assert payload["exclude_patterns"] == ["*.tmp"]
    assert payload["file_count"] == 2
    assert payload["total_size"] == len(b"weights") + len(b"config")
    assert payload["artifact_id"] == manifest.artifact_id
    # requested_revision 不写入不可变 Manifest。
    assert "requested_revision" not in payload
    # generation 协议字段不写入统一 Artifact Manifest。
    assert "cache_key" not in payload
    assert "selection_digest" not in payload
    assert "generation_id" not in payload
    assert "identity" not in payload


def test_requested_revision_does_not_change_artifact_manifest(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    identity_master = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
        requested_revision="master",
    )
    identity_commit = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
        requested_revision="8f73c6a91b",
    )

    via_master = build_model_preheat_manifest(tmp_path, identity_master)
    via_commit = build_model_preheat_manifest(tmp_path, identity_commit)

    # 请求身份不同……
    assert identity_master.request_digest != identity_commit.request_digest
    # ……但相同内容通过 master 和 Commit SHA 发布必须得到相同 Manifest。
    assert via_master.to_artifact_json_bytes() == via_commit.to_artifact_json_bytes()
    assert via_master.artifact_id == via_commit.artifact_id


def test_artifact_manifest_json_is_canonical(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "b.bin").write_bytes(b"b")
    (tmp_path / "weights" / "a.bin").write_bytes(b"a")
    (tmp_path / "config.json").write_bytes(b"config")

    first = _manifest(tmp_path)
    second = _manifest(tmp_path)

    assert first.to_artifact_json_bytes() == second.to_artifact_json_bytes()
    # 往返解析必须保持字段精确一致。
    parsed = parse_artifact_manifest(
        json.loads(first.to_artifact_json_bytes().decode("utf-8"))
    )
    assert parsed.to_artifact_dict() == first.to_artifact_dict()
    assert parsed.artifact_id == first.artifact_id


def test_artifact_id_changes_when_revision_changes(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    main = _manifest(tmp_path, _identity("8f73c6a91b"))
    pinned = _manifest(tmp_path, _identity("9d1e2f3040"))

    assert main.artifact_id != pinned.artifact_id


def test_manifest_rejects_unsupported_schema_version():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=["*.bin"],
    )
    file = ManifestFile(path="model.bin", size=1, sha256="0" * 64)

    with pytest.raises(ModelPreheatManifestError, match="unsupported_schema_version"):
        ModelPreheatManifest(identity=identity, files=(file,), schema_version=2)
    with pytest.raises(ModelPreheatManifestError, match="unsupported_schema_version"):
        ModelPreheatManifest(identity=identity, files=(file,), schema_version=0)


def test_parse_artifact_manifest_requires_schema_version_strictly_one(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
    )
    manifest = build_model_preheat_manifest(tmp_path, identity)

    def payload_for(schema_version):
        payload = json.loads(manifest.to_artifact_json_bytes().decode("utf-8"))
        payload["schema_version"] = schema_version
        return payload

    for bad in (0, 2, 1.0, "1", None, True):
        with pytest.raises(ModelPreheatManifestError):
            parse_artifact_manifest(payload_for(bad))


def test_parse_artifact_manifest_rejects_legacy_or_tampered_payload(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
    )
    manifest = build_model_preheat_manifest(tmp_path, identity)
    payload = manifest.to_artifact_dict()

    # 旧协议 Manifest（带 generation 字段）不是合法统一 Artifact。
    legacy = dict(payload)
    legacy["cache_key"] = "cache-key"
    legacy["generation_id"] = "generation-id"
    with pytest.raises(ModelPreheatManifestError, match="s3_manifest_invalid"):
        parse_artifact_manifest(legacy)

    # 篡改 artifact_id 被拒绝。
    tampered = dict(payload)
    tampered["artifact_id"] = "0" * 64
    with pytest.raises(ModelPreheatManifestError, match="s3_manifest_invalid"):
        parse_artifact_manifest(tampered)

    # 缺失字段被拒绝。
    missing = dict(payload)
    del missing["file_count"]
    with pytest.raises(ModelPreheatManifestError, match="s3_manifest_invalid"):
        parse_artifact_manifest(missing)

    # 文件摘要与 artifact_id 不一致被拒绝。
    inconsistent = dict(payload)
    inconsistent["files"] = [dict(file) for file in payload["files"]]
    inconsistent["files"][0]["sha256"] = "1" * 64
    with pytest.raises(ModelPreheatManifestError, match="s3_manifest_invalid"):
        parse_artifact_manifest(inconsistent)


def test_parse_artifact_manifest_rejects_invalid_sha256_type_and_case(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
    )
    manifest = build_model_preheat_manifest(tmp_path, identity)
    payload = manifest.to_artifact_dict()

    # 大写十六进制被拒绝。
    upper = dict(payload)
    upper["files"] = [dict(f, sha256=f["sha256"].upper()) for f in payload["files"]]
    with pytest.raises(ModelPreheatManifestError):
        parse_artifact_manifest(upper)

    # 非字符串 sha256 被拒绝。
    not_str = dict(payload)
    not_str["files"] = [
        dict(f, sha256=int("a", 16) * (16**63)) for f in payload["files"]
    ]
    with pytest.raises(ModelPreheatManifestError):
        parse_artifact_manifest(not_str)

    # 长度错误的 sha256 被拒绝。
    short = dict(payload)
    short["files"] = [dict(f, sha256="a" * 63) for f in payload["files"]]
    with pytest.raises(ModelPreheatManifestError):
        parse_artifact_manifest(short)


def test_manifest_file_sha256_must_be_lowercase_hex():
    with pytest.raises(ModelPreheatManifestError):
        ManifestFile(path="model.bin", size=1, sha256="A" * 64)
    with pytest.raises(ModelPreheatManifestError):
        ManifestFile(path="model.bin", size=1, sha256="g" * 64)
    with pytest.raises(ModelPreheatManifestError):
        ManifestFile(path="model.bin", size=1, sha256=12345)
    with pytest.raises(ModelPreheatManifestError):
        ManifestFile(path="model.bin", size=1, sha256="a" * 63)


def test_manifest_rejects_traversal_patterns(tmp_path):
    with pytest.raises((ModelPreheatManifestError, ModelPreheatIdentityError)):
        identity = ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=["../secret.bin"],
        )
        _manifest(tmp_path, identity)


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "/absolute.bin",
        "weights\\model.bin",
        "weights/\x1f.bin",
        ".",
        "..",
        "weights/../secret.bin",
        "%2E%2E/secret.bin",
    ],
)
def test_manifest_file_rejects_invalid_or_encoded_traversal_path(bad_path):
    with pytest.raises(ModelPreheatManifestError):
        ManifestFile(
            path=bad_path,
            size=1,
            sha256=hashlib.sha256(b"x").hexdigest(),
        )


def test_manifest_rejects_duplicate_file_paths():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=["*.bin"],
    )

    with pytest.raises(ModelPreheatManifestError, match="duplicate_path"):
        ModelPreheatManifest(
            identity=identity,
            files=(
                ManifestFile(
                    path="model.bin",
                    size=1,
                    sha256=hashlib.sha256(b"a").hexdigest(),
                ),
                ManifestFile(
                    path="model.bin",
                    size=1,
                    sha256=hashlib.sha256(b"a").hexdigest(),
                ),
            ),
        )


def test_manifest_rejects_file_count_path_and_total_size_limits():
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="main",
        file_patterns=[],
    )
    file = ManifestFile(path="model.bin", size=1, sha256="0" * 64)

    with pytest.raises(ModelPreheatManifestError, match="too_many_manifest_files"):
        ModelPreheatManifest(
            identity=identity,
            files=tuple(
                ManifestFile(path=f"{index}.bin", size=1, sha256="0" * 64)
                for index in range(1025)
            ),
        )
    with pytest.raises(
        ModelPreheatManifestError, match="manifest_total_size_too_large"
    ):
        ModelPreheatManifest(
            identity=identity,
            files=(ManifestFile(path=file.path, size=2**50 + 1, sha256=file.sha256),),
        )
    with pytest.raises(ModelPreheatManifestError, match="manifest_path_too_long"):
        ManifestFile(path=f"{'a' * 1021}.bin", size=1, sha256="0" * 64)


def test_manifest_artifact_prefix_uses_unified_object_layout(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"weights")
    identity = ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="8f73c6a91b",
        file_patterns=["model.bin"],
        requested_revision="master",
    )
    manifest = build_model_preheat_manifest(tmp_path, identity)

    prefix = manifest.artifact_prefix("model-storage")
    assert prefix == (
        "model-storage/modelscope/org/model/" f"{manifest.artifact_id}"
    ).rstrip("/")
    # 不含 resolved revision 路径段、requested_revision、generation 或协议版本目录。
    assert "8f73c6a91b" not in prefix
    assert "master" not in prefix
    assert "generation" not in prefix
    assert "/v1/" not in prefix
    assert "cache_key" not in prefix
