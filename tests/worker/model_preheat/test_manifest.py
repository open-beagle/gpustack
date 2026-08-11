import hashlib

import pytest

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)
from gpustack.worker.model_preheat.manifest import (
    ManifestFile,
    ModelPreheatManifest,
    ModelPreheatManifestError,
    build_model_preheat_manifest,
)


def _identity(revision: str = "main") -> ModelPreheatIdentity:
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="Qwen/Qwen 2.5",
        revision=revision,
        file_patterns=["weights/*.bin", "config.json"],
    )


def test_manifest_records_file_sha256_size_and_encoded_paths(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model 1.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b'{"model":"qwen"}')

    manifest = build_model_preheat_manifest(tmp_path, _identity())

    assert [file.path for file in manifest.files] == [
        "config.json",
        "weights/model%201.bin",
    ]
    assert manifest.total_size == len(b'{"model":"qwen"}') + len(b"weights")
    assert manifest.files[1].sha256 == hashlib.sha256(b"weights").hexdigest()


def test_manifest_protocol_fields_are_present(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    manifest = build_model_preheat_manifest(tmp_path, _identity())
    payload = manifest.to_dict()

    assert payload["cache_key"] == manifest.identity.digest
    assert payload["generation_id"] == manifest.digest
    assert payload["source"] == "modelscope"
    assert payload["model_id"] == "Qwen/Qwen%202.5"
    assert payload["requested_revision"] == "main"
    assert payload["resolved_revision"] == "main"
    assert payload["include_patterns"] == ["config.json", "weights/*.bin"]
    assert payload["exclude_patterns"] == []
    assert payload["selection_digest"] == manifest.identity.digest
    assert payload["file_count"] == 2
    assert payload["total_size"] == len(b"weights") + len(b"config")
    assert payload["aggregate_sha256"] == manifest.aggregate_sha256
    assert len(payload["files"]) == 2


def test_manifest_digest_is_stable_for_same_content_and_identity(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "b.bin").write_bytes(b"b")
    (tmp_path / "weights" / "a.bin").write_bytes(b"a")
    (tmp_path / "config.json").write_bytes(b"config")

    first = build_model_preheat_manifest(tmp_path, _identity())
    second = build_model_preheat_manifest(tmp_path, _identity())

    assert first.digest == second.digest
    assert first.to_json_bytes() == second.to_json_bytes()


def test_manifest_digest_changes_when_revision_changes(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    main = build_model_preheat_manifest(tmp_path, _identity("main"))
    pinned = build_model_preheat_manifest(tmp_path, _identity("v1"))

    assert main.digest != pinned.digest


def test_manifest_rejects_traversal_patterns(tmp_path):
    with pytest.raises((ModelPreheatManifestError, ModelPreheatIdentityError)):
        identity = ModelPreheatIdentity(
            source="modelscope",
            model_id="org/model",
            revision="main",
            file_patterns=["../secret.bin"],
        )
        build_model_preheat_manifest(tmp_path, identity)


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
