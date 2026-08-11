import hashlib

import pytest

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    decode_path,
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


def _manifest(root, identity=None):
    return build_model_preheat_manifest(
        root,
        identity or _identity(),
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="parent-generation-id",
        exclude_patterns=["*.tmp"],
    )


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
    assert manifest.files[1].sha256 == hashlib.sha256(b"weights").hexdigest()


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
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="generation-id",
        exclude_patterns=["**/*.tmp"],
    )

    assert identity.file_patterns == ()
    assert manifest.to_dict()["include_patterns"] == []
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
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="generation-id",
        exclude_patterns=exclude,
    )

    assert [decode_path(file.path) for file in manifest.files] == expected


def test_manifest_protocol_fields_are_present(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    manifest = _manifest(tmp_path)
    payload = manifest.to_dict()

    assert payload["cache_key"] == "cache-key"
    assert payload["generation_id"] == "parent-generation-id"
    assert payload["source"] == "modelscope"
    assert payload["model_id"] == "Qwen/Qwen%202.5"
    assert payload["requested_revision"] == "main"
    assert payload["resolved_revision"] == "main"
    assert payload["include_patterns"] == ["config.json", "weights/*.bin"]
    assert payload["exclude_patterns"] == ["*.tmp"]
    assert payload["selection_digest"] == "selection-digest"
    assert payload["file_count"] == 2
    assert payload["total_size"] == len(b"weights") + len(b"config")
    assert payload["aggregate_sha256"] == manifest.aggregate_sha256
    assert len(payload["files"]) == 2


def test_manifest_preserves_requested_revision_separately_from_resolved_revision(
    tmp_path,
):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    manifest = build_model_preheat_manifest(
        tmp_path,
        _identity("resolved-commit"),
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="parent-generation-id",
        requested_revision="release branch",
    )

    assert manifest.to_dict()["requested_revision"] == "release%20branch"
    assert manifest.to_dict()["resolved_revision"] == "resolved-commit"


def test_manifest_digest_is_stable_for_same_content_and_identity(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "b.bin").write_bytes(b"b")
    (tmp_path / "weights" / "a.bin").write_bytes(b"a")
    (tmp_path / "config.json").write_bytes(b"config")

    first = _manifest(tmp_path)
    second = _manifest(tmp_path)

    assert first.digest == second.digest
    assert first.to_json_bytes() == second.to_json_bytes()


def test_manifest_digest_changes_when_revision_changes(tmp_path):
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "model.bin").write_bytes(b"weights")
    (tmp_path / "config.json").write_bytes(b"config")

    main = _manifest(tmp_path, _identity("main"))
    pinned = _manifest(tmp_path, _identity("v1"))

    assert main.digest != pinned.digest


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
            cache_key="cache-key",
            selection_digest="selection-digest",
            generation_id="generation-id",
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
            cache_key="cache-key",
            selection_digest="selection-digest",
            generation_id="generation-id",
        )
    with pytest.raises(
        ModelPreheatManifestError, match="manifest_total_size_too_large"
    ):
        ModelPreheatManifest(
            identity=identity,
            files=(ManifestFile(path=file.path, size=2**50 + 1, sha256=file.sha256),),
            cache_key="cache-key",
            selection_digest="selection-digest",
            generation_id="generation-id",
        )
    with pytest.raises(ModelPreheatManifestError, match="manifest_path_too_long"):
        ManifestFile(path=f"{'a' * 1021}.bin", size=1, sha256="0" * 64)
