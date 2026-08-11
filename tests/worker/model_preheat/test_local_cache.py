import os

import pytest
from filelock import SoftFileLock

from gpustack.worker.model_preheat import local_cache
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.local_cache import (
    LocalCacheError,
    LocalCacheState,
    create_staging_dir,
    inspect_local_cache,
    model_lock_path,
    publish_staging,
    trusted_manifest_path,
    write_trusted_manifest,
)
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest


def _identity(revision="main"):
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision=revision,
        file_patterns=["config.json", "weights/model.bin"],
    )


def _write_model(root, content=b"weights"):
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "weights" / "model.bin").write_bytes(content)
    (root / "config.json").write_bytes(b"config")


def _manifest(root, revision="main"):
    _write_model(root)
    return build_model_preheat_manifest(root, _identity(revision))


def test_inspect_distinguishes_missing_candidate_valid_and_conflict(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    cache_key = "cache-key"

    assert (
        inspect_local_cache(cache_dir, target, cache_key).state
        == LocalCacheState.MISSING
    )

    manifest = _manifest(target)
    assert (
        inspect_local_cache(cache_dir, target, cache_key).state
        == LocalCacheState.CANDIDATE
    )

    write_trusted_manifest(cache_dir, cache_key, manifest)
    valid = inspect_local_cache(cache_dir, target, cache_key, manifest)
    assert valid.state == LocalCacheState.VALID
    assert valid.total_size == manifest.total_size

    (target / "weights" / "model.bin").write_bytes(b"different")
    assert (
        inspect_local_cache(cache_dir, target, cache_key, manifest).state
        == LocalCacheState.CONFLICT
    )


def test_directory_without_trusted_manifest_is_never_valid(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _manifest(target)

    result = inspect_local_cache(cache_dir, target, "cache-key")

    assert result.state == LocalCacheState.CANDIDATE
    assert result.manifest is None


def test_remote_manifest_validates_existing_directory_without_local_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    manifest = _manifest(target)

    result = inspect_local_cache(cache_dir, target, "cache-key", manifest)

    assert result.state == LocalCacheState.VALID
    assert not trusted_manifest_path(cache_dir, "cache-key").exists()


@pytest.mark.parametrize("payload", [b"{", b'{"identity": {}}'])
def test_missing_or_corrupt_trusted_manifest_is_not_valid(tmp_path, payload):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _manifest(target)
    path = trusted_manifest_path(cache_dir, "cache-key")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    result = inspect_local_cache(cache_dir, target, "cache-key")

    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_manifest_invalid"


def test_create_staging_dir_uses_cache_filesystem(tmp_path):
    cache_dir = tmp_path / "cache"

    staging = create_staging_dir(cache_dir, 12, 3)

    assert staging == cache_dir / ".preheat" / "12" / "3"
    assert os.stat(staging).st_dev == os.stat(cache_dir).st_dev


def test_create_staging_dir_reuses_partial_files_for_same_attempt(tmp_path):
    cache_dir = tmp_path / "cache"
    staging = create_staging_dir(cache_dir, 12, 3)
    partial = staging / "weights.part"
    partial.write_bytes(b"partial")

    reused = create_staging_dir(cache_dir, 12, 3)

    assert reused == staging
    assert partial.read_bytes() == b"partial"


def test_create_staging_dir_rejects_existing_regular_file(tmp_path):
    cache_dir = tmp_path / "cache"
    staging = cache_dir / ".preheat" / "12" / "3"
    staging.parent.mkdir(parents=True)
    staging.write_bytes(b"not-a-directory")

    with pytest.raises(LocalCacheError, match="local_cache_staging_conflict"):
        create_staging_dir(cache_dir, 12, 3)


def test_create_staging_dir_rejects_cross_device_directory(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    staging = create_staging_dir(cache_dir, 12, 3)
    partial = staging / "weights.part"
    partial.write_bytes(b"partial")
    original_stat = os.stat

    def cross_device(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if str(path).endswith("/.preheat/12/3"):
            return os.stat_result((result.st_mode, result.st_ino, 999, *result[3:]))
        return result

    monkeypatch.setattr(
        "gpustack.worker.model_preheat.local_cache.os.stat", cross_device
    )

    with pytest.raises(LocalCacheError, match="local_cache_staging_cross_device"):
        create_staging_dir(cache_dir, 12, 3)
    assert partial.read_bytes() == b"partial"


def test_publish_replaces_missing_target_and_writes_trusted_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 1)
    manifest = _manifest(staging)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.VALID
    assert result.published is True
    assert not staging.exists()
    assert (target / "weights" / "model.bin").read_bytes() == b"weights"
    assert trusted_manifest_path(cache_dir, "cache-key").exists()


def test_publish_matching_target_discards_staging_idempotently(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    manifest = _manifest(target)
    staging = create_staging_dir(cache_dir, 12, 2)
    _write_model(staging)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.VALID
    assert result.published is False
    assert not staging.exists()
    assert (target / "weights" / "model.bin").read_bytes() == b"weights"


def test_publish_conflict_never_overwrites_or_deletes_target(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    manifest = _manifest(target)
    (target / "weights" / "model.bin").write_bytes(b"in-use-content")
    staging = create_staging_dir(cache_dir, 12, 3)
    _write_model(staging)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.CONFLICT
    assert result.error_code == "local_cache_conflict"
    assert staging.exists()
    assert (target / "weights" / "model.bin").read_bytes() == b"in-use-content"


def test_publish_does_not_run_when_model_lock_is_unavailable(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 4)
    manifest = _manifest(staging)
    lock_path = model_lock_path(cache_dir, target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with SoftFileLock(str(lock_path)):
        result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_cache_lock_unavailable"
    assert staging.exists()
    assert not target.exists()


def test_write_trusted_manifest_is_idempotent_and_preserves_conflicts(tmp_path):
    cache_dir = tmp_path / "cache"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _manifest(first_root, revision="main")
    second = _manifest(second_root, revision="other")
    path = write_trusted_manifest(cache_dir, "cache-key", first)
    original = path.read_bytes()

    assert write_trusted_manifest(cache_dir, "cache-key", first) == path
    assert path.read_bytes() == original

    with pytest.raises(LocalCacheError, match="local_manifest_conflict"):
        write_trusted_manifest(cache_dir, "cache-key", second)
    assert path.read_bytes() == original


def test_write_trusted_manifest_rejects_corrupt_existing_file(tmp_path):
    cache_dir = tmp_path / "cache"
    manifest = _manifest(tmp_path / "model")
    path = trusted_manifest_path(cache_dir, "cache-key")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{")

    with pytest.raises(LocalCacheError, match="local_manifest_invalid"):
        write_trusted_manifest(cache_dir, "cache-key", manifest)
    assert path.read_bytes() == b"{"


def test_publish_missing_target_stops_before_manifest_conflict(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    existing = _manifest(tmp_path / "existing", revision="other")
    write_trusted_manifest(cache_dir, "cache-key", existing)
    staging = create_staging_dir(cache_dir, 12, 5)
    manifest = _manifest(staging)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.CONFLICT
    assert result.error_code == "local_manifest_conflict"
    assert not target.exists()
    assert staging.exists()


def test_publish_converts_target_replace_oserror_to_stable_result(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 6)
    manifest = _manifest(staging)
    original_replace = local_cache.os.replace

    def failing_replace(source, destination):
        if os.fspath(destination) == os.fspath(target):
            raise OSError("target replace failed")
        return original_replace(source, destination)

    monkeypatch.setattr(local_cache.os, "replace", failing_replace)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_cache_publish_failed"
    assert staging.exists()
    assert not target.exists()


def test_publish_recovers_when_manifest_write_fails_after_replace(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 7)
    manifest = _manifest(staging)
    original_write = local_cache.write_trusted_manifest

    def failing_write(*args, **kwargs):
        raise LocalCacheError("local_manifest_write_failed")

    monkeypatch.setattr(local_cache, "write_trusted_manifest", failing_write)
    failed = publish_staging(cache_dir, target, "cache-key", staging, manifest)
    monkeypatch.setattr(local_cache, "write_trusted_manifest", original_write)

    recovered = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert failed.state == LocalCacheState.ERROR
    assert failed.error_code == "local_manifest_write_failed"
    assert target.exists()
    assert not staging.exists()
    assert recovered.state == LocalCacheState.VALID
    assert recovered.published is False
    assert trusted_manifest_path(cache_dir, "cache-key").exists()
