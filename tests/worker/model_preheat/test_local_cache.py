import os
from pathlib import Path

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
    return build_model_preheat_manifest(
        root,
        _identity(revision),
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id=f"generation-{revision}",
    )


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


def test_publish_missing_target_cancel_after_rename_restores_staging(
    tmp_path, monkeypatch
):
    class PublishCanceled(RuntimeError):
        pass

    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 36)
    manifest = _manifest(staging)
    original_replace = local_cache.os.replace
    renamed = False

    def track_publish(source, destination):
        nonlocal renamed
        result = original_replace(source, destination)
        if source == staging and destination == target:
            renamed = True
        return result

    def cancel_after_publish():
        if renamed:
            raise PublishCanceled()

    monkeypatch.setattr(local_cache.os, "replace", track_publish)

    with pytest.raises(PublishCanceled):
        publish_staging(
            cache_dir,
            target,
            "cache-key",
            staging,
            manifest,
            cancel_callback=cancel_after_publish,
        )

    assert not target.exists()
    assert (staging / "weights" / "model.bin").read_bytes() == b"weights"
    assert not trusted_manifest_path(cache_dir, "cache-key").exists()


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


def test_publish_matching_target_checks_cancel_before_staging_cleanup(
    tmp_path, monkeypatch
):
    class CleanupCanceled(RuntimeError):
        pass

    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    manifest = _manifest(target)
    staging = create_staging_dir(cache_dir, 12, 37)
    _write_model(staging)
    original_inspect = local_cache.inspect_local_cache
    inspected = False

    def track_inspection(*args, **kwargs):
        nonlocal inspected
        result = original_inspect(*args, **kwargs)
        inspected = True
        return result

    def cancel_after_inspection():
        if inspected:
            raise CleanupCanceled()

    monkeypatch.setattr(local_cache, "inspect_local_cache", track_inspection)

    with pytest.raises(CleanupCanceled):
        publish_staging(
            cache_dir,
            target,
            "cache-key",
            staging,
            manifest,
            cancel_callback=cancel_after_inspection,
        )

    assert staging.exists()
    assert (target / "weights" / "model.bin").read_bytes() == b"weights"
    assert not trusted_manifest_path(cache_dir, "cache-key").exists()


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


@pytest.mark.parametrize("unsafe_root", ["symlink", "fifo"])
def test_publish_does_not_replace_unsafe_target_root(tmp_path, unsafe_root):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    target.parent.mkdir(parents=True)
    if unsafe_root == "symlink":
        original = cache_dir / "original"
        _write_model(original, content=b"stale")
        target.symlink_to(original, target_is_directory=True)
    else:
        os.mkfifo(target)

    staging = create_staging_dir(cache_dir, 12, 28)
    manifest = _manifest(staging)

    inspection = inspect_local_cache(cache_dir, target, "cache-key", manifest)
    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert inspection.state == LocalCacheState.ERROR
    assert result.state == LocalCacheState.ERROR
    assert result.published is False
    assert os.path.lexists(target)
    assert staging.exists()
    if unsafe_root == "symlink":
        assert (
            cache_dir / "original" / "weights" / "model.bin"
        ).read_bytes() == b"stale"


@pytest.mark.parametrize("unsafe_root", ["symlink", "fifo"])
def test_publish_does_not_publish_unsafe_staging_root(tmp_path, unsafe_root):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    source = create_staging_dir(cache_dir, 12, 281)
    manifest = _manifest(source)
    staging = cache_dir / ".preheat" / "12" / "282"
    if unsafe_root == "symlink":
        staging.symlink_to(source, target_is_directory=True)
    else:
        os.mkfifo(staging)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert result.state == LocalCacheState.ERROR
    assert result.published is False
    assert not target.exists()
    assert os.path.lexists(staging)


def test_publish_rejects_selected_file_swapped_between_scan_and_hash(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    staging = create_staging_dir(cache_dir, 12, 29)
    manifest = _manifest(staging)
    selected = staging / "weights" / "model.bin"
    replacement = staging / "replacement.bin"
    replacement.write_bytes(b"changed")
    original_sha256 = local_cache._sha256_file
    switched = False

    def switch_file_before_hash(path, expected_stat, **kwargs):
        nonlocal switched
        if path == selected and not switched:
            switched = True
            replacement.replace(selected)
        return original_sha256(path, expected_stat, **kwargs)

    monkeypatch.setattr(local_cache, "_sha256_file", switch_file_before_hash)

    result = publish_staging(cache_dir, target, "cache-key", staging, manifest)

    assert switched is True
    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_cache_staging_invalid"
    assert not target.exists()
    assert staging.exists()


def test_manifest_path_rejects_lexical_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(LocalCacheError, match="local_cache_path_escape"):
        local_cache._manifest_file_path(
            root, type("ManifestPath", (), {"path": "../outside"})()
        )


def test_publish_replacement_preserves_safe_unselected_files(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _write_model(target, content=b"stale")
    (target / "tokenizer.json").write_bytes(b"tokenizer")
    staging = create_staging_dir(cache_dir, 12, 30)
    manifest = _manifest(staging)

    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert result.state == LocalCacheState.VALID
    assert (target / "weights" / "model.bin").read_bytes() == b"weights"
    assert (target / "tokenizer.json").read_bytes() == b"tokenizer"
    assert (
        inspect_local_cache(cache_dir, target, "cache-key", manifest).state == "valid"
    )


def test_publish_replacement_rejects_target_changes_after_extra_merge(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _write_model(target, content=b"stale")
    (target / "early.extra").write_bytes(b"early")
    staging = create_staging_dir(cache_dir, 12, 301)
    manifest = _manifest(staging)
    original_verify = local_cache._verify_directory
    injected = False

    def inject_late_extra(*args, **kwargs):
        nonlocal injected
        result = original_verify(*args, **kwargs)
        if Path(args[0]) == staging and kwargs.get("allow_extra") and not injected:
            (target / "late.extra").write_bytes(b"late")
            injected = True
        return result

    monkeypatch.setattr(local_cache, "_verify_directory", inject_late_extra)

    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert injected is True
    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_cache_extra_source_changed"
    assert (target / "weights" / "model.bin").read_bytes() == b"stale"
    assert (target / "early.extra").read_bytes() == b"early"
    assert (target / "late.extra").read_bytes() == b"late"
    assert (staging / "weights" / "model.bin").read_bytes() == b"weights"
    assert not (staging / "early.extra").exists()
    assert not list(target.parent.glob(".model.preheat-backup-*"))


def test_publish_replacement_repairs_missing_selected_files(tmp_path):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    target.mkdir(parents=True)
    (target / "tokenizer.json").write_bytes(b"tokenizer")
    staging = create_staging_dir(cache_dir, 12, 33)
    manifest = _manifest(staging)

    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert result.state == LocalCacheState.VALID
    assert result.published is True
    assert (target / "weights" / "model.bin").read_bytes() == b"weights"
    assert (target / "tokenizer.json").read_bytes() == b"tokenizer"


def test_publish_replacement_rolls_back_target_and_merge_on_manifest_failure(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _write_model(target, content=b"stale")
    (target / "tokenizer.json").write_bytes(b"tokenizer")
    staging = create_staging_dir(cache_dir, 12, 31)
    manifest = _manifest(staging)

    def fail_manifest(*args, **kwargs):
        raise LocalCacheError("local_manifest_write_failed")

    monkeypatch.setattr(local_cache, "_overwrite_trusted_manifest", fail_manifest)
    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert result.state == LocalCacheState.ERROR
    assert result.error_code == "local_manifest_write_failed"
    assert (target / "weights" / "model.bin").read_bytes() == b"stale"
    assert (target / "tokenizer.json").read_bytes() == b"tokenizer"
    assert (staging / "weights" / "model.bin").read_bytes() == b"weights"
    assert not (staging / "tokenizer.json").exists()
    assert not list(target.parent.glob(".model.preheat-backup-*"))


@pytest.mark.parametrize(
    ("failure_phase", "error_code"),
    [
        ("merge", "local_cache_extra_source_changed"),
        ("verify", "local_cache_staging_invalid"),
        ("staging_rename", "local_cache_publish_failed"),
    ],
)
def test_publish_replacement_rolls_back_before_manifest(
    tmp_path, monkeypatch, failure_phase, error_code
):
    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _write_model(target, content=b"stale")
    (target / "tokenizer.json").write_bytes(b"tokenizer")
    staging = create_staging_dir(cache_dir, 12, f"rollback-{failure_phase}")
    manifest = _manifest(staging)

    if failure_phase == "merge":
        monkeypatch.setattr(
            local_cache,
            "_merge_unselected_files",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                LocalCacheError("local_cache_extra_source_changed")
            ),
        )
    elif failure_phase == "verify":
        original_verify = local_cache._verify_directory

        def fail_final_verification(root, checked_manifest, **kwargs):
            if Path(root) == staging and kwargs.get("allow_extra"):
                return local_cache.LocalCacheInspection(LocalCacheState.ERROR)
            return original_verify(root, checked_manifest, **kwargs)

        monkeypatch.setattr(local_cache, "_verify_directory", fail_final_verification)
    else:
        original_replace = local_cache.os.replace

        def fail_staging_rename(source, destination):
            if source == staging and destination == target:
                raise OSError("injected staging rename failure")
            return original_replace(source, destination)

        monkeypatch.setattr(local_cache.os, "replace", fail_staging_rename)

    result = publish_staging(
        cache_dir,
        target,
        "cache-key",
        staging,
        manifest,
        replace_conflicting=True,
    )

    assert result.state == LocalCacheState.ERROR
    assert result.error_code == error_code
    assert (target / "weights" / "model.bin").read_bytes() == b"stale"
    assert (target / "tokenizer.json").read_bytes() == b"tokenizer"
    assert (staging / "weights" / "model.bin").read_bytes() == b"weights"
    assert not (staging / "tokenizer.json").exists()
    assert not list(target.parent.glob(".model.preheat-backup-*"))


def test_publish_replacement_cancel_cleans_merge_and_preserves_target(tmp_path):
    class MergeCanceled(RuntimeError):
        pass

    cache_dir = tmp_path / "cache"
    target = cache_dir / "org" / "model"
    _write_model(target, content=b"stale")
    (target / "large.extra").write_bytes(b"x" * (3 * 1024 * 1024))
    staging = create_staging_dir(cache_dir, 12, 32)
    manifest = _manifest(staging)
    checks = 0

    def cancel_during_merge():
        nonlocal checks
        checks += 1
        if checks >= 8:
            raise MergeCanceled()

    with pytest.raises(MergeCanceled):
        publish_staging(
            cache_dir,
            target,
            "cache-key",
            staging,
            manifest,
            replace_conflicting=True,
            cancel_callback=cancel_during_merge,
        )

    assert (target / "weights" / "model.bin").read_bytes() == b"stale"
    assert (target / "large.extra").exists()
    assert (staging / "weights" / "model.bin").read_bytes() == b"weights"
    assert not (staging / "large.extra").exists()


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
