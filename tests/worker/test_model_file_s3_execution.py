import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError, ReadTimeoutError

from gpustack.schemas.models import SourceEnum
from gpustack.worker import downloaders
from gpustack.worker.model_file_manager import _download_error_code
from gpustack.worker.model_preheat.executor import (
    TargetExecutionRequest,
    execute_target_preheat,
)
from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
)
from gpustack.worker.model_preheat.manifest import ModelPreheatManifest
from gpustack.worker.model_preheat.manifest import ManifestFile
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3ManifestError,
)


def _model(source=SourceEnum.HUGGING_FACE):
    return SimpleNamespace(
        source=source,
        huggingface_repo_id="org/model",
        huggingface_filename=None,
        model_scope_model_id="org/model",
        model_scope_file_path=None,
        ollama_library_model_name="qwen2.5:7b",
        mmproj_filename=None,
    )


def _execution(**updates):
    values = {
        "source": "huggingface",
        "model_id": "org/model",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "include_patterns": [],
        "exclude_patterns": [],
        "artifact_id": None,
        "manifest_path": None,
        "source_fallback_enabled": True,
        "profile": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_s3_exact_hit_does_not_call_public_source(tmp_path):
    execution = _execution(
        artifact_id="b" * 64,
        manifest_path="storage/huggingface/org/model/artifact/manifest.json",
        profile=SimpleNamespace(id=3),
    )
    with (
        patch.object(
            downloaders, "_download_execution_artifact", return_value=["/model"]
        ) as s3,
        patch.object(downloaders.HfDownloader, "download") as hub,
    ):
        result = downloaders.download_model(
            _model(), cache_dir=str(tmp_path), execution=execution
        )
    assert result == ["/model"]
    s3.assert_called_once()
    hub.assert_not_called()


def test_confirmed_miss_with_fallback_disabled_never_calls_hub(tmp_path):
    execution = _execution(source_fallback_enabled=False)
    with patch.object(downloaders.HfDownloader, "download") as hub:
        with pytest.raises(ValueError, match="model_artifact_not_found"):
            downloaders.download_model(
                _model(), cache_dir=str(tmp_path), execution=execution
            )
    hub.assert_not_called()


def test_s3_failure_never_silently_falls_back(tmp_path):
    execution = _execution(
        artifact_id="b" * 64,
        manifest_path="storage/huggingface/org/model/artifact/manifest.json",
        profile=SimpleNamespace(id=3),
    )
    with (
        patch.object(
            downloaders,
            "_download_execution_artifact",
            side_effect=ModelPreheatS3ManifestError("s3_manifest_invalid"),
        ),
        patch.object(downloaders.HfDownloader, "download") as hub,
    ):
        with pytest.raises(ModelPreheatS3ManifestError):
            downloaders.download_model(
                _model(), cache_dir=str(tmp_path), execution=execution
            )
    hub.assert_not_called()


def test_s3_manifest_accepts_task3_concrete_selection_for_raw_request(tmp_path):
    manifest = ModelPreheatManifest(
        identity=ModelPreheatIdentity(
            source="huggingface",
            model_id="org/model",
            revision="a" * 40,
            file_patterns=["model.gguf", "model.gguf/**"],
        ),
        files=(),
    )
    execution = _execution(
        include_patterns=["model.gguf"],
        artifact_id=manifest.artifact_id,
        manifest_path="storage/huggingface/org/model/artifact/manifest.json",
        profile=SimpleNamespace(
            endpoint="https://s3.example.com",
            access_key="access",
            secret_key="secret",
            tls_enabled=True,
            tls_verify=True,
            region="",
            use_virtual_hosted_style=True,
            bucket="models",
            prefix="storage",
        ),
    )
    client = SimpleNamespace(
        read_artifact_manifest_path=lambda bucket, path: manifest,
        artifact_manifest_object=lambda prefix, value: execution.manifest_path,
    )
    with patch.object(
        downloaders.ModelPreheatS3Client, "from_minio", return_value=client
    ):
        assert (
            downloaders._download_execution_artifact(
                execution, str(tmp_path / "model"), str(tmp_path)
            )
            == []
        )


def test_confirmed_miss_falls_back_to_resolved_revision_without_upload(tmp_path):
    execution = _execution()
    with (
        patch.object(
            downloaders.HfDownloader, "download", return_value=["/model"]
        ) as hub,
        patch.object(downloaders.ModelPreheatS3Client, "from_minio") as s3,
    ):
        result = downloaders.download_model(
            _model(),
            cache_dir=str(tmp_path),
            huggingface_token="token",
            execution=execution,
        )
    assert result == ["/model"]
    assert hub.call_args.kwargs["revision"] == "a" * 40
    s3.assert_not_called()


def test_ollama_s3_hit_uses_artifact_instead_of_registry(tmp_path):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        requested_revision=None,
        resolved_revision="local-snapshot-" + "a" * 64,
        artifact_id="b" * 64,
        manifest_path=(
            "storage/ollama_library/qwen2.5:7b/" + "b" * 64 + "/manifest.json"
        ),
        profile=SimpleNamespace(id=3),
    )
    with (
        patch.object(
            downloaders, "_download_execution_artifact", return_value=["/ollama/model"]
        ) as s3,
        patch.object(downloaders.OllamaLibraryDownloader, "download") as registry,
    ):
        result = downloaders.download_model(
            _model(SourceEnum.OLLAMA_LIBRARY),
            cache_dir=str(tmp_path),
            execution=execution,
        )

    assert result == ["/ollama/model"]
    s3.assert_called_once()
    registry.assert_not_called()


def test_ollama_s3_miss_falls_back_to_registry_when_enabled(tmp_path):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        requested_revision=None,
        resolved_revision="local-snapshot-" + "a" * 64,
    )
    with patch.object(
        downloaders.OllamaLibraryDownloader,
        "download",
        return_value=["/ollama/model"],
    ) as registry:
        result = downloaders.download_model(
            _model(SourceEnum.OLLAMA_LIBRARY),
            cache_dir=str(tmp_path),
            ollama_library_base_url="https://ollama.example.com",
            execution=execution,
        )

    assert result == ["/ollama/model"]
    assert registry.call_args.kwargs["model_name"] == "qwen2.5:7b"


@pytest.mark.parametrize("fallback_enabled", [True, False])
def test_ollama_s3_hit_failure_honors_frozen_fallback(tmp_path, fallback_enabled):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        artifact_id="b" * 64,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        source_fallback_enabled=fallback_enabled,
        profile=SimpleNamespace(id=3),
    )
    with (
        patch.object(
            downloaders,
            "_download_execution_artifact",
            side_effect=ModelPreheatS3ManifestError("s3_manifest_invalid"),
        ),
        patch.object(
            downloaders.OllamaLibraryDownloader,
            "download",
            return_value=["/ollama/model"],
        ) as registry,
    ):
        if fallback_enabled:
            assert downloaders.download_model(
                _model(SourceEnum.OLLAMA_LIBRARY),
                cache_dir=str(tmp_path),
                execution=execution,
            ) == ["/ollama/model"]
            registry.assert_called_once()
        else:
            with pytest.raises(ModelPreheatS3ManifestError):
                downloaders.download_model(
                    _model(SourceEnum.OLLAMA_LIBRARY),
                    cache_dir=str(tmp_path),
                    execution=execution,
                )
            registry.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [ModelPreheatCanceled("canceled"), ModelPreheatIdentityError("invalid_source")],
)
def test_ollama_s3_hit_does_not_swallow_cancel_or_identity_errors(tmp_path, error):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        artifact_id="b" * 64,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        source_fallback_enabled=True,
        profile=SimpleNamespace(id=3),
    )
    with (
        patch.object(downloaders, "_download_execution_artifact", side_effect=error),
        patch.object(downloaders.OllamaLibraryDownloader, "download") as registry,
    ):
        with pytest.raises(type(error)):
            downloaders.download_model(
                _model(SourceEnum.OLLAMA_LIBRARY),
                cache_dir=str(tmp_path),
                execution=execution,
            )
    registry.assert_not_called()


def test_ollama_s3_hit_does_not_fallback_on_credential_error(tmp_path):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        artifact_id="b" * 64,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        source_fallback_enabled=True,
        profile=SimpleNamespace(id=3),
    )
    credential_error = S3Error(None, "AccessDenied", "denied", None, None, None)
    with (
        patch.object(
            downloaders,
            "_download_execution_artifact",
            side_effect=credential_error,
        ),
        patch.object(downloaders.OllamaLibraryDownloader, "download") as registry,
    ):
        with pytest.raises(S3Error):
            downloaders.download_model(
                _model(SourceEnum.OLLAMA_LIBRARY),
                cache_dir=str(tmp_path),
                execution=execution,
            )
    registry.assert_not_called()


def test_ollama_s3_hit_does_not_fallback_on_invalid_security(tmp_path):
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        artifact_id="b" * 64,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        source_fallback_enabled=True,
        profile=SimpleNamespace(id=3),
    )
    credential_error = S3Error(
        None, "InvalidSecurity", "bad security token", None, None, None
    )
    with (
        patch.object(
            downloaders,
            "_download_execution_artifact",
            side_effect=credential_error,
        ),
        patch.object(downloaders.OllamaLibraryDownloader, "download") as registry,
    ):
        with pytest.raises(S3Error):
            downloaders.download_model(
                _model(SourceEnum.OLLAMA_LIBRARY),
                cache_dir=str(tmp_path),
                execution=execution,
            )
    registry.assert_not_called()


def test_ollama_artifact_downloads_to_native_cache_file(tmp_path):
    content = b"ollama-model"
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="qwen2.5:7b",
        revision="local-snapshot-" + "a" * 64,
        file_patterns=["qwen2_5_7b", "qwen2_5_7b/**"],
    )
    manifest = ModelPreheatManifest(
        identity=identity,
        files=(
            ManifestFile(
                path="qwen2_5_7b",
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
    )
    manifest_path = (
        f"storage/ollama_library/qwen2.5:7b/{manifest.artifact_id}/manifest.json"
    )
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        requested_revision=None,
        resolved_revision=identity.revision,
        include_patterns=["qwen2_5_7b", "qwen2_5_7b/**"],
        artifact_id=manifest.artifact_id,
        manifest_path=manifest_path,
        profile=SimpleNamespace(
            endpoint="https://s3.example.com",
            access_key="access",
            secret_key="secret",
            tls_enabled=True,
            tls_verify=True,
            region="",
            use_virtual_hosted_style=True,
            bucket="models",
            prefix="storage",
        ),
    )

    def download_file(bucket, prefix, value, manifest_file, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    client = SimpleNamespace(
        read_artifact_manifest_path=lambda bucket, path: manifest,
        artifact_manifest_object=lambda prefix, value: manifest_path,
        download_artifact_file=download_file,
    )
    with patch.object(
        downloaders.ModelPreheatS3Client, "from_minio", return_value=client
    ):
        result = downloaders._download_execution_artifact(
            execution, None, str(tmp_path)
        )

    target = tmp_path / "ollama" / "qwen2_5_7b"
    assert result == [str(target)]
    assert target.read_bytes() == content


@pytest.mark.parametrize(
    "paths",
    [
        ["wrong_name"],
        ["qwen2_5_7b", "extra_file"],
    ],
)
def test_ollama_artifact_rejects_noncanonical_manifest(tmp_path, paths):
    content = b"ollama-model"
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="qwen2.5:7b",
        revision="local-snapshot-" + "a" * 64,
        file_patterns=paths,
    )
    manifest = ModelPreheatManifest(
        identity=identity,
        files=tuple(
            ManifestFile(
                path=path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for path in paths
        ),
    )
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        resolved_revision=identity.revision,
        artifact_id=manifest.artifact_id,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        profile=SimpleNamespace(
            endpoint="https://s3.example.com",
            access_key="access",
            secret_key="secret",
            tls_enabled=True,
            tls_verify=True,
            region="",
            use_virtual_hosted_style=True,
            bucket="models",
            prefix="storage",
        ),
    )
    client = SimpleNamespace(
        read_artifact_manifest_path=lambda bucket, path: manifest,
        artifact_manifest_object=lambda prefix, value: execution.manifest_path,
    )

    with patch.object(
        downloaders.ModelPreheatS3Client, "from_minio", return_value=client
    ):
        with pytest.raises(ModelPreheatS3ManifestError):
            downloaders._download_execution_artifact(execution, None, str(tmp_path))


def test_ollama_artifact_failure_preserves_existing_model_and_cleans_staging(
    tmp_path,
):
    old_content = b"old-model"
    target = tmp_path / "ollama" / "qwen2_5_7b"
    target.parent.mkdir(parents=True)
    target.write_bytes(old_content)
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="qwen2.5:7b",
        revision="local-snapshot-" + "a" * 64,
        file_patterns=["qwen2_5_7b", "qwen2_5_7b/**"],
    )
    manifest = ModelPreheatManifest(
        identity=identity,
        files=(
            ManifestFile(
                path="qwen2_5_7b",
                size=9,
                sha256=hashlib.sha256(b"new-model").hexdigest(),
            ),
        ),
    )
    execution = _execution(
        source="ollama_library",
        model_id="qwen2.5:7b",
        resolved_revision=identity.revision,
        artifact_id=manifest.artifact_id,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
        profile=SimpleNamespace(
            endpoint="https://s3.example.com",
            access_key="access",
            secret_key="secret",
            tls_enabled=True,
            tls_verify=True,
            region="",
            use_virtual_hosted_style=True,
            bucket="models",
            prefix="storage",
        ),
    )

    def fail_midway(bucket, prefix, value, manifest_file, staging_target):
        staging_target.parent.mkdir(parents=True, exist_ok=True)
        staging_target.write_bytes(b"partial")
        raise OSError("network interrupted")

    client = SimpleNamespace(
        read_artifact_manifest_path=lambda bucket, path: manifest,
        artifact_manifest_object=lambda prefix, value: execution.manifest_path,
        download_artifact_file=fail_midway,
    )
    with patch.object(
        downloaders.ModelPreheatS3Client, "from_minio", return_value=client
    ):
        with pytest.raises(OSError):
            downloaders._download_execution_artifact(execution, None, str(tmp_path))

    assert target.read_bytes() == old_content
    assert not list(target.parent.glob(".qwen2_5_7b.staging-*"))


def test_ollama_distribution_target_uses_native_cache_root(tmp_path):
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="qwen2.5:7b",
        revision="local-snapshot-" + "a" * 64,
        file_patterns=["qwen2_5_7b", "qwen2_5_7b/**"],
    )

    assert downloaders.preheat_model_target_dir(tmp_path, identity) == (
        tmp_path / "ollama"
    )


def _ollama_distribution_request(tmp_path, manifest):
    return TargetExecutionRequest(
        cache_dir=tmp_path / "cache",
        target_dir=tmp_path / "cache" / "ollama",
        task_id=7,
        attempt=1,
        request_digest=manifest.identity.request_digest,
        identity=manifest.identity,
        exclude_patterns=[],
        bucket="models",
        prefix="storage",
        artifact_id=manifest.artifact_id,
        manifest_path="storage/ollama_library/qwen2.5:7b/manifest.json",
    )


def _ollama_distribution_manifest(paths):
    content = b"ollama-model"
    identity = ModelPreheatIdentity(
        source="ollama_library",
        model_id="qwen2.5:7b",
        revision="local-snapshot-" + "a" * 64,
        file_patterns=paths,
    )
    return ModelPreheatManifest(
        identity=identity,
        files=tuple(
            ManifestFile(
                path=path,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for path in paths
        ),
    )


def _ollama_distribution_client(request, manifest, download):
    return SimpleNamespace(
        read_artifact_manifest_path=lambda bucket, path: manifest,
        artifact_manifest_object=lambda prefix, value: request.manifest_path,
        download_artifact_file=download,
    )


def test_ollama_distribution_installs_single_file_atomically(tmp_path):
    content = b"ollama-model"
    manifest = _ollama_distribution_manifest(["qwen2_5_7b"])
    request = _ollama_distribution_request(tmp_path, manifest)
    progress = []

    def download(bucket, prefix, value, manifest_file, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(request, manifest, download),
        progress_callback=lambda completed, downloaded_size, total_size: progress.append(
            (completed, downloaded_size, total_size)
        ),
    )

    assert result["state"] == "ready"
    assert (request.target_dir / "qwen2_5_7b").read_bytes() == content
    assert progress == [(["qwen2_5_7b"], len(content), len(content))]


def test_ollama_distribution_skips_matching_file_without_download(tmp_path):
    content = b"ollama-model"
    manifest = _ollama_distribution_manifest(["qwen2_5_7b"])
    request = _ollama_distribution_request(tmp_path, manifest)
    target = request.target_dir / "qwen2_5_7b"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    downloads = []

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(
            request,
            manifest,
            lambda *args: downloads.append(args),
        ),
    )

    assert result["state"] == "ready"
    assert result["downloaded"] == 0
    assert result["skipped"] == 1
    assert downloads == []
    assert target.read_bytes() == content


def test_ollama_distribution_matching_file_hash_honors_cancellation(tmp_path):
    content = b"ollama-model"
    manifest = _ollama_distribution_manifest(["qwen2_5_7b"])
    request = _ollama_distribution_request(tmp_path, manifest)
    target = request.target_dir / "qwen2_5_7b"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    checks = iter((False, True))
    downloads = []

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(
            request,
            manifest,
            lambda *args: downloads.append(args),
        ),
        cancel_check=lambda: next(checks),
    )

    assert result == {"state": "error", "error_code": "canceled"}
    assert downloads == []
    assert target.read_bytes() == content


@pytest.mark.parametrize(
    "paths",
    [["wrong_name"], ["qwen2_5_7b", "extra_file"]],
)
def test_ollama_distribution_rejects_noncanonical_manifest(tmp_path, paths):
    manifest = _ollama_distribution_manifest(paths)
    request = _ollama_distribution_request(tmp_path, manifest)
    downloads = []

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(
            request,
            manifest,
            lambda *args: downloads.append(args),
        ),
    )

    assert result == {"state": "error", "error_code": "s3_manifest_invalid"}
    assert downloads == []


def test_ollama_distribution_failure_preserves_target_and_cleans_staging(tmp_path):
    manifest = _ollama_distribution_manifest(["qwen2_5_7b"])
    request = _ollama_distribution_request(tmp_path, manifest)
    target = request.target_dir / "qwen2_5_7b"
    sibling = request.target_dir / "keep.txt"
    request.target_dir.mkdir(parents=True)
    target.write_bytes(b"old-model")
    sibling.write_bytes(b"sibling")

    def fail(bucket, prefix, value, manifest_file, staging_target):
        staging_target.parent.mkdir(parents=True, exist_ok=True)
        staging_target.write_bytes(b"partial")
        raise OSError("network interrupted")

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(request, manifest, fail),
    )

    assert result == {"state": "error", "error_code": "worker_execution_failed"}
    assert target.read_bytes() == b"old-model"
    assert sibling.read_bytes() == b"sibling"
    assert not list(request.target_dir.glob(".qwen2_5_7b.staging-*"))


def test_ollama_distribution_progress_failure_preserves_target_and_cleans_staging(
    tmp_path,
):
    content = b"ollama-model"
    manifest = _ollama_distribution_manifest(["qwen2_5_7b"])
    request = _ollama_distribution_request(tmp_path, manifest)
    target = request.target_dir / "qwen2_5_7b"
    sibling = request.target_dir / "keep.txt"
    request.target_dir.mkdir(parents=True)
    target.write_bytes(b"old-model")
    sibling.write_bytes(b"sibling")

    def download(bucket, prefix, value, manifest_file, staging_target):
        staging_target.parent.mkdir(parents=True, exist_ok=True)
        staging_target.write_bytes(content)

    def fail_progress(completed, downloaded_size, total_size):
        raise OSError("progress rejected")

    result = execute_target_preheat(
        request,
        _ollama_distribution_client(request, manifest, download),
        progress_callback=fail_progress,
    )

    assert result == {"state": "error", "error_code": "worker_execution_failed"}
    assert target.read_bytes() == b"old-model"
    assert sibling.read_bytes() == b"sibling"
    assert not list(request.target_dir.glob(".qwen2_5_7b.staging-*"))


def test_modelscope_filelist_revision_falls_back_with_requested_revision(tmp_path):
    files = [
        type(
            "File",
            (),
            {"path": "model.bin", "size": 10, "blob_id": "a" * 64},
        )()
    ]
    from gpustack.server.model_preheat_revision import modelscope_filelist_revision

    execution = _execution(
        source="modelscope",
        requested_revision="release",
        resolved_revision=modelscope_filelist_revision(files),
    )
    with (
        patch.object(
            downloaders.ModelScopeDownloader, "download", return_value=["/model"]
        ) as hub,
        patch.object(downloaders, "ModelScopeHubApi") as hub_api,
    ):
        hub_api.return_value.list_repo_files.return_value = files
        result = downloaders.download_model(
            _model(SourceEnum.MODEL_SCOPE),
            cache_dir=str(tmp_path),
            execution=execution,
        )

    assert result == ["/model"]
    assert hub.call_args.kwargs["revision"] == "release"


def test_s3_authentication_error_has_stable_code():
    error = S3Error(None, "AccessDenied", "denied", None, None, None)
    assert _download_error_code(error) == "s3_authentication_failed"


def test_s3_timeout_has_stable_code():
    timeout = ReadTimeoutError(None, "/models", "timed out")
    error = MaxRetryError(None, "/models", reason=timeout)
    assert _download_error_code(error) == "network_timeout"
