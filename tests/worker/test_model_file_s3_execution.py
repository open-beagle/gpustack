from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpustack.schemas.models import SourceEnum
from gpustack.worker import downloaders
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3ManifestError


def _model(source=SourceEnum.HUGGING_FACE):
    return SimpleNamespace(
        source=source,
        huggingface_repo_id="org/model",
        huggingface_filename=None,
        model_scope_model_id="org/model",
        model_scope_file_path=None,
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
