from unittest.mock import patch

import pytest

from gpustack.config.config import Config
from gpustack.worker.downloader_s3 import S3Downloader
from gpustack.worker.downloaders import ModelScopeDownloader


def test_legacy_worker_s3_config_maps_to_center_s3(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        worker_s3_host="minio.example.com",
        worker_s3_access_key="access",
        worker_s3_secret_key="secret",
        worker_s3_ssl=True,
        worker_s3_use_virtual_hosted_style=True,
        worker_s3_region="cn-test",
    )

    assert cfg.worker_center_s3_host == "minio.example.com"
    assert cfg.worker_center_s3_access_key == "access"
    assert cfg.worker_center_s3_secret_key == "secret"
    assert cfg.worker_center_s3_ssl is True
    assert cfg.worker_center_s3_use_virtual_hosted_style is True
    assert cfg.worker_center_s3_region == "cn-test"


def test_local_s3_host_enables_default_modelscope_prefix(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        worker_local_s3_host="local-minio.example.com",
        worker_local_s3_access_key="access",
        worker_local_s3_secret_key="secret",
    )

    assert cfg.worker_local_s3_modelscope_prefix == "s3://bd-wind/modelscope"


def test_center_s3_legacy_uri_maps_to_beagle_cache_path():
    s3_path = S3Downloader.normalize_s3_path(
        "s3://beagle_wind/bd-wind/datamodel/model-id/v1/model.gguf"
    )
    bucket_name, _ = S3Downloader.parse_s3_path(s3_path)

    assert bucket_name == "bd-wind"
    assert (
        S3Downloader._cache_relative_path(s3_path, bucket_name)
        == "model-id/v1/model.gguf"
    )


def test_s3_download_object_reuses_local_file_when_size_matches(tmp_path):
    local_file = tmp_path / "model.gguf"
    local_file.write_bytes(b"1234")
    downloader = S3Downloader(
        "minio.example.com",
        access_key="access",
        secret_key="secret",
    )

    downloaded = downloader._download_object(
        "bucket",
        "datamodel/model.gguf",
        str(local_file),
        total_size=4,
    )

    assert downloaded is False
    assert local_file.read_bytes() == b"1234"


def test_modelscope_local_s3_cache_miss_raises_without_fallback(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        worker_local_s3_host="local-minio.example.com",
        worker_local_s3_access_key="access",
        worker_local_s3_secret_key="secret",
        worker_local_s3_modelscope_prefix="s3://model-cache/modelscope",
        worker_local_s3_modelscope_fallback=False,
    )

    with (
        patch.object(ModelScopeDownloader, "check_s3_model_exists", return_value=False),
        patch.object(ModelScopeDownloader, "_snapshot_download_with_retry") as snapshot,
    ):
        with pytest.raises(ValueError, match="local S3 cache miss"):
            ModelScopeDownloader.download(
                model_id="Qwen/Qwen3.5-35B-A3B-FP8",
                file_path=None,
                extra_file_path=None,
                cache_dir=str(tmp_path / "cache" / "model_scope"),
                cfg=cfg,
            )

    snapshot.assert_not_called()
