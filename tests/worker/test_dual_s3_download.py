from unittest.mock import Mock, patch

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

    assert cfg.worker_local_s3_modelscope_prefix == "s3://bd-wind/datamodel"
    assert cfg.worker_local_s3_modelscope_fallback is True


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


def test_modelscope_file_info_local_s3_cache_miss_falls_back_by_default(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        worker_local_s3_host="local-minio.example.com",
        worker_local_s3_access_key="access",
        worker_local_s3_secret_key="secret",
        worker_local_s3_modelscope_prefix="s3://model-cache/modelscope",
    )
    downloader = Mock()
    downloader.list_file_entries.return_value = []
    api = Mock()
    api.get_model_files.return_value = [{"Path": "config.json", "Size": 123}]
    model = Mock(model_scope_model_id="Qwen/Qwen-7B-Chat-Int8")

    with (
        patch("gpustack.worker.downloaders.get_s3_downloader", return_value=downloader),
        patch("gpustack.worker.downloaders.HubApi", return_value=api),
    ):
        file_list = ModelScopeDownloader.get_model_file_info(model, cfg=cfg)

    assert len(file_list) == 1
    assert file_list[0].rfilename == "config.json"
    assert file_list[0].size == 123
    api.get_model_files.assert_called_once_with(
        "Qwen/Qwen-7B-Chat-Int8", recursive=True
    )


def test_modelscope_file_info_local_s3_cache_miss_raises_without_fallback(tmp_path):
    cfg = Config(
        data_dir=str(tmp_path),
        worker_local_s3_host="local-minio.example.com",
        worker_local_s3_access_key="access",
        worker_local_s3_secret_key="secret",
        worker_local_s3_modelscope_prefix="s3://model-cache/modelscope",
        worker_local_s3_modelscope_fallback=False,
    )
    downloader = Mock()
    downloader.list_file_entries.return_value = []
    model = Mock(model_scope_model_id="Qwen/Qwen-7B-Chat-Int8")

    with patch(
        "gpustack.worker.downloaders.get_s3_downloader", return_value=downloader
    ):
        with pytest.raises(ValueError, match="local S3 cache not found"):
            ModelScopeDownloader.get_model_file_info(model, cfg=cfg)
