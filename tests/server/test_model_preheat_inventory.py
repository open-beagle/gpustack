from gpustack.server.model_preheat_inventory import (
    LocalCacheProbeResult,
    RemoteRevisionFile,
    aggregate_local_cache_probes,
    manifest_from_remote_revision,
)
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity


def _identity():
    return ModelPreheatIdentity(
        source="modelscope",
        model_id="org/model",
        revision="commit-123",
        file_patterns=["config.json", "weights/*.bin"],
    )


def test_remote_revision_files_are_adapted_to_manifest_contract():
    manifest = manifest_from_remote_revision(
        _identity(),
        [
            RemoteRevisionFile(
                path="weights/model.bin",
                size=7,
                sha256="a" * 64,
            ),
            RemoteRevisionFile(path="config.json", size=3, sha256="b" * 64),
        ],
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="generation-id",
    )

    assert [file.path for file in manifest.files] == [
        "config.json",
        "weights/model.bin",
    ]
    assert manifest.identity == _identity()


def test_remote_revision_raw_paths_are_canonically_encoded():
    manifest = manifest_from_remote_revision(
        _identity(),
        [
            RemoteRevisionFile(
                path="weights/模型 文件.bin",
                size=7,
                sha256="a" * 64,
            ),
            RemoteRevisionFile(path="配置 文件.json", size=3, sha256="b" * 64),
        ],
        cache_key="cache-key",
        selection_digest="selection-digest",
        generation_id="generation-id",
    )

    assert [file.path for file in manifest.files] == [
        "%E9%85%8D%E7%BD%AE%20%E6%96%87%E4%BB%B6.json",
        "weights/%E6%A8%A1%E5%9E%8B%20%E6%96%87%E4%BB%B6.bin",
    ]


def test_local_probe_aggregation_is_worker_result_only():
    summary = aggregate_local_cache_probes(
        [
            LocalCacheProbeResult("worker-a", "valid", 10),
            LocalCacheProbeResult("worker-b", "candidate", 10),
            LocalCacheProbeResult("worker-c", "missing", 0),
        ]
    )

    assert summary.counts == {"candidate": 1, "missing": 1, "valid": 1}
    assert summary.all_valid is False
    assert summary.results[0].worker_uuid == "worker-a"
