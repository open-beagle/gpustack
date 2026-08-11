from dataclasses import dataclass

from gpustack.worker.model_preheat.identity import ModelPreheatIdentity, encode_path
from gpustack.worker.model_preheat.manifest import ManifestFile, ModelPreheatManifest


@dataclass(frozen=True)
class RemoteRevisionFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class LocalCacheProbeResult:
    worker_uuid: str
    state: str
    total_size: int = 0
    error_code: str | None = None


@dataclass(frozen=True)
class LocalCacheProbeSummary:
    results: tuple[LocalCacheProbeResult, ...]
    counts: dict[str, int]
    all_valid: bool


def manifest_from_remote_revision(
    identity: ModelPreheatIdentity, files: list[RemoteRevisionFile]
) -> ModelPreheatManifest:
    return ModelPreheatManifest(
        identity=identity,
        files=tuple(
            sorted(
                (
                    ManifestFile(
                        path=encode_path(file.path),
                        size=file.size,
                        sha256=file.sha256,
                    )
                    for file in files
                ),
                key=lambda item: item.path,
            )
        ),
    )


def aggregate_local_cache_probes(
    results: list[LocalCacheProbeResult],
) -> LocalCacheProbeSummary:
    ordered = tuple(sorted(results, key=lambda item: item.worker_uuid))
    counts: dict[str, int] = {}
    for result in ordered:
        counts[result.state] = counts.get(result.state, 0) + 1
    return LocalCacheProbeSummary(
        results=ordered,
        counts=dict(sorted(counts.items())),
        all_valid=bool(ordered) and all(result.state == "valid" for result in ordered),
    )
