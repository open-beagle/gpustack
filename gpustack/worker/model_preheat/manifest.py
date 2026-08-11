import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from huggingface_hub.utils import filter_repo_objects

from gpustack.worker.model_preheat.identity import (
    ModelPreheatIdentity,
    ModelPreheatIdentityError,
    decode_path,
    encode_path,
)


class ModelPreheatManifestError(ValueError):
    pass


MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_FILES = 1024
MAX_MANIFEST_PATH_LENGTH = 1024
MAX_MANIFEST_TOTAL_SIZE = 1 << 50


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "path", validate_manifest_path(self.path))
        if len(self.path) > MAX_MANIFEST_PATH_LENGTH:
            raise ModelPreheatManifestError("manifest_path_too_long")
        if not isinstance(self.size, int) or self.size < 0:
            raise ModelPreheatManifestError("invalid_file_size")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ModelPreheatManifestError("invalid_file_sha256")

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True)
class ModelPreheatManifest:
    identity: ModelPreheatIdentity
    files: tuple[ManifestFile, ...]
    cache_key: str
    selection_digest: str
    generation_id: str
    exclude_patterns: tuple[str, ...] | list[str] = ()
    requested_revision: str | None = None
    schema_version: int = 1

    def __post_init__(self):
        if len(self.files) > MAX_MANIFEST_FILES:
            raise ModelPreheatManifestError("too_many_manifest_files")
        if sum(file.size for file in self.files) > MAX_MANIFEST_TOTAL_SIZE:
            raise ModelPreheatManifestError("manifest_total_size_too_large")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ModelPreheatManifestError("duplicate_path:files")
        for value in (self.cache_key, self.selection_digest, self.generation_id):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or "/" in value
                or "\\" in value
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                raise ModelPreheatManifestError("invalid_manifest_identifier")
        try:
            exclude_patterns = tuple(
                sorted(encode_path(pattern) for pattern in self.exclude_patterns)
            )
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc
        if len(exclude_patterns) != len(set(exclude_patterns)):
            raise ModelPreheatManifestError("duplicate_path:exclude_patterns")
        object.__setattr__(self, "exclude_patterns", exclude_patterns)
        try:
            requested_revision = encode_path(
                self.requested_revision or decode_path(self.identity.revision_path)
            )
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc
        object.__setattr__(self, "requested_revision", requested_revision)

    @property
    def total_size(self) -> int:
        return sum(file.size for file in self.files)

    @property
    def aggregate_sha256(self) -> str:
        payload = [
            {
                "path": file.path,
                "sha256": file.sha256,
                "size": file.size,
            }
            for file in self.files
        ]
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._canonical_bytes(include_digest=False)).hexdigest()

    def to_dict(self, include_digest: bool = True) -> dict:
        payload = {
            "schema_version": self.schema_version,
            "cache_key": self.cache_key,
            "source": self.identity.source,
            "model_id": self.identity.model_path,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.identity.revision_path,
            "include_patterns": list(self.identity.file_patterns),
            "exclude_patterns": list(self.exclude_patterns),
            "selection_digest": self.selection_digest,
            "generation_id": self.generation_id,
            "file_count": len(self.files),
            "total_size": self.total_size,
            "aggregate_sha256": self.aggregate_sha256,
            "identity": self.identity.to_dict(),
            "identity_digest": self.identity.digest,
            "files": [file.to_dict() for file in self.files],
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def to_json_bytes(self) -> bytes:
        return self._canonical_bytes(include_digest=True)

    def _canonical_bytes(self, include_digest: bool) -> bytes:
        return json.dumps(
            self.to_dict(include_digest=include_digest),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


def build_model_preheat_manifest(
    root_dir: str | Path,
    identity: ModelPreheatIdentity,
    *,
    cache_key: str,
    selection_digest: str,
    generation_id: str,
    exclude_patterns: tuple[str, ...] | list[str] = (),
    requested_revision: str | None = None,
) -> ModelPreheatManifest:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise ModelPreheatManifestError("root_dir_not_found")

    try:
        decoded_exclude_patterns = tuple(
            decode_path(encode_path(pattern)) for pattern in exclude_patterns
        )
    except ModelPreheatIdentityError as exc:
        raise ModelPreheatManifestError(str(exc)) from exc

    decoded_include_patterns = []
    for pattern in identity.file_patterns:
        try:
            decoded_pattern = decode_path(pattern)
            encode_path(decoded_pattern)
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc

        decoded_include_patterns.append(decoded_pattern)

    candidates = {}
    for path in root.rglob("*"):
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise ModelPreheatManifestError("path_traversal")
        if resolved.is_file() and not path.is_symlink():
            candidates[resolved.relative_to(root).as_posix()] = resolved
    selected_paths = filter_repo_objects(
        candidates,
        allow_patterns=decoded_include_patterns or None,
        ignore_patterns=list(decoded_exclude_patterns) or None,
    )
    matched_paths = {candidates[relative_path] for relative_path in selected_paths}

    if not matched_paths:
        raise ModelPreheatManifestError("no_manifest_files")

    files = []
    for path in sorted(
        matched_paths, key=lambda item: item.relative_to(root).as_posix()
    ):
        relative_path = path.relative_to(root).as_posix()
        try:
            encoded_path = encode_path(relative_path)
        except ModelPreheatIdentityError as exc:
            raise ModelPreheatManifestError(str(exc)) from exc
        files.append(
            ManifestFile(
                path=encoded_path,
                size=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )

    return ModelPreheatManifest(
        identity=identity,
        files=tuple(files),
        cache_key=cache_key,
        selection_digest=selection_digest,
        generation_id=generation_id,
        exclude_patterns=exclude_patterns,
        requested_revision=requested_revision,
    )


def validate_manifest_path(path: str) -> str:
    if not isinstance(path, str) or path == "":
        raise ModelPreheatManifestError("empty_path")
    _reject_raw_manifest_path(path)

    decoded = decode_path(path)
    _reject_decoded_manifest_path(decoded)

    try:
        canonical = encode_path(decoded)
    except ModelPreheatIdentityError as exc:
        raise ModelPreheatManifestError(str(exc)) from exc
    if canonical != path:
        raise ModelPreheatManifestError("non_canonical_path")
    return canonical


def _reject_raw_manifest_path(path: str):
    if path.startswith("/") or "\\" in path:
        raise ModelPreheatManifestError("invalid_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ModelPreheatManifestError("control_char")
    segments = path.split("/")
    if any(segment == "" for segment in segments):
        raise ModelPreheatManifestError("empty_path_segment")
    if any(segment in (".", "..") for segment in segments):
        raise ModelPreheatManifestError("path_traversal")


def _reject_decoded_manifest_path(path: str):
    if path.startswith("/") or "\\" in path:
        raise ModelPreheatManifestError("invalid_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise ModelPreheatManifestError("control_char")
    segments = path.split("/")
    if any(segment == "" for segment in segments):
        raise ModelPreheatManifestError("empty_path_segment")
    if any(unquote(segment) in (".", "..") for segment in segments):
        raise ModelPreheatManifestError("path_traversal")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
