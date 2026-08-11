import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote, unquote


class ModelPreheatIdentityError(ValueError):
    pass


MAX_FILE_PATTERNS = 128
MAX_PATTERN_LENGTH = 1024

_PATH_SEGMENT_SAFE = "!$&'()*+,;=:@"
_SOURCE_ALIASES = {
    "huggingface": "huggingface",
    "model_scope": "modelscope",
    "modelscope": "modelscope",
}


def _has_control_char(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def encode_path(path: str) -> str:
    if not isinstance(path, str) or path == "":
        raise ModelPreheatIdentityError("empty_path")
    if path.startswith("/"):
        raise ModelPreheatIdentityError("absolute_path")
    if "\\" in path:
        raise ModelPreheatIdentityError("invalid_path_separator")
    if _has_control_char(path):
        raise ModelPreheatIdentityError("control_char")

    segments = path.split("/")
    if any(segment == "" for segment in segments):
        raise ModelPreheatIdentityError("empty_path_segment")
    if any(segment in (".", "..") for segment in segments):
        raise ModelPreheatIdentityError("path_traversal")

    return "/".join(quote(segment, safe=_PATH_SEGMENT_SAFE) for segment in segments)


def decode_path(path: str) -> str:
    return "/".join(unquote(segment) for segment in path.split("/"))


def normalize_source(source: str | Enum) -> str:
    value = source.value if isinstance(source, Enum) else source
    if not isinstance(value, str):
        raise ModelPreheatIdentityError("invalid_source")

    canonical = _SOURCE_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ModelPreheatIdentityError("invalid_source")
    return canonical


def _validate_glob_pattern(pattern: str):
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ModelPreheatIdentityError("pattern_too_long")
    encode_path(pattern)
    for segment in pattern.split("/"):
        if "**" in segment and segment != "**":
            raise ModelPreheatIdentityError("invalid_glob_pattern")
        _validate_glob_character_classes(segment)


def _validate_glob_character_classes(segment: str):
    index = 0
    while index < len(segment):
        if segment[index] == "]":
            raise ModelPreheatIdentityError("invalid_glob_pattern")
        if segment[index] != "[":
            index += 1
            continue
        close_index = segment.find("]", index + 1)
        if close_index == -1 or close_index == index + 1:
            raise ModelPreheatIdentityError("invalid_glob_pattern")
        index = close_index + 1


def _normalize_unique_paths(
    paths: tuple[str, ...] | list[str], field: str
) -> tuple[str, ...]:
    normalized = tuple(sorted(encode_path(path) for path in paths))
    if len(normalized) != len(set(normalized)):
        raise ModelPreheatIdentityError(f"duplicate_path:{field}")
    return normalized


def _normalize_unique_patterns(
    patterns: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    if len(patterns) > MAX_FILE_PATTERNS:
        raise ModelPreheatIdentityError("too_many_patterns")

    for pattern in patterns:
        _validate_glob_pattern(pattern)

    return _normalize_unique_paths(patterns, "file_patterns")


@dataclass(frozen=True)
class ModelPreheatIdentity:
    source: str
    model_id: str
    revision: str
    file_patterns: tuple[str, ...] | list[str]

    def __post_init__(self):
        object.__setattr__(self, "source", normalize_source(self.source))
        object.__setattr__(self, "model_path", encode_path(self.model_id))
        object.__setattr__(self, "revision_path", encode_path(self.revision))
        object.__setattr__(
            self,
            "file_patterns",
            _normalize_unique_patterns(self.file_patterns),
        )
        object.__setattr__(self, "digest", self._digest())
        object.__setattr__(
            self,
            "storage_prefix",
            (f"{self.source}/{self.model_path}/" f"{self.revision_path}/{self.digest}"),
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "model_id": self.model_path,
            "revision": self.revision_path,
            "file_patterns": list(self.file_patterns),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _digest(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()
