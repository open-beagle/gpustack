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
    # 请求身份字段：仅参与 request_digest，不参与 Artifact 身份。
    # main/master 等移动 revision 只允许出现在这里。
    requested_revision: str | None = None
    exclude_patterns: tuple[str, ...] | list[str] = ()

    def __post_init__(self):
        object.__setattr__(self, "source", normalize_source(self.source))
        object.__setattr__(self, "model_path", encode_path(self.model_id))
        object.__setattr__(self, "revision_path", encode_path(self.revision))
        object.__setattr__(
            self,
            "file_patterns",
            _normalize_unique_patterns(self.file_patterns),
        )
        object.__setattr__(
            self,
            "exclude_patterns",
            _normalize_unique_patterns(self.exclude_patterns),
        )
        requested_revision = (
            None
            if self.requested_revision is None
            else encode_path(self.requested_revision)
        )
        object.__setattr__(self, "requested_revision_path", requested_revision)
        # request_digest 使用编码后的规范值做摘要，
        # 与 manifest.compute_request_digest 保持同一语义。
        object.__setattr__(
            self,
            "request_digest",
            _canonical_sha256(
                {
                    "source": self.source,
                    "model_id": self.model_path,
                    "requested_revision": requested_revision,
                    "include_patterns": sorted(self.file_patterns),
                    "exclude_patterns": sorted(self.exclude_patterns),
                }
            ),
        )

    def artifact_prefix(self, profile_prefix: str) -> str:
        """统一 Artifact 的模型级前缀（不含 artifact_id 末段）。

        结构为 `<prefix>/<source>/<organization>/<model>`（设计文档第 6 节），
        不包含 resolved revision 路径段（revision 已编码进 artifact_id）、
        协议版本目录、requested_revision 或 generation。

        profile_prefix 必须显式传入：空串表示“无前缀”，非空时必须是
        安全的对象段（多段以 `/` 分隔，禁止 `.`/`..`/控制字符）。
        """
        return self._join_object_name(
            profile_prefix,
            self.source,
            self.model_path,
        )

    @staticmethod
    def _join_object_name(*segments: str) -> str:
        clean_segments = []
        for segment in segments:
            if not isinstance(segment, str):
                raise ModelPreheatIdentityError("invalid_path")
            if segment == "":
                continue
            clean_segment = segment.strip("/")
            if clean_segment == "":
                continue
            ModelPreheatIdentity._validate_segment(clean_segment)
            clean_segments.append(clean_segment)
        if not clean_segments:
            raise ModelPreheatIdentityError("empty_path")
        return "/".join(clean_segments)

    @staticmethod
    def _validate_segment(segment: str):
        if segment.startswith("/") or "\\" in segment:
            raise ModelPreheatIdentityError("invalid_path")
        if any(ord(char) < 32 or ord(char) == 127 for char in segment):
            raise ModelPreheatIdentityError("control_char")
        for part in segment.split("/"):
            if part in ("", ".", ".."):
                raise ModelPreheatIdentityError("invalid_path_segment")

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


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
