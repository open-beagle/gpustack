"""统一模型存储的冻结扫描规约。

扫描规约同时包含两类信息：

- ``root`` 是 Worker 执行时使用的物理绝对路径，只用于定位本地文件；
- ``include_patterns`` 是模型仓库内的逻辑文件选择，会进入请求和 Artifact
  身份，不能包含 Worker 的挂载点、缓存目录或其他宿主路径。

完整仓库使用空 patterns，Manifest 路径相对仓库内容。部分文件同步只接受
同一物理父目录下的直接子项，并使用稳定基名；跨父目录的多源无法在不引入
宿主目录或 staging 映射的情况下表达，因此稳定拒绝。
"""

from pathlib import PurePosixPath


def compute_scan_spec(
    source_paths: list[str], *, repository_complete: bool = True
) -> tuple[str, list[str]]:
    """返回冻结的 ``(physical_root, logical_include_patterns)``。

    ``repository_complete=True`` 表示唯一源路径是完整模型仓库目录，逻辑选择
    为空（全量）。否则每个源必须是同一父目录下的直接子项，逻辑选择使用其
    基名及子树形式。函数不访问文件系统，Server 和 Worker 可共享同一结果。
    """
    normalized = [PurePosixPath(str(path).rstrip("/")) for path in (source_paths or [])]
    normalized = [
        path
        for path in normalized
        if path not in (PurePosixPath("."), PurePosixPath(""))
    ]
    if not normalized or any(not path.is_absolute() for path in normalized):
        raise ValueError("model_sync_source_not_found")

    if repository_complete:
        if len(normalized) != 1:
            raise ValueError("model_sync_source_conflict")
        return normalized[0].as_posix(), []

    names = [path.name for path in normalized]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("model_sync_source_conflict")
    parents = {path.parent for path in normalized}
    if len(parents) != 1:
        raise ValueError("model_sync_source_conflict")

    root = next(iter(parents))
    patterns = sorted(pattern for name in names for pattern in (name, f"{name}/**"))
    return root.as_posix(), patterns
