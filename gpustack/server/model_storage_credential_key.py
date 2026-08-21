"""统一模型存储凭据加密主密钥的最小生命周期管理。

任务 2 步骤 5（设计文档 §5.3）：

- ``GPUSTACK_MODEL_PREHEAT_CREDENTIAL_KEY``（映射到 ``config.model_preheat_credential_key``）
  存在时始终优先，并沿用 ``model_preheat_credential_key_version``；
- 环境变量缺失时，在 ``<data_dir>/model_preheat_credential_key`` 原子生成密钥文件，
  POSIX 权限为 ``0600``；重启复用同一文件；
- 数据目录不可持久化或权限不安全时启动失败，不降级为明文保存；
- 多 Server 部署必须通过同一个 Secret 注入相同环境变量，不依赖各实例本地生成的文件；
- 保留现有 ``GPUSTACK_MODEL_PREHEAT_CREDENTIAL_OLD_KEYS`` 解密能力（由
  :class:`gpustack.model_preheat_credentials.ModelPreheatCredentialCipher` 处理），
  不新增数据库密钥指纹、在线轮换 API 或后台服务；
- 密钥只写入返回的 ``config`` 内存字段与受保护文件，绝不进入日志、数据库或 API。
"""

import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from typing import Optional

from gpustack.model_preheat_credentials import generate_model_preheat_credential_key

logger = logging.getLogger(__name__)

CREDENTIAL_KEY_FILENAME = "model_preheat_credential_key"
# 密钥文件目标权限：仅属主可读写（POSIX 0600）。
CREDENTIAL_KEY_MODE = 0o600


class ModelStorageCredentialKeyError(RuntimeError):
    """密钥文件无法安全生成或读取。

    消息固定为稳定错误码，不包含密钥。
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ModelStorageCredentialKeyResult:
    # 供 :class:`ModelPreheatCredentialCipher` 使用的主密钥（urlsafe base64 或十六进制）。
    key: str
    key_version: str
    # 是否来自环境变量（多 Server 必须一致注入），还是本地生成的密钥文件。
    from_environment: bool
    # 密钥文件路径；来自环境变量时为 ``None``。
    key_path: Optional[str] = None


def ensure_model_preheat_credential_key(config) -> ModelStorageCredentialKeyResult:
    """确保当前 Server 拥有可用于加密 Profile 凭据的主密钥，并写回 ``config``。

    优先级与失败语义见模块 docstring。成功时更新
    ``config.model_preheat_credential_key``（以及缺失时的 version），失败时抛出
    :class:`ModelStorageCredentialKeyError`，调用方应使 Server 启动失败。
    """
    env_key = getattr(config, "model_preheat_credential_key", None)
    if env_key:
        version = getattr(config, "model_preheat_credential_key_version", None) or "v1"
        config.model_preheat_credential_key = env_key
        config.model_preheat_credential_key_version = version
        # 环境变量密钥由部署 Secret 保证多 Server 一致；这里不做文件持久化。
        return ModelStorageCredentialKeyResult(
            key=env_key,
            key_version=version,
            from_environment=True,
            key_path=None,
        )

    data_dir = getattr(config, "data_dir", None)
    if not data_dir:
        raise ModelStorageCredentialKeyError("credential_key_data_dir_unavailable")

    key = _read_or_create_key_file(data_dir)
    version = getattr(config, "model_preheat_credential_key_version", None) or "v1"
    config.model_preheat_credential_key = key
    config.model_preheat_credential_key_version = version
    # 只记录来源类别，绝不记录密钥内容。
    logger.info("model storage credential key loaded from key file")
    return ModelStorageCredentialKeyResult(
        key=key,
        key_version=version,
        from_environment=False,
        key_path=os.path.join(data_dir, CREDENTIAL_KEY_FILENAME),
    )


def _read_or_create_key_file(data_dir: str) -> str:
    _ensure_persistent_secure_dir(data_dir)
    key_path = os.path.join(data_dir, CREDENTIAL_KEY_FILENAME)
    if not os.path.lexists(key_path):
        return _write_key_file_atomically(data_dir)

    # 已存在：校验并复用，绝不允许覆盖（覆盖会丢失既有凭据的解密能力）。
    _validate_existing_key_file(key_path)
    key = _read_key_file(key_path)
    if not key:
        # 文件存在但为空：视为损坏。覆盖会丢失解密能力，必须失败并要求人工处理。
        raise ModelStorageCredentialKeyError("credential_key_file_empty")
    return key


def _validate_existing_key_file(key_path: str) -> None:
    """校验既有密钥文件安全属性，不安全即失败，避免被替换/窃取。

    - 必须是普通文件（拒绝符号链接、目录、FIFO 等）；
    - POSIX 权限必须为 ``0600``；
    - 属主必须是当前进程用户（防止跨用户窃取）。
    """
    st = os.lstat(key_path)
    if not stat.S_ISREG(st.st_mode):
        raise ModelStorageCredentialKeyError("credential_key_file_not_secure")
    mode = stat.S_IMODE(st.st_mode)
    if mode != CREDENTIAL_KEY_MODE:
        # 尝试收紧到 0600；无法收紧（非属主等）则失败。
        try:
            os.chmod(key_path, CREDENTIAL_KEY_MODE)
            mode = stat.S_IMODE(os.lstat(key_path).st_mode)
        except OSError:
            raise ModelStorageCredentialKeyError(
                "credential_key_file_not_secure"
            ) from None
        if mode != CREDENTIAL_KEY_MODE:
            raise ModelStorageCredentialKeyError("credential_key_file_not_secure")
    if st.st_uid != _current_uid():
        raise ModelStorageCredentialKeyError("credential_key_file_not_owned")


def _current_uid() -> int:
    # 属主校验以“当前进程用户”为准（POSIX 为 os.getuid()）；
    # 在无 getuid 的平台回退到 HOME 目录属主。
    try:
        return os.getuid()
    except Exception:
        return os.stat(os.path.expanduser("~")).st_uid


def _ensure_persistent_secure_dir(data_dir: str) -> None:
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as exc:
        # 目录不可持久化（只读/权限不足/路径非法）时启动失败。
        raise ModelStorageCredentialKeyError(
            "credential_key_data_dir_not_persistent"
        ) from exc
    if not os.path.isdir(data_dir):
        raise ModelStorageCredentialKeyError(
            "credential_key_data_dir_not_persistent"
        )
    # 目录若可被其他用户写入，则密钥文件可能被窃取或替换，视为不安全。
    try:
        mode = stat.S_IMODE(os.stat(data_dir).st_mode)
    except OSError as exc:
        raise ModelStorageCredentialKeyError(
            "credential_key_data_dir_not_persistent"
        ) from exc
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ModelStorageCredentialKeyError("credential_key_data_dir_not_secure")


def _read_key_file(key_path: str) -> Optional[str]:
    try:
        with open(key_path, "r", encoding="utf-8") as file:
            key = file.read().strip()
    except OSError:
        # 读失败不得当作“无密钥”而覆盖，交由调用方以明确错误失败。
        raise ModelStorageCredentialKeyError(
            "credential_key_file_unreadable"
        ) from None
    return key or None


def _write_key_file_atomically(data_dir: str) -> str:
    key = generate_model_preheat_credential_key()
    key_path = os.path.join(data_dir, CREDENTIAL_KEY_FILENAME)
    fd, tmp_path = tempfile.mkstemp(
        prefix="." + CREDENTIAL_KEY_FILENAME + ".", dir=data_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(key + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.chmod(tmp_path, CREDENTIAL_KEY_MODE)
        os.replace(tmp_path, key_path)
    except OSError as exc:
        # 原子写入或权限设置失败视为启动失败。
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ModelStorageCredentialKeyError(
            "credential_key_file_not_secure"
        ) from exc
    # 复核最终权限，确保 0600。
    try:
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
    except OSError as exc:
        raise ModelStorageCredentialKeyError(
            "credential_key_file_not_secure"
        ) from exc
    if mode != CREDENTIAL_KEY_MODE:
        try:
            os.chmod(key_path, CREDENTIAL_KEY_MODE)
            mode = stat.S_IMODE(os.stat(key_path).st_mode)
        except OSError as exc:
            raise ModelStorageCredentialKeyError(
                "credential_key_file_not_secure"
            ) from exc
        if mode != CREDENTIAL_KEY_MODE:
            raise ModelStorageCredentialKeyError("credential_key_file_not_secure")
    return key
