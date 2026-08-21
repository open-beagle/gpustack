"""任务 2 步骤 5：凭据加密主密钥最小生命周期的定向测试。

覆盖：环境变量优先、首次生成、重启复用、目录不可持久/不安全启动失败、
文件权限 0600、日志脱敏（日志不含密钥）。
"""

import logging
import os
import stat
from types import SimpleNamespace

import pytest

from gpustack.model_preheat_credentials import (
    ModelPreheatCredentialCipher,
    generate_model_preheat_credential_key,
)
from gpustack.server.model_storage_credential_key import (
    CREDENTIAL_KEY_FILENAME,
    ModelStorageCredentialKeyError,
    ensure_model_preheat_credential_key,
)


def _config(data_dir, **overrides):
    base = dict(
        data_dir=data_dir,
        model_preheat_credential_key=None,
        model_preheat_credential_key_version="v1",
        model_preheat_credential_old_keys=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_environment_variable_key_takes_priority(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "GPUSTACK_MODEL_PREHEAT_CREDENTIAL_KEY", "env-provided-key"
    )
    config = _config(
        tmp_path,
        model_preheat_credential_key="env-provided-key",
        model_preheat_credential_key_version="v7",
    )
    result = ensure_model_preheat_credential_key(config)
    assert result.from_environment is True
    assert result.key == "env-provided-key"
    assert result.key_version == "v7"
    assert result.key_path is None
    # 环境变量优先：不生成密钥文件。
    assert not os.path.exists(str(tmp_path / CREDENTIAL_KEY_FILENAME))
    assert config.model_preheat_credential_key == "env-provided-key"
    assert config.model_preheat_credential_key_version == "v7"


def test_first_run_generates_key_file_with_0600(tmp_path):
    config = _config(tmp_path)
    result = ensure_model_preheat_credential_key(config)
    assert result.from_environment is False
    key_path = tmp_path / CREDENTIAL_KEY_FILENAME
    assert result.key_path == str(key_path)
    assert key_path.exists()
    mode = stat.S_IMODE(os.stat(str(key_path)).st_mode)
    assert mode == 0o600
    assert result.key == key_path.read_text().strip()
    assert config.model_preheat_credential_key == result.key


def test_restart_reuses_existing_key_file(tmp_path):
    config = _config(tmp_path)
    first = ensure_model_preheat_credential_key(config)
    # 模拟重启：新 config 实例，同一 data_dir。
    restarted = _config(tmp_path)
    second = ensure_model_preheat_credential_key(restarted)
    assert first.key == second.key
    assert second.from_environment is False


def test_key_file_reusable_for_encryption(tmp_path):
    config = _config(tmp_path)
    result = ensure_model_preheat_credential_key(config)
    cipher = ModelPreheatCredentialCipher(
        current_key=result.key,
        current_key_version=result.key_version,
        old_keys=None,
    )
    encrypted = cipher.encrypt("s3-secret-value")
    assert cipher.decrypt(encrypted) == "s3-secret-value"


def test_old_key_still_decrypts_with_env_key_and_old_keys(tmp_path):
    old_key = generate_model_preheat_credential_key()
    new_key = generate_model_preheat_credential_key()
    old_cipher = ModelPreheatCredentialCipher(
        current_key=old_key, current_key_version="v1", old_keys=None
    )
    encrypted = old_cipher.encrypt("legacy-credential")

    config = _config(
        tmp_path,
        model_preheat_credential_key=new_key,
        model_preheat_credential_key_version="v2",
        model_preheat_credential_old_keys={"v1": old_key},
    )
    result = ensure_model_preheat_credential_key(config)
    assert result.key == new_key
    # 旧版本凭据仍可用 OLD_KEYS 解密。
    cipher = ModelPreheatCredentialCipher(
        current_key=result.key,
        current_key_version=result.key_version,
        old_keys={"v1": old_key},
    )
    assert cipher.decrypt(encrypted) == "legacy-credential"


def test_data_dir_group_writable_is_rejected(tmp_path, monkeypatch):
    # 目录若可被其他用户写入，则密钥文件可能被窃取/替换，视为不安全。
    os.chmod(str(tmp_path), 0o777)
    try:
        config = _config(tmp_path)
        with pytest.raises(ModelStorageCredentialKeyError) as exc:
            ensure_model_preheat_credential_key(config)
        assert exc.value.reason == "credential_key_data_dir_not_secure"
    finally:
        os.chmod(str(tmp_path), 0o755)


def test_missing_data_dir_fails_with_stable_error(tmp_path):
    config = _config(None)
    with pytest.raises(ModelStorageCredentialKeyError) as exc:
        ensure_model_preheat_credential_key(config)
    assert exc.value.reason == "credential_key_data_dir_unavailable"


def test_unreadable_data_dir_fails(tmp_path, monkeypatch):
    # 路径指向已存在的普通文件，无法作为持久化目录。
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir")
    config = _config(str(blocker))
    with pytest.raises(ModelStorageCredentialKeyError) as exc:
        ensure_model_preheat_credential_key(config)
    assert exc.value.reason == "credential_key_data_dir_not_persistent"


def test_log_does_not_leak_key(tmp_path, caplog):
    config = _config(tmp_path)
    with caplog.at_level(logging.INFO):
        result = ensure_model_preheat_credential_key(config)
    for record in caplog.records:
        assert result.key not in record.getMessage()
        assert result.key not in record.args


def test_existing_key_file_is_reused_not_overwritten(tmp_path):
    """已存在的合法密钥文件必须复用，绝不能覆盖（覆盖会丢失解密能力）。"""
    key_path = tmp_path / CREDENTIAL_KEY_FILENAME
    original = generate_model_preheat_credential_key()
    key_path.write_text(original + "\n")
    os.chmod(str(key_path), 0o600)

    result = ensure_model_preheat_credential_key(_config(tmp_path))
    assert result.key == original
    assert key_path.read_text().strip() == original


def test_existing_key_file_world_writable_is_tightened_and_reused(tmp_path):
    """已有密钥文件若权限不安全（如 0644），收紧到 0600 后复用，不覆盖。"""
    key_path = tmp_path / CREDENTIAL_KEY_FILENAME
    original = generate_model_preheat_credential_key()
    key_path.write_text(original + "\n")
    os.chmod(str(key_path), 0o644)

    result = ensure_model_preheat_credential_key(_config(tmp_path))
    assert result.key == original
    assert stat.S_IMODE(os.stat(str(key_path)).st_mode) == 0o600


def test_existing_empty_key_file_fails_not_overwritten(tmp_path):
    """密钥文件存在但为空：视为损坏，必须失败且不得覆盖（避免丢失解密能力）。"""
    key_path = tmp_path / CREDENTIAL_KEY_FILENAME
    key_path.write_text("")
    os.chmod(str(key_path), 0o600)

    with pytest.raises(ModelStorageCredentialKeyError) as exc:
        ensure_model_preheat_credential_key(_config(tmp_path))
    assert exc.value.reason == "credential_key_file_empty"
    # 未覆盖：仍为空。
    assert key_path.read_text() == ""


def test_existing_key_file_symlink_is_rejected(tmp_path):
    """密钥文件若是指向别处的符号链接，必须拒绝（防止被替换/窃取）。"""
    target = tmp_path / "real.key"
    target.write_text(generate_model_preheat_credential_key() + "\n")
    key_path = tmp_path / CREDENTIAL_KEY_FILENAME
    os.symlink(str(target), str(key_path))

    with pytest.raises(ModelStorageCredentialKeyError) as exc:
        ensure_model_preheat_credential_key(_config(tmp_path))
    assert exc.value.reason == "credential_key_file_not_secure"
