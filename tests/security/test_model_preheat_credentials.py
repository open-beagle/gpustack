import json

import pytest

from gpustack.config.config import Config
from gpustack.model_preheat_credentials import (
    CredentialEncryptionUnavailable,
    ModelPreheatCredentialCipher,
    ModelPreheatCredentialError,
    generate_model_preheat_credential_key,
)


def test_encrypt_decrypt_round_trip_uses_authenticated_payload():
    cipher = ModelPreheatCredentialCipher(
        current_key=generate_model_preheat_credential_key(),
        current_key_version="v1",
    )

    encrypted = cipher.encrypt("access-key-1")

    assert encrypted["algorithm"] == "AESGCM"
    assert encrypted["key_version"] == "v1"
    assert encrypted["nonce"]
    assert encrypted["ciphertext"]
    assert encrypted["tag"]
    assert "access-key-1" not in repr(encrypted)
    assert cipher.decrypt(encrypted) == "access-key-1"


def test_decrypt_rejects_wrong_key():
    encrypted = ModelPreheatCredentialCipher(
        current_key=generate_model_preheat_credential_key(),
        current_key_version="v1",
    ).encrypt("secret-key-1")
    wrong_cipher = ModelPreheatCredentialCipher(
        current_key=generate_model_preheat_credential_key(),
        current_key_version="v1",
    )

    with pytest.raises(ModelPreheatCredentialError):
        wrong_cipher.decrypt(encrypted)


def test_missing_current_key_is_unavailable():
    cipher = ModelPreheatCredentialCipher(current_key=None, current_key_version="v1")

    with pytest.raises(CredentialEncryptionUnavailable):
        cipher.encrypt("secret-key-1")


def test_old_key_version_can_be_read_and_reencrypted_with_current_key():
    old_key = generate_model_preheat_credential_key()
    new_key = generate_model_preheat_credential_key()
    old_cipher = ModelPreheatCredentialCipher(
        current_key=old_key,
        current_key_version="v1",
    )
    rotated_cipher = ModelPreheatCredentialCipher(
        current_key=new_key,
        current_key_version="v2",
        old_keys={"v1": old_key},
    )

    encrypted = old_cipher.encrypt("rotated-secret")
    plaintext, reencrypted = rotated_cipher.decrypt_and_rotate(encrypted)

    assert plaintext == "rotated-secret"
    assert reencrypted is not None
    assert reencrypted["key_version"] == "v2"
    assert rotated_cipher.decrypt(reencrypted) == "rotated-secret"


def test_config_reads_model_preheat_credential_env(monkeypatch, tmp_path):
    key = generate_model_preheat_credential_key()
    monkeypatch.setenv("GPUSTACK_MODEL_PREHEAT_CREDENTIAL_KEY", key)
    monkeypatch.setenv("GPUSTACK_MODEL_PREHEAT_CREDENTIAL_KEY_VERSION", "v9")

    config = Config(data_dir=str(tmp_path))

    assert config.model_preheat_credential_key == key
    assert config.model_preheat_credential_key_version == "v9"


def test_config_reads_model_preheat_credential_old_keys_json_env(monkeypatch, tmp_path):
    old_key = generate_model_preheat_credential_key()
    monkeypatch.setenv(
        "GPUSTACK_MODEL_PREHEAT_CREDENTIAL_OLD_KEYS",
        json.dumps({"v1": old_key}),
    )

    config = Config(data_dir=str(tmp_path))

    assert config.model_preheat_credential_old_keys == {"v1": old_key}


def test_config_accepts_model_preheat_credential_kwargs(tmp_path):
    key = generate_model_preheat_credential_key()

    config = Config(
        data_dir=str(tmp_path),
        model_preheat_credential_key=key,
        model_preheat_credential_key_version="v3",
        model_preheat_credential_old_keys={"v2": key},
    )

    assert config.model_preheat_credential_key == key
    assert config.model_preheat_credential_key_version == "v3"
    assert config.model_preheat_credential_old_keys == {"v2": key}
