import base64
import json
import secrets
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ModelPreheatCredentialError(Exception):
    pass


class CredentialEncryptionUnavailable(ModelPreheatCredentialError):
    pass


def generate_model_preheat_credential_key() -> str:
    key = AESGCM.generate_key(bit_length=256)
    return base64.urlsafe_b64encode(key).decode("ascii")


class ModelPreheatCredentialCipher:
    def __init__(
        self,
        current_key: Optional[str],
        current_key_version: Optional[str],
        old_keys: Optional[dict[str, str]] = None,
    ):
        self.current_key = current_key
        self.current_key_version = current_key_version or "v1"
        self.old_keys = old_keys or {}

    def encrypt(self, plaintext: str) -> dict[str, str]:
        if not self.current_key:
            raise CredentialEncryptionUnavailable("credential_encryption_unavailable")

        nonce = secrets.token_bytes(12)
        encrypted = AESGCM(self._decode_key(self.current_key)).encrypt(
            nonce,
            plaintext.encode("utf-8"),
            self.current_key_version.encode("utf-8"),
        )
        ciphertext, tag = encrypted[:-16], encrypted[-16:]
        return {
            "algorithm": "AESGCM",
            "key_version": self.current_key_version,
            "nonce": self._b64encode(nonce),
            "ciphertext": self._b64encode(ciphertext),
            "tag": self._b64encode(tag),
        }

    def decrypt(self, encrypted: dict[str, Any] | str) -> str:
        payload = self._payload(encrypted)
        key_version = payload.get("key_version")
        key = self._key_for_version(key_version)
        if not key:
            raise ModelPreheatCredentialError("credential_key_version_unavailable")

        try:
            decrypted = AESGCM(self._decode_key(key)).decrypt(
                self._b64decode(payload["nonce"]),
                self._b64decode(payload["ciphertext"])
                + self._b64decode(payload["tag"]),
                str(key_version).encode("utf-8"),
            )
        except Exception as exc:
            raise ModelPreheatCredentialError("credential_decryption_failed") from exc

        return decrypted.decode("utf-8")

    def decrypt_and_rotate(
        self, encrypted: dict[str, Any] | str
    ) -> tuple[str, Optional[dict[str, str]]]:
        plaintext = self.decrypt(encrypted)
        payload = self._payload(encrypted)
        if payload.get("key_version") == self.current_key_version:
            return plaintext, None
        return plaintext, self.encrypt(plaintext)

    def _key_for_version(self, key_version: str) -> Optional[str]:
        if key_version == self.current_key_version:
            return self.current_key
        return self.old_keys.get(key_version)

    @staticmethod
    def _payload(encrypted: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(encrypted, str):
            encrypted = json.loads(encrypted)
        if encrypted.get("algorithm") != "AESGCM":
            raise ModelPreheatCredentialError("unsupported_credential_cipher")
        required = {"key_version", "nonce", "ciphertext", "tag"}
        if not required.issubset(encrypted):
            raise ModelPreheatCredentialError("invalid_credential_ciphertext")
        return encrypted

    @staticmethod
    def _decode_key(key: str) -> bytes:
        for decoder in (_urlsafe_b64decode, bytes.fromhex):
            try:
                decoded = decoder(key)
                if len(decoded) in {16, 24, 32}:
                    return decoded
            except Exception:
                pass

        raw = key.encode("utf-8")
        if len(raw) in {16, 24, 32}:
            return raw
        raise CredentialEncryptionUnavailable("invalid_model_preheat_credential_key")

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        return _urlsafe_b64decode(value)


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
