import os
import secrets
import tempfile

import requests


WORKER_CREDENTIAL_FILENAME = "model_preheat_worker_credential"
WORKER_UPGRADE_PROOF_FILENAME = "model_preheat_worker_upgrade_proof"
WORKER_UUID_FILENAME = "worker_uuid"


class WorkerCredentialBootstrapError(RuntimeError):
    """远程 Worker 引导失败时返回不包含敏感信息的错误。"""


def store_preheat_credential(credential_path: str, credential: str) -> None:
    """原子写入 Worker 专用凭据，失败时保留原文件。"""
    directory = os.path.dirname(credential_path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor = None
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(credential_path)}.",
            dir=directory,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            descriptor = None
            file.write(credential)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, credential_path)
        temporary_path = None
        _fsync_directory(directory)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise


def load_or_create_worker_upgrade_proof(data_dir: str) -> str:
    """返回 Worker 升级交接 proof；仅在本机专用凭据缺失时由调用方使用。"""
    proof_path = os.path.join(data_dir, WORKER_UPGRADE_PROOF_FILENAME)
    try:
        with open(proof_path, "r", encoding="utf-8") as file:
            proof = file.read().strip()
        if _is_upgrade_proof(proof):
            return proof
    except FileNotFoundError:
        pass
    proof = secrets.token_urlsafe(32)
    store_preheat_credential(proof_path, proof)
    return proof


def clear_worker_upgrade_proof(data_dir: str) -> None:
    """凭据落盘后尽力删除 proof；失败不记录敏感内容。"""
    proof_path = os.path.join(data_dir, WORKER_UPGRADE_PROOF_FILENAME)
    try:
        os.unlink(proof_path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def bootstrap_remote_worker_credential(
    server_url: str,
    data_dir: str,
    admin_api_key: str,
    *,
    timeout: float = 15,
) -> None:
    """使用管理员 API key 为已有远程 Worker 安全写入一次性凭据。"""
    if not admin_api_key:
        raise WorkerCredentialBootstrapError("administrator_api_key_required")
    credential_path = os.path.join(data_dir, WORKER_CREDENTIAL_FILENAME)
    if os.path.exists(credential_path):
        raise WorkerCredentialBootstrapError("worker_credential_file_already_exists")
    worker_uuid = _read_worker_uuid(data_dir)
    base_url = server_url.rstrip("/")
    if not base_url:
        raise WorkerCredentialBootstrapError("server_url_required")
    headers = {"Authorization": f"Bearer {admin_api_key}"}
    try:
        workers_response = requests.get(
            f"{base_url}/v1/workers",
            params={"uuid": worker_uuid, "page": 1, "perPage": 2},
            headers=headers,
            timeout=timeout,
        )
        workers_response.raise_for_status()
        workers = workers_response.json().get("items", [])
        matching_workers = [
            worker
            for worker in workers
            if worker.get("worker_uuid") == worker_uuid and worker.get("id") is not None
        ]
        if len(matching_workers) != 1:
            raise WorkerCredentialBootstrapError("worker_identity_not_unique")
        worker_id = matching_workers[0]["id"]
        credential_response = requests.post(
            f"{base_url}/v1/workers/{worker_id}/model-preheat-credential",
            headers=headers,
            timeout=timeout,
        )
        credential_response.raise_for_status()
        payload = credential_response.json()
    except WorkerCredentialBootstrapError:
        raise
    except requests.RequestException as error:
        raise WorkerCredentialBootstrapError(
            "worker_credential_bootstrap_request_failed"
        ) from error
    except (TypeError, ValueError, KeyError) as error:
        raise WorkerCredentialBootstrapError(
            "worker_credential_bootstrap_response_invalid"
        ) from error

    credential = payload.get("credential")
    if (
        payload.get("worker_id") != worker_id
        or payload.get("worker_uuid") != worker_uuid
        or not isinstance(credential, str)
        or not credential.startswith("mpw_")
    ):
        raise WorkerCredentialBootstrapError(
            "worker_credential_bootstrap_response_invalid"
        )
    store_preheat_credential(credential_path, credential)


def _read_worker_uuid(data_dir: str) -> str:
    worker_uuid_path = os.path.join(data_dir, WORKER_UUID_FILENAME)
    try:
        with open(worker_uuid_path, "r", encoding="utf-8") as file:
            worker_uuid = file.read().strip()
    except OSError as error:
        raise WorkerCredentialBootstrapError("worker_uuid_unavailable") from error
    if not worker_uuid:
        raise WorkerCredentialBootstrapError("worker_uuid_unavailable")
    return worker_uuid


def _fsync_directory(directory: str) -> None:
    """POSIX 文件系统上尽力持久化目录项，其他平台保持兼容。"""
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | directory_flag)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _is_upgrade_proof(proof: str) -> bool:
    return (
        isinstance(proof, str)
        and len(proof) >= 43
        and all(character.isalnum() or character in "-_" for character in proof)
    )
