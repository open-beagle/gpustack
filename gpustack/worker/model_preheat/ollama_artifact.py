import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from filelock import SoftFileLock

from gpustack.worker.model_preheat.identity import ollama_model_filename
from gpustack.worker.model_preheat.s3_client import (
    ModelPreheatCanceled,
    ModelPreheatS3Conflict,
    ModelPreheatS3ManifestError,
)


def install_ollama_artifact(
    client,
    manifest,
    *,
    bucket: str,
    prefix: str,
    target_root: str | Path,
    model_id: str,
    cancel_check=None,
    progress_callback=None,
) -> Path:
    """校验并原子安装 Ollama 单文件 Artifact。"""
    expected_filename = ollama_model_filename(model_id)
    if len(manifest.files) != 1 or manifest.files[0].path != expected_filename:
        raise ModelPreheatS3ManifestError("s3_manifest_invalid")

    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / expected_filename
    lock_path = root / f"{expected_filename}.lock"
    with SoftFileLock(str(lock_path)):
        _raise_if_cancelled(cancel_check)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{expected_filename}.staging-",
                dir=root,
            )
        )
        staging_target = staging_root / expected_filename
        try:
            manifest_file = manifest.files[0]
            client.download_artifact_file(
                bucket,
                prefix,
                manifest,
                manifest_file,
                staging_target,
            )
            _raise_if_cancelled(cancel_check)
            _verify_staged_file(staging_target, manifest_file)
            if progress_callback is not None:
                progress_callback(
                    [expected_filename], manifest_file.size, manifest.total_size
                )
            os.replace(staging_target, target)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
    return target


def _verify_staged_file(path: Path, manifest_file) -> None:
    if not path.is_file():
        raise ModelPreheatS3ManifestError("s3_manifest_invalid")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != manifest_file.size or digest.hexdigest() != manifest_file.sha256:
        raise ModelPreheatS3Conflict("checksum_mismatch")


def _raise_if_cancelled(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise ModelPreheatCanceled("canceled")
