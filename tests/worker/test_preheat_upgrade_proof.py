import os
import stat

from gpustack.worker.preheat_credential import (
    WORKER_UPGRADE_PROOF_FILENAME,
    clear_worker_upgrade_proof,
    load_or_create_worker_upgrade_proof,
)


def test_worker_upgrade_proof_is_private_persistent_and_cleared_after_credential(
    tmp_path, monkeypatch
):
    proof = load_or_create_worker_upgrade_proof(str(tmp_path))
    proof_path = tmp_path / WORKER_UPGRADE_PROOF_FILENAME

    assert proof_path.read_text(encoding="utf-8") == proof
    assert stat.S_IMODE(os.stat(proof_path).st_mode) == 0o600
    assert load_or_create_worker_upgrade_proof(str(tmp_path)) == proof

    unlink = os.unlink
    monkeypatch.setattr("gpustack.worker.preheat_credential.os.unlink", unlink)
    clear_worker_upgrade_proof(str(tmp_path))

    assert not proof_path.exists()


def test_worker_upgrade_proof_clear_failure_is_ignored(tmp_path, monkeypatch):
    proof = load_or_create_worker_upgrade_proof(str(tmp_path))
    proof_path = tmp_path / WORKER_UPGRADE_PROOF_FILENAME

    monkeypatch.setattr(
        "gpustack.worker.preheat_credential.os.unlink",
        lambda _: (_ for _ in ()).throw(OSError("simulated unlink failure")),
    )

    clear_worker_upgrade_proof(str(tmp_path))

    assert proof_path.read_text(encoding="utf-8") == proof
