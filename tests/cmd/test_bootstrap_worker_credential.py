from unittest.mock import Mock

from gpustack.worker.preheat_credential import bootstrap_remote_worker_credential


def test_remote_worker_bootstrap_uses_admin_api_key_from_stdin_and_writes_credential(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "worker_uuid").write_text("worker-uuid", encoding="utf-8")
    get_response = Mock()
    get_response.raise_for_status = Mock()
    get_response.json.return_value = {
        "items": [{"id": 7, "worker_uuid": "worker-uuid"}],
    }
    post_response = Mock()
    post_response.raise_for_status = Mock()
    post_response.json.return_value = {
        "worker_id": 7,
        "worker_uuid": "worker-uuid",
        "credential": "mpw_7_bootstrap-secret",
    }
    get = Mock(return_value=get_response)
    post = Mock(return_value=post_response)
    monkeypatch.setattr("gpustack.worker.preheat_credential.requests.get", get)
    monkeypatch.setattr("gpustack.worker.preheat_credential.requests.post", post)

    bootstrap_remote_worker_credential(
        "https://server.example", str(tmp_path), "admin-api-key"
    )

    assert (tmp_path / "model_preheat_worker_credential").read_text(
        encoding="utf-8"
    ) == "mpw_7_bootstrap-secret"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer admin-api-key"}
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer admin-api-key"}
    assert "bootstrap-secret" not in capsys.readouterr().out
