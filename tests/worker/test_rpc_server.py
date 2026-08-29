from pathlib import Path

from gpustack.worker import rpc_server


def test_rpc_server_uses_visible_device_without_unsupported_origin_argument(
    monkeypatch, tmp_path
):
    captured = {}

    monkeypatch.setattr(
        rpc_server,
        "third_party_bin_path",
        lambda _: tmp_path / "llama-box-default",
    )
    monkeypatch.setattr(rpc_server, "is_disabled_dynamic_link", lambda _: False)
    monkeypatch.setattr(rpc_server.platform, "system", lambda: "linux")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]

    monkeypatch.setattr(rpc_server.subprocess, "run", fake_run)

    rpc_server.RPCServer._start(
        port=40064,
        gpu_index=3,
        vendor="nvidia",
        cache_dir=str(tmp_path / "cache"),
    )

    command = [
        str(part) if isinstance(part, Path) else part for part in captured["command"]
    ]
    assert command[command.index("--rpc-server-main-gpu") + 1] == "0"
    assert "--origin-rpc-server-main-gpu" not in command
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert str(tmp_path / "llama-box-default") in captured["env"]["LD_LIBRARY_PATH"]
