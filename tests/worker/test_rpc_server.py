import json
from pathlib import Path

from gpustack.worker import rpc_server
from gpustack.worker.tools_manager import BUILTIN_LLAMA_BOX_VERSION


def test_rpc_server_uses_visible_device_without_unsupported_origin_argument(
    monkeypatch, tmp_path
):
    captured = {}
    third_party_bin = tmp_path / "third-party"
    version_dir = f"llama-box-{BUILTIN_LLAMA_BOX_VERSION}-linux-amd64-cuda"
    bundled_command = third_party_bin / "llama-box" / version_dir / "llama-box"
    bundled_command.parent.mkdir(parents=True)
    bundled_command.write_text("#!/bin/sh\n", encoding="utf-8")
    (third_party_bin / "versions.json").write_text(
        json.dumps({version_dir: BUILTIN_LLAMA_BOX_VERSION}), encoding="utf-8"
    )
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(third_party_bin))
    monkeypatch.setenv("GPUSTACK_DISABLE_DYNAMIC_LINK_LLAMA_BOX", "true")

    monkeypatch.setattr(
        rpc_server,
        "third_party_bin_path",
        lambda _: tmp_path / "llama-box-default",
    )
    monkeypatch.setattr(rpc_server.platform, "system", lambda: "linux")
    monkeypatch.setattr(rpc_server.platform, "arch", lambda: "amd64")
    monkeypatch.setattr(rpc_server.platform, "device", lambda: "cuda")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]

    monkeypatch.setattr(rpc_server.subprocess, "run", fake_run)

    rpc_server.RPCServer._start(
        port=40064,
        gpu_index=3,
        vendor="nvidia",
        cache_dir=str(tmp_path / "cache"),
        bin_dir=str(tmp_path / "data-bin"),
    )

    command = [
        str(part) if isinstance(part, Path) else part for part in captured["command"]
    ]
    assert command[command.index("--rpc-server-main-gpu") + 1] == "0"
    assert "--origin-rpc-server-main-gpu" not in command
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert str(tmp_path / "llama-box-default") in captured["env"]["LD_LIBRARY_PATH"]
