from pathlib import Path

from gpustack.utils.third_party import third_party_bin_path


def test_third_party_bin_path_prefers_external_tools_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUSTACK_THIRD_PARTY_BIN", str(tmp_path))

    assert third_party_bin_path("fastfetch", "fastfetch") == (
        tmp_path / "fastfetch" / "fastfetch"
    )


def test_runtime_tool_paths_use_third_party_helper():
    checked_files = [
        Path("gpustack/detectors/fastfetch/fastfetch.py"),
        Path("gpustack/scheduler/calculator.py"),
        Path("gpustack/worker/backends/llama_box.py"),
        Path("gpustack/worker/rpc_server.py"),
    ]

    for path in checked_files:
        assert "gpustack.third_party.bin" not in path.read_text(encoding="utf-8")
