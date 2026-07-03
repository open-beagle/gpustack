import enum
import importlib


def test_sitecustomize_adds_strenum_when_missing(monkeypatch):
    monkeypatch.delattr(enum, "StrEnum", raising=False)

    import gpustack._sitecustomize as sitecustomize

    importlib.reload(sitecustomize)

    assert issubclass(enum.StrEnum, str)
    assert issubclass(enum.StrEnum, enum.Enum)

    class Backend(enum.StrEnum):
        VLLM_OMNI = "vllm-omni"

    assert str(Backend.VLLM_OMNI) == "vllm-omni"
