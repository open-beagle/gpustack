import os
from pathlib import Path

from gpustack.utils.compat_importlib import pkg_resources


THIRD_PARTY_BIN_ENV = "GPUSTACK_THIRD_PARTY_BIN"


def third_party_bin_path(*parts: str) -> Path:
    third_party_bin = os.getenv(THIRD_PARTY_BIN_ENV)
    if third_party_bin:
        return Path(third_party_bin).joinpath(*parts)

    return Path(
        str(pkg_resources.files("gpustack.third_party").joinpath("bin", *parts))
    )
