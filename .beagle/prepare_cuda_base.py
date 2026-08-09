#!/usr/bin/env python3
"""Validate the Python and CUDA runtime in the base image."""

import importlib.metadata as metadata
import sys
import sysconfig
from pathlib import Path

from packaging.version import Version


EXPECTED_DISTRIBUTIONS = {
    "torch": "2.11.0",
    "vllm": "0.26.0",
    "vllm-omni": "0.26.0",
}


def nccl_library_directory() -> Path:
    return Path(sysconfig.get_path("purelib")) / "nvidia" / "nccl" / "lib"


def validate_argcomplete() -> None:
    version = Version(metadata.version("argcomplete"))
    if version < Version("1.9.4"):
        raise RuntimeError(f"argcomplete>=1.9.4 is required, found {version}")
    print(f"argcomplete {version}")


def validate_distribution_versions() -> None:
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        actual = metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"{distribution}=={expected} is required, found {actual}"
            )
        print(f"{distribution} {actual}")


def validate_cuda_runtime() -> None:
    import torch

    if torch.version.cuda != "13.0":
        raise RuntimeError(
            f"PyTorch CUDA 13.0 runtime is required, found {torch.version.cuda}"
        )
    print(f"PyTorch CUDA {torch.version.cuda}")


def validate_nccl_runtime() -> None:
    libraries = sorted(nccl_library_directory().glob("libnccl.so*"))
    if not libraries:
        raise RuntimeError(f"NCCL runtime not found in {nccl_library_directory()}")

    print("NCCL runtime found:")
    for library in libraries:
        print(f"  {library}")


def main() -> int:
    validate_argcomplete()
    validate_distribution_versions()
    validate_cuda_runtime()
    validate_nccl_runtime()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CUDA base preparation failed: {error}", file=sys.stderr)
        raise
