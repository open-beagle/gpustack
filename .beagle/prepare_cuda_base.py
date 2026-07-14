#!/usr/bin/env python3
"""Prepare and validate the Python CUDA runtime in the base image."""

import importlib.metadata as metadata
from pathlib import Path
import sys
import sysconfig

from packaging.version import Version


def patch_transformers_rope_validation() -> None:
    import transformers.modeling_rope_utils as rope_utils

    path = Path(rope_utils.__file__)
    content = path.read_text()
    patched = content.replace(
        "received_keys -= ignore_keys", "received_keys -= set(ignore_keys)"
    )
    if patched != content:
        path.write_text(patched)
        print(f"Patched Transformers RoPE validation in {path}")


def nccl_library_directory() -> Path:
    return Path(sysconfig.get_path("purelib")) / "nvidia" / "nccl" / "lib"


def validate_argcomplete() -> None:
    version = Version(metadata.version("argcomplete"))
    if version < Version("1.9.4"):
        raise RuntimeError(f"argcomplete>=1.9.4 is required, found {version}")
    print(f"argcomplete {version}")


def validate_nccl_runtime() -> None:
    libraries = sorted(nccl_library_directory().glob("libnccl.so*"))
    if not libraries:
        raise RuntimeError(
            f"NCCL runtime not found in {nccl_library_directory()}"
        )

    print("NCCL runtime found:")
    for library in libraries:
        print(f"  {library}")


def main() -> int:
    patch_transformers_rope_validation()
    validate_argcomplete()
    validate_nccl_runtime()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"CUDA base preparation failed: {error}", file=sys.stderr)
        raise
