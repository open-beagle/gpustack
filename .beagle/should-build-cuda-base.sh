#!/bin/sh

set -eu

if [ "${BUILD_RUNTIME_ASSETS:-}" = "true" ]; then
  echo "BUILD_RUNTIME_ASSETS=true, building CUDA runtime base." >&2
  exit 0
fi

if [ "${BUILD_RUNTIME_ASSETS:-}" = "false" ]; then
  echo "BUILD_RUNTIME_ASSETS=false, skipping CUDA runtime base." >&2
  exit 1
fi

before="${DRONE_COMMIT_BEFORE:-}"
after="${DRONE_COMMIT_AFTER:-${DRONE_COMMIT_SHA:-}}"

if [ -z "$before" ] || [ -z "$after" ]; then
  echo "Missing commit range, building CUDA runtime base." >&2
  exit 0
fi

case "$before" in
  0000000000000000000000000000000000000000)
    echo "Initial commit range, building CUDA runtime base." >&2
    exit 0
    ;;
esac

if python3 - "$before" "$after" <<'PY'
import re
import subprocess
import sys

before, after = sys.argv[1:3]


def read_file(revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout if result.returncode == 0 else ""


def normalized_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def extract_heredoc_containing(text: str, keyword: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "python3 - <<'PY'" not in line:
            continue
        collected = [line.rstrip()]
        for following_line in lines[index + 1 :]:
            collected.append(following_line.rstrip())
            if following_line.strip() == "PY":
                block = "\n".join(collected)
                if keyword in block:
                    return normalized_lines(block)
                break
    return ""


def extract_pyproject_runtime(text: str) -> str:
    section = None
    dependency_lines = []
    extras_lines = []
    extras_dependency_names = set()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if section == "[tool.poetry.extras]":
            name = stripped.split("=", 1)[0].strip()
            if name in {"vllm", "all"}:
                extras_lines.append(stripped)
                extras_dependency_names.update(
                    re.findall(r'"([^"]+)"', stripped)
                )

    section = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue
        if section != "[tool.poetry.dependencies]" or "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name == "python":
            dependency_lines.append(stripped)
        elif "optional = true" not in stripped or name in extras_dependency_names:
            dependency_lines.append(stripped)

    return "\n".join(dependency_lines + extras_lines)


def extract_poetry_lock_runtime(text: str) -> str:
    tracked_names = {
        "bitsandbytes",
        "flashinfer-cubin",
        "flashinfer-python",
        "gguf",
        "mistral-common",
        "mistral_common",
        "openai",
        "openai-harmony",
        "ray",
        "timm",
        "tokenizers",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "triton",
        "vllm",
        "vllm-omni",
        "xformers",
    }
    blocks = re.split(r"(?=^\[\[package\]\])", text, flags=re.MULTILINE)
    selected = []
    for block in blocks:
        name_match = re.search(r'^name = "([^"]+)"', block, flags=re.MULTILINE)
        if not name_match:
            continue
        package_name = name_match.group(1)
        if package_name in tracked_names or package_name.startswith("nvidia-"):
            selected.append(normalized_lines(block))
    return "\n\n".join(selected)


def extract_pipeline_runtime(text: str) -> str:
    selected = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(
            token in stripped
            for token in [
                "docker-cuda-base",
                "should-build-cuda-base.sh",
                "cuda-base.dockerfile",
                "wod/windstackbase",
                "cuda12.",
            ]
        ):
            selected.append(stripped)
    return "\n".join(selected)


def runtime_signature(revision: str) -> dict[str, str]:
    build_script = read_file(revision, ".beagle/build.sh")
    return {
        ".beagle/cuda-runtime.env": normalized_lines(
            read_file(revision, ".beagle/cuda-runtime.env")
        ),
        ".beagle/cuda-base.dockerfile": normalized_lines(
            read_file(revision, ".beagle/cuda-base.dockerfile")
        ),
        ".beagle/build.sh:requirements-vllm": extract_heredoc_containing(
            build_script, "requirements-vllm.txt"
        ),
        ".beagle/should-build-cuda-base.sh": normalized_lines(
            read_file(revision, ".beagle/should-build-cuda-base.sh")
        ),
        ".beagle.yml:cuda-base": extract_pipeline_runtime(
            read_file(revision, ".beagle.yml")
        ),
        "pyproject.toml:runtime-dependencies": extract_pyproject_runtime(
            read_file(revision, "pyproject.toml")
        ),
        "poetry.lock:runtime-heavy-packages": extract_poetry_lock_runtime(
            read_file(revision, "poetry.lock")
        ),
    }


before_signature = runtime_signature(before)
after_signature = runtime_signature(after)
changed_sections = [
    name
    for name in sorted(before_signature.keys() | after_signature.keys())
    if before_signature.get(name) != after_signature.get(name)
]

if changed_sections:
    print("CUDA runtime base signature changed:", file=sys.stderr)
    for section in changed_sections:
        print(f"- {section}", file=sys.stderr)
    sys.exit(0)

print("CUDA runtime base signature unchanged.", file=sys.stderr)
sys.exit(1)
PY
then
  exit 0
else
  status=$?
  if [ "$status" = "1" ]; then
    exit 1
  fi
  echo "Failed to compare CUDA runtime base inputs, building CUDA runtime base." >&2
  exit 0
fi
