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
  echo "Missing commit range, skipping CUDA runtime base." >&2
  exit 1
fi

case "$before" in
  0000000000000000000000000000000000000000)
    echo "Initial commit range, building CUDA runtime base." >&2
    exit 0
    ;;
esac

ensure_commit_available() {
  revision="$1"

  if git cat-file -e "$revision^{commit}" 2>/dev/null; then
    return 0
  fi

  echo "Commit $revision is missing from local git history, fetching it." >&2

  branch="${DRONE_BRANCH:-}"
  if [ -z "$branch" ]; then
    branch="$(git branch --show-current 2>/dev/null || true)"
  fi

  if [ -n "$branch" ]; then
    git fetch --no-tags --deepen=100 origin "$branch" >/dev/null 2>&1 || true
  fi

  if git cat-file -e "$revision^{commit}" 2>/dev/null; then
    return 0
  fi

  git fetch --no-tags --depth=1 origin "$revision" >/dev/null 2>&1 || true
  git cat-file -e "$revision^{commit}" 2>/dev/null
}

if ! ensure_commit_available "$before" || ! ensure_commit_available "$after"; then
  echo "Failed to fetch commit range, skipping CUDA runtime base." >&2
  exit 1
fi

normalize_revision_file() {
  revision="$1"
  path="$2"

  git show "$revision:$path" 2>/dev/null | awk '
    {
      line = $0
      gsub(/^[ \t]+|[ \t]+$/, "", line)
      if (line == "" || line ~ /^#/) {
        next
      }
      print line
    }
  '
}

extract_build_requirements_export() {
  revision="$1"

  git show "$revision:.beagle/build.sh" 2>/dev/null | awk '
    function flush_block() {
      if (in_block && block ~ /requirements-vllm.txt/) {
        count = split(block, lines, "\n")
        for (i = 1; i <= count; i++) {
          line = lines[i]
          gsub(/^[ \t]+|[ \t]+$/, "", line)
          if (line != "" && line !~ /^#/) {
            print line
          }
        }
      }
      in_block = 0
      block = ""
    }

    index($0, "python3 - <<") && index($0, "PY") {
      flush_block()
      in_block = 1
    }

    in_block {
      block = block $0 "\n"
      if ($0 ~ /^[ \t]*PY[ \t]*$/) {
        flush_block()
      }
    }

    END {
      flush_block()
    }
  '
}

extract_pyproject_runtime() {
  revision="$1"

  git show "$revision:pyproject.toml" 2>/dev/null | awk '
    {
      line = $0
      gsub(/^[ \t]+|[ \t]+$/, "", line)
      if (line == "" || line ~ /^#/) {
        next
      }
      if (line ~ /^\[/) {
        section = line
        next
      }
      if (section == "[tool.poetry.dependencies]") {
        name = line
        sub(/[ \t]*=.*/, "", name)
        if (name ~ /^(python|openai|ray|vllm|vllm-omni|mistral_common|transformers|bitsandbytes|flashinfer-cubin|flashinfer-python|gguf|mistral-common|openai-harmony|timm|tokenizers|torch|torchaudio|torchvision|triton|xformers)$/) {
          print line
        }
      } else if (section == "[tool.poetry.extras]") {
        name = line
        sub(/[ \t]*=.*/, "", name)
        if (name == "vllm" || name == "all") {
          print line
        }
      }
    }
  '
}

extract_poetry_lock_runtime() {
  revision="$1"

  git show "$revision:poetry.lock" 2>/dev/null | awk '
    function normalize_and_print(block_text) {
      count = split(block_text, lines, "\n")
      for (i = 1; i <= count; i++) {
        line = lines[i]
        gsub(/^[ \t]+|[ \t]+$/, "", line)
        if (line != "" && line !~ /^#/) {
          print line
        }
      }
      print ""
    }

    function tracked(package_name) {
      return package_name ~ /^nvidia-/ || package_name ~ /^(bitsandbytes|flashinfer-cubin|flashinfer-python|gguf|mistral-common|mistral_common|openai|openai-harmony|ray|timm|tokenizers|torch|torchaudio|torchvision|transformers|triton|vllm|vllm-omni|xformers)$/
    }

    /^\[\[package\]\]/ {
      if (package_name != "" && tracked(package_name)) {
        normalize_and_print(block)
      }
      block = $0 "\n"
      package_name = ""
      next
    }

    {
      block = block $0 "\n"
      if ($0 ~ /^name = "/) {
        package_name = $0
        sub(/^name = "/, "", package_name)
        sub(/".*/, "", package_name)
      }
    }

    END {
      if (package_name != "" && tracked(package_name)) {
        normalize_and_print(block)
      }
    }
  '
}

extract_pipeline_runtime() {
  revision="$1"

  git show "$revision:.beagle.yml" 2>/dev/null | awk '
    {
      line = $0
      gsub(/^[ \t]+|[ \t]+$/, "", line)
      if (index(line, "docker-cuda-base") || index(line, "should-build-cuda-base.sh") || index(line, "cuda-base.dockerfile") || index(line, "wod/windstackbase") || index(line, "cuda12.")) {
        print line
      }
    }
  '
}

write_runtime_signature() {
  revision="$1"
  output="$2"

  {
    echo "[.beagle/cuda-runtime.env]"
    normalize_revision_file "$revision" ".beagle/cuda-runtime.env"
    echo "[.beagle/cuda-base.dockerfile]"
    normalize_revision_file "$revision" ".beagle/cuda-base.dockerfile"
    echo "[.beagle/build.sh:requirements-vllm]"
    extract_build_requirements_export "$revision"
    echo "[.beagle.yml:cuda-base]"
    extract_pipeline_runtime "$revision"
    echo "[pyproject.toml:runtime-dependencies]"
    extract_pyproject_runtime "$revision"
    echo "[poetry.lock:runtime-heavy-packages]"
    extract_poetry_lock_runtime "$revision"
  } > "$output"
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

before_signature="$tmp_dir/before.signature"
after_signature="$tmp_dir/after.signature"

if ! write_runtime_signature "$before" "$before_signature" || ! write_runtime_signature "$after" "$after_signature"; then
  echo "Failed to compare CUDA runtime base inputs, skipping CUDA runtime base." >&2
  exit 1
fi

if cmp -s "$before_signature" "$after_signature"; then
  echo "CUDA runtime base signature unchanged." >&2
  exit 1
fi

echo "CUDA runtime base signature changed:" >&2
diff -u "$before_signature" "$after_signature" >&2 || true
exit 0
