#!/bin/bash

git config --global --add safe.directory "$PWD"

set -ex

VERSION="${VERSION:-v0.7}"
UI_VERSION="${UI_VERSION:-v0.7}"

PIP_INSTALL_ARGS=()
if [ -n "${PYPI_MIRROR:-}" ]; then
  PIP_INSTALL_ARGS+=("-i" "$PYPI_MIRROR")
fi
if [ -n "${PYPI_EXTRA_INDEX_URL:-}" ]; then
  PIP_INSTALL_ARGS+=("--extra-index-url" "$PYPI_EXTRA_INDEX_URL")
fi
if [ -n "${PYPI_HOST:-}" ]; then
  PIP_INSTALL_ARGS+=("--trusted-host" "$PYPI_HOST")
fi

# 检查并安装 poetry
if ! command -v poetry &> /dev/null; then
  echo "Poetry not found, installing..."
  python3 -m pip install "${PIP_INSTALL_ARGS[@]}" poetry
  export PATH="$HOME/.local/bin:$PATH"
fi

POETRY_CMD=(poetry)
if ! command -v poetry &> /dev/null; then
  POETRY_CMD=(python3 -m poetry)
fi

# 清理旧的构建产物
rm -rf "$PWD/dist"

# UI 产物目录（打包进 wheel 的目录）
UI_PATH="$PWD/gpustack/ui"

# 优先级：本地 gpustack/ui > ../gpustack-ui/dist（本地构建） > 远程下载
if [ -f "$UI_PATH/index.html" ]; then
  echo "UI build artifacts already exist at $UI_PATH, skipping download."
elif [ -f "$PWD/../gpustack-ui/dist/index.html" ]; then
  echo "Found local UI build at ../gpustack-ui/dist, copying to $UI_PATH..."
  rm -rf "$UI_PATH"
  mkdir -p "$UI_PATH"
  cp -a "$PWD/../gpustack-ui/dist/." "$UI_PATH"
else
  echo "Downloading UI assets for ${UI_VERSION}..."
  rm -rf "$UI_PATH"
  mkdir -p "$UI_PATH"

  UI_TMP="$PWD/.tmp/ui-download"
  rm -rf "$UI_TMP"
  mkdir -p "$UI_TMP"

  if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL \
    "https://cache.ali.wodcloud.com/vscode/gpustack/gpustack-ui-${UI_VERSION}.tar.gz" | \
    tar -xzf - --directory "$UI_TMP" 2>/dev/null; then
    echo "Failed to download UI assets for ${UI_VERSION}."
    exit 1
  fi

  cp -a "$UI_TMP/dist/." "$UI_PATH"
  rm -rf "$UI_TMP"
fi

# 验证 UI 目录
if [ ! -f "$UI_PATH/index.html" ]; then
  echo "ERROR: UI directory not found or incomplete!"
  exit 1
fi

echo "UI directory contents:"
ls -la "$UI_PATH/"

# 设置版本号
VERSION_FILE="$PWD/gpustack/__init__.py"
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "HEAD")
GIT_COMMIT_SHORT="${GIT_COMMIT:0:7}"

python3 -c "
import re
with open('$VERSION_FILE', 'r') as f:
    content = f.read()
content = re.sub(r\"__version__ = .*\", \"__version__ = '${VERSION}'\", content)
content = re.sub(r\"__git_commit__ = .*\", \"__git_commit__ = '${GIT_COMMIT_SHORT}'\", content)
with open('$VERSION_FILE', 'w') as f:
    f.write(content)
"

# 更新 pyproject.toml 中的版本号
"${POETRY_CMD[@]}" version "${VERSION}"

# 使用 poetry build 构建 wheel 和 sdist
echo "Building with poetry..."
"${POETRY_CMD[@]}" build

# 从 wheel 元数据导出运行时依赖清单，供 CUDA 基础镜像和 CoreX 镜像安装稳定依赖。
python3 - <<'PY'
import email
import glob
import os
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement

wheel_files = glob.glob("dist/*.whl")
if len(wheel_files) != 1:
    raise SystemExit(f"Expected exactly one wheel in dist, got {wheel_files}")

env = default_environment()
requirements = []

with zipfile.ZipFile(wheel_files[0]) as wheel:
    metadata_name = next(
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = email.message_from_bytes(wheel.read(metadata_name))

for value in metadata.get_all("Requires-Dist") or []:
    req = Requirement(value)
    marker = req.marker
    if marker is not None:
        include = marker.evaluate({**env, "extra": ""}) or marker.evaluate(
            {**env, "extra": "vllm"}
        )
        if not include:
            continue
        req.marker = None
    requirements.append(str(req))

requirements.append("argcomplete>=1.9.4")
requirements = sorted(set(requirements), key=str.lower)

requirements_path = os.path.join("dist", "requirements-vllm.txt")
with open(requirements_path, "w", encoding="utf-8") as f:
    f.write("\n".join(requirements))
    f.write("\n")

print(f"Wrote {requirements_path} with {len(requirements)} requirements.")
PY

# 还原版本文件
git checkout -- "$VERSION_FILE"
git checkout -- "$PWD/pyproject.toml"

echo "Build completed successfully!"
