#!/bin/bash

git config --global --add safe.directory "$PWD"

# 配置 PyPI 镜像源：优先阿里云内网，不可达则回退公网
if curl -s --connect-timeout 2 http://mirrors.cloud.aliyuncs.com/pypi/simple/ > /dev/null 2>&1; then
  PYPI_MIRROR="http://mirrors.cloud.aliyuncs.com/pypi/simple/"
  PYPI_HOST="mirrors.cloud.aliyuncs.com"
else
  PYPI_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
  PYPI_HOST="mirrors.aliyun.com"
fi
pip config set global.index-url "$PYPI_MIRROR"
pip config set global.trusted-host "$PYPI_HOST"

set -ex

VERSION="${VERSION:-v0.7.1}"

# 检查并安装 poetry
if ! command -v poetry &> /dev/null; then
  echo "Poetry not found, installing..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi

# 清理旧的构建产物
rm -rf "$PWD/dist"

# UI 产物目录（打包进 wheel 的目录）
UI_PATH="$PWD/gpustack/ui"

# 优先级：本地 gpustack/ui > ui/dist（本地构建） > 远程下载
if [ -f "$UI_PATH/index.html" ]; then
  echo "UI build artifacts already exist at $UI_PATH, skipping download."
elif [ -f "$PWD/ui/dist/index.html" ]; then
  echo "Found local UI build at ui/dist, copying to $UI_PATH..."
  rm -rf "$UI_PATH"
  mkdir -p "$UI_PATH"
  cp -a "$PWD/ui/dist/." "$UI_PATH"
else
  echo "Downloading UI assets for ${VERSION}..."
  rm -rf "$UI_PATH"
  mkdir -p "$UI_PATH"

  UI_TMP="$PWD/.tmp/ui-download"
  rm -rf "$UI_TMP"
  mkdir -p "$UI_TMP"

  if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL \
    "https://cache.ali.wodcloud.com/vscode/gpustack/gpustack-ui-${VERSION}.tar.gz" | \
    tar -xzf - --directory "$UI_TMP" 2>/dev/null; then
    echo "Failed to download UI assets for ${VERSION}."
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
poetry version "${VERSION}"

# 使用 poetry build 构建 wheel 和 sdist
echo "Building with poetry..."
poetry build

# # 验证 wheel 包含 UI
# echo "Checking wheel contents for UI files..."
# WHEEL_FILE=$(ls -t dist/*.whl | head -1)
# if [ -f "$WHEEL_FILE" ]; then
#   echo "Wheel file: $WHEEL_FILE"
#   unzip -l "$WHEEL_FILE" | grep -E "(ui/|index.html)" || {
#     echo "WARNING: UI files not found in wheel!"
#     echo "Full wheel contents:"
#     unzip -l "$WHEEL_FILE" | head -100
#   }
# else
#   echo "ERROR: Wheel file not found!"
#   exit 1
# fi

# 还原版本文件
git checkout -- "$VERSION_FILE"

echo "Build completed successfully!"
