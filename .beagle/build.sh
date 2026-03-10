#!/bin/bash

git config --global --add safe.directory "$PWD"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

set -ex

VENV_DIR="$PWD/.venv"
VERSION="${VERSION:-v0.7.1}"

# 清理旧的构建产物
rm -rf "$PWD/dist"

# 检查 venv 是否与当前 Python 版本匹配
CURRENT_PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
if [ -e "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON_VERSION=$("$VENV_DIR/bin/python" --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2 || echo "")
  if [ "$CURRENT_PYTHON_VERSION" != "$VENV_PYTHON_VERSION" ]; then
    echo "Python version mismatch (current: $CURRENT_PYTHON_VERSION, venv: $VENV_PYTHON_VERSION), recreating venv..."
    rm -rf "$VENV_DIR"
  fi
fi

if ! [ -e "$VENV_DIR/bin/activate" ]; then
  python3 -m venv "$VENV_DIR"
fi

# 使用虚拟环境中的 pip 和 poetry
export PATH="$VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$VENV_DIR"

# 安装 poetry（如果不存在）
if ! [ -e "$VENV_DIR/bin/poetry" ]; then
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install poetry==1.8.3
fi

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

"$VENV_DIR/bin/pip" install msgpack
"$VENV_DIR/bin/poetry" version "${VERSION}"

# 验证 UI 目录
if [ ! -f "$UI_PATH/index.html" ]; then
  echo "ERROR: UI directory not found or incomplete!"
  exit 1
fi

echo "UI directory contents:"
ls -la "$UI_PATH/"

# 使用 poetry 构建
"$VENV_DIR/bin/poetry" build

# 验证 wheel 包含 UI
echo "Checking wheel contents..."
unzip -l dist/*.whl | head -50

# 还原版本文件
git checkout -- "$VERSION_FILE"
