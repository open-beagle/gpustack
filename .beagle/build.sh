#!/bin/bash

git config --global --add safe.directory "$PWD"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

set -ex

VENV_DIR="$PWD/.venv"
VERSION="${VERSION:-v0.7.1}"

# 清理旧的构建产物
rm -rf "$PWD/dist" "$PWD/gpustack/ui"

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

# 下载 UI
UI_PATH="$PWD/gpustack/ui"
rm -rf "$UI_PATH"
mkdir -p "$UI_PATH/tmp/ui"

echo "Downloading UI assets for ${VERSION}..."
# 尝试从自己的 S3 服务器下载
if ! curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL \
  "https://cache.ali.wodcloud.com/gpustack-ui/releases/${VERSION}.tar.gz" | \
  tar -xzf - --directory "$UI_PATH/tmp/ui" 2>/dev/null; then
  echo "Failed to download ${VERSION}, trying latest..."
  curl --retry 3 --retry-connrefused --retry-delay 3 -sSfL \
    "https://cache.ali.wodcloud.com/gpustack-ui/releases/latest.tar.gz" | \
    tar -xzf - --directory "$UI_PATH/tmp/ui"
fi
cp -a "$UI_PATH/tmp/ui/dist/." "$UI_PATH"

# 复制自定义静态文件
cp -r "$PWD/.beagle/static/"* "$UI_PATH/static/"

# 修改版权信息
UMI_JS="$(ls $UI_PATH/js/umi.*.js)"
sed -i 's/数澈软件/北京比格/g' "$UMI_JS"

# 修改帮助超链接
sed -i 's|https://docs.gpustack.ai|https://www.bc-cloud.com|g' "$UI_PATH/index.html"

# 禁用 help & lang 菜单
UMI_CSS="$(ls $UI_PATH/css/umi.*.css)"
echo 'div[data-menu-id^="rc-menu-uuid-"][data-menu-id$="-help"]{display:none;}' >> "$UMI_CSS"
echo 'div[data-menu-id^="rc-menu-uuid-"][data-menu-id$="-lang"]{display:none;}' >> "$UMI_CSS"

rm -rf "$UI_PATH/tmp"

# 复制额外静态文件
if [ -d "$PWD/static" ]; then
  cp -a "$PWD/static/." "$UI_PATH/static/"
fi

# 设置版本号
VERSION_FILE="$PWD/gpustack/__init__.py"
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "HEAD")
GIT_COMMIT_SHORT="${GIT_COMMIT:0:7}"

# 使用 Python 修改版本，避免 sed 引号问题
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

# 验证 UI 目录存在
if [ ! -d "$PWD/gpustack/ui" ] || [ ! -f "$PWD/gpustack/ui/index.html" ]; then
  echo "ERROR: UI directory not found or incomplete!"
  exit 1
fi

echo "UI directory contents:"
ls -la "$PWD/gpustack/ui/"

# 使用 poetry 构建
"$VENV_DIR/bin/poetry" build

# 验证 wheel 包含 UI
echo "Checking wheel contents..."
unzip -l dist/*.whl | head -50

# 还原版本文件
git checkout -- "$VERSION_FILE"
