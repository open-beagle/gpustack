#!/bin/bash

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.0.3}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}"
LLAMA_BOX_REPOSITORY="https://github.com/gpustack/llama-box.git"
LOCK_FILE=/workspace/.beagle/llama-box.lock
DIST_DIR=/workspace/dist
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

. "${LOCK_FILE}"
test "${LLAMA_BOX_PACKAGE}" = "llama-box-cuda-${CUDA_VERSION}-${LLAMA_BOX_VERSION}-linux-x64"

sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
    -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' \
    /etc/apt/sources.list.d/ubuntu.sources

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    pkg-config
rm -rf /var/lib/apt/lists/*

mkdir -p "${DIST_DIR}"
git clone --branch "${LLAMA_BOX_VERSION}" --depth 1 \
    --recurse-submodules --shallow-submodules \
    "${LLAMA_BOX_REPOSITORY}" "${BUILD_ROOT}/llama-box"
test "$(git -C "${BUILD_ROOT}/llama-box" rev-parse HEAD)" = "${LLAMA_BOX_COMMIT}"

git config --global --add safe.directory '*'
export LLAMA_BOX_BUILD_VERSION="${LLAMA_BOX_VERSION}"
cmake -S "${BUILD_ROOT}/llama-box" -B "${BUILD_ROOT}/llama-box-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH="\$ORIGIN" \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DBOX_PATCH_CI=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_RPC=ON
cmake --build "${BUILD_ROOT}/llama-box-build" \
    --target llama-box --config Release --parallel "$(nproc)"

mkdir -p "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}"
find "${BUILD_ROOT}/llama-box-build/bin" -maxdepth 1 \
    \( -name llama-box -o -name 'lib*.so*' \) \
    -exec cp -a {} "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}/" \;
test -x "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}/llama-box"

if find "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}" -maxdepth 1 -type f \
    \( -name llama-box -o -name 'lib*.so*' \) -exec ldd {} \; \
    | grep -Eq 'lib(cudart|cublas).*\.so\.12'; then
    echo "llama-box unexpectedly links against CUDA 12" >&2
    exit 1
fi

tar -czf "${DIST_DIR}/llama-box.tar.gz" \
    -C "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}" .
chmod a+r "${DIST_DIR}/llama-box.tar.gz"
cat "${LOCK_FILE}"
ls -lh "${DIST_DIR}/llama-box.tar.gz"
