#!/bin/bash

set -euo pipefail

CUDA_VERSION="${CUDA_VERSION:-13.0.3}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;90;100;120}"
LLAMA_CPP_REPOSITORY="https://github.com/ggml-org/llama.cpp.git"

DIST_DIR=/workspace/dist
LOCK_FILE=/workspace/.beagle/llama.cpp.lock
BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

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

LLAMA_CPP_REF="$({
    git ls-remote --tags --refs --sort=-version:refname \
        "${LLAMA_CPP_REPOSITORY}" 'refs/tags/b*' || true
} | head -n 1)"
test -n "${LLAMA_CPP_REF}"
LLAMA_CPP_VERSION="${LLAMA_CPP_REF##*refs/tags/}"
[[ "${LLAMA_CPP_VERSION}" =~ ^b[0-9]+$ ]]

CUDA_STUB_DIR=/usr/local/cuda/lib64/stubs
if [ ! -e "${CUDA_STUB_DIR}/libcuda.so.1" ] \
    && [ -e "${CUDA_STUB_DIR}/libcuda.so" ]; then
    ln -s "${CUDA_STUB_DIR}/libcuda.so" "${CUDA_STUB_DIR}/libcuda.so.1"
fi
export LD_LIBRARY_PATH="${CUDA_STUB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_STUB_DIR}:${LIBRARY_PATH:-}"

git clone --branch "${LLAMA_CPP_VERSION}" --depth 1 \
    "${LLAMA_CPP_REPOSITORY}" "${BUILD_ROOT}/llama.cpp"
LLAMA_CPP_COMMIT="$(git -C "${BUILD_ROOT}/llama.cpp" rev-parse HEAD)"
cmake -S "${BUILD_ROOT}/llama.cpp" -B "${BUILD_ROOT}/llama.cpp-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH="\$ORIGIN" \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DGGML_CUDA=ON \
    -DGGML_RPC=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"
cmake --build "${BUILD_ROOT}/llama.cpp-build" --config Release --parallel "$(nproc)"

LLAMA_CPP_PACKAGE="llama-cpp-cuda-${CUDA_VERSION}-${LLAMA_CPP_VERSION}-linux-x64"
mkdir -p "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}"
for file in llama-server llama-cli ggml-rpc-server lib*.so*; do
    find "${BUILD_ROOT}/llama.cpp-build/bin" -maxdepth 1 -name "${file}" \
        -exec cp -a {} "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}/" \;
done
test -x "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}/llama-server"
test -x "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}/ggml-rpc-server"
tar -czf "${DIST_DIR}/llama.cpp.tar.gz" \
    -C "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}" .

if ldd "${BUILD_ROOT}/${LLAMA_CPP_PACKAGE}/llama-server" \
    | grep -Eq 'lib(cudart|cublas).*\.so\.12'; then
    echo "llama.cpp unexpectedly links against CUDA 12" >&2
    exit 1
fi

chmod a+r "${DIST_DIR}/llama.cpp.tar.gz"

LOCK_TMP="$(mktemp "${LOCK_FILE}.XXXXXX")"
printf '%s\n' \
    "LLAMA_CPP_VERSION=${LLAMA_CPP_VERSION}" \
    "LLAMA_CPP_COMMIT=${LLAMA_CPP_COMMIT}" \
    "CUDA_VERSION=${CUDA_VERSION}" \
    "LLAMA_CPP_PACKAGE=${LLAMA_CPP_PACKAGE}" \
    > "${LOCK_TMP}"
chmod a+r "${LOCK_TMP}"
mv -f "${LOCK_TMP}" "${LOCK_FILE}"

cat "${LOCK_FILE}"
ls -lh "${DIST_DIR}/llama.cpp.tar.gz"
