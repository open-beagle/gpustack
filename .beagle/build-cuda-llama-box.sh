#!/bin/bash

set -euo pipefail

validate_package() {
    local archive="$1"
    local validation_root
    local entry
    local link
    local target
    local file
    local dynamic_output

    test -f "${archive}"
    validation_root="$(mktemp -d)"
    trap 'rm -rf "${validation_root}"' RETURN

    while IFS= read -r entry; do
        case "${entry}" in
            /*|../*|*/../*|*/..)
                echo "llama-box package contains unsafe path: ${entry}" >&2
                return 1
                ;;
        esac
    done < <(tar -tzf "${archive}")

    if tar -tvzf "${archive}" | awk 'substr($1, 1, 1) !~ /[-dl]/ { exit 1 }'; then
        :
    else
        echo "llama-box package contains unsupported special files" >&2
        return 1
    fi

    tar -xzf "${archive}" -C "${validation_root}"
    test -x "${validation_root}/llama-box" \
        && test ! -L "${validation_root}/llama-box" || {
        echo "llama-box package is missing executable llama-box" >&2
        return 1
    }

    while IFS= read -r entry; do
        case "$(basename "${entry}")" in
            llama-box|lib*.so*) ;;
            *)
                echo "llama-box package contains unexpected entry: ${entry}" >&2
                return 1
                ;;
        esac
        if [ ! -f "${entry}" ] && [ ! -L "${entry}" ]; then
            echo "llama-box package contains unexpected entry type: ${entry}" >&2
            return 1
        fi
    done < <(find "${validation_root}" -mindepth 1 -maxdepth 1 -print)

    while IFS= read -r link; do
        target="$(readlink "${link}")"
        case "${target}" in
            /*|../*|*/../*|*/..)
                echo "llama-box package contains unsafe symlink: ${link} -> ${target}" >&2
                return 1
                ;;
        esac
        target="$(readlink -f "${link}")"
        case "${target}" in
            "${validation_root}"/*) ;;
            *)
                echo "llama-box package symlink escapes package root: ${link}" >&2
                return 1
                ;;
        esac
    done < <(find "${validation_root}" -type l -print)

    while IFS= read -r file; do
        if ! dynamic_output="$(readelf -d "${file}" 2>&1)"; then
            echo "failed to inspect llama-box package ELF ${file}: ${dynamic_output}" >&2
            return 1
        fi
        if grep -Eq 'NEEDED.*lib(cudart|cublas).*\.so\.12' <<<"${dynamic_output}"; then
            echo "llama-box package has invalid CUDA ABI dependencies: ${file}" >&2
            echo "${dynamic_output}" >&2
            return 1
        fi
    done < <(
        find "${validation_root}" -maxdepth 1 -type f \
            \( -name llama-box -o -name 'lib*.so*' \) -print
    )

    if ! readelf -d "${validation_root}/llama-box" \
        | grep -Eq '\((RPATH|RUNPATH)\).*[\$]ORIGIN'; then
        echo "llama-box package executable is missing an ORIGIN RPATH/RUNPATH" >&2
        return 1
    fi

    echo "Validated llama-box package: ${archive}"
}

validate_dynamic_linkage() {
    local package_root="$1"
    local cuda_stub_dir=/usr/local/cuda/lib64/stubs
    local file
    local ldd_output

    if [ ! -e "${cuda_stub_dir}/libcuda.so.1" ] \
        && [ -e "${cuda_stub_dir}/libcuda.so" ]; then
        ln -s "${cuda_stub_dir}/libcuda.so" "${cuda_stub_dir}/libcuda.so.1"
    fi

    while IFS= read -r file; do
        if ! ldd_output="$(
            LD_LIBRARY_PATH="${package_root}:${cuda_stub_dir}:${LD_LIBRARY_PATH:-}" \
                ldd "${file}" 2>&1
        )"; then
            echo "failed to resolve llama-box ELF dependencies: ${file}" >&2
            echo "${ldd_output}" >&2
            return 1
        fi
        if grep -Eq 'not found|lib(cudart|cublas).*\.so\.12' <<<"${ldd_output}"; then
            echo "llama-box package has invalid dynamic dependencies: ${file}" >&2
            echo "${ldd_output}" >&2
            return 1
        fi
    done < <(
        find "${package_root}" -maxdepth 1 -type f \
            \( -name llama-box -o -name 'lib*.so*' \) -print
    )
}

if [ "${1:-}" = "--validate-package" ]; then
    test -n "${2:-}"
    validate_package "$2"
    exit 0
fi

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
validate_dynamic_linkage "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}"

tar -czf "${DIST_DIR}/llama-box.tar.gz" \
    -C "${BUILD_ROOT}/${LLAMA_BOX_PACKAGE}" .
validate_package "${DIST_DIR}/llama-box.tar.gz"
chmod a+r "${DIST_DIR}/llama-box.tar.gz"
cat "${LOCK_FILE}"
sha256sum "${DIST_DIR}/llama-box.tar.gz"
ls -lh "${DIST_DIR}/llama-box.tar.gz"
