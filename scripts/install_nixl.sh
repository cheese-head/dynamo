#!/bin/bash
#
# Build and install NIXL 0.9.0 from source inside a container.
#
# Assumptions:
#   - Running as root
#   - UCX is pre-installed at /usr/local/ucx
#   - Architecture is x86_64

set -euo pipefail

NIXL_VERSION="${NIXL_VERSION:-0.9.0}"
NIXL_REPO="${NIXL_REPO:-https://github.com/ai-dynamo/nixl.git}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/nvidia/nvda_nixl}"
UCX_PATH="${UCX_PATH:-/usr/local/ucx}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12-9}"
BUILD_DIR="${BUILD_DIR:-/tmp/nixl-build}"

ARCH_NAME="x86_64-linux-gnu"

echo "=== Installing system dependencies ==="
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" \
    libaio-dev \
    libaio1t64 \
    pkg-config \
    liburing-dev

echo "=== Installing Python build tools ==="
pip install --no-cache-dir meson ninja pybind11 tomlkit

echo "=== Setting up environment ==="
export PATH="${CUDA_PATH}/bin:$PATH"
export CPLUS_INCLUDE_PATH="$(python3 -c 'import pybind11; print(pybind11.get_include())'):${CPLUS_INCLUDE_PATH:-}"

echo "=== Cloning NIXL ${NIXL_VERSION} ==="
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
git clone --depth 1 -b "${NIXL_VERSION}" "${NIXL_REPO}" "${BUILD_DIR}/nixl"
cd "${BUILD_DIR}/nixl"

echo "=== Configuring with Meson ==="
meson setup builddir \
    --prefix="${INSTALL_PREFIX}" \
    --buildtype=release \
    -Dcudapath_lib="${CUDA_PATH}/lib64" \
    -Dcudapath_inc="${CUDA_PATH}/include" \
    -Ducx_path="${UCX_PATH}"

echo "=== Building and installing ==="
ninja -C builddir
ninja -C builddir install
ldconfig

echo "=== Cleaning up build directory ==="
rm -rf "${BUILD_DIR}"

echo "=== NIXL ${NIXL_VERSION} installed to ${INSTALL_PREFIX} ==="
echo ""
echo "Add the following to your environment (e.g. ~/.bashrc):"
echo "  export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib/${ARCH_NAME}:${INSTALL_PREFIX}/lib64:\$LD_LIBRARY_PATH"

