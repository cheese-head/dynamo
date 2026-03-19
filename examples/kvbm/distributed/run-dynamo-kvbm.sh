#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "=== Rebuilding KVBM wheel from workspace source ==="
cd /workspace/lib/bindings/kvbm
maturin build --profile dev --out /workspace/dist
uv pip install --upgrade --force-reinstall --no-deps /workspace/dist/*.whl
echo "=== KVBM wheel rebuilt and installed ==="

cd /workspace/examples/backends/vllm

prefix_flag=()
if [[ "${ENABLE_PREFIX_CACHING:-false}" == "false" || "${ENABLE_PREFIX_CACHING:-false}" == "0" ]]; then
    prefix_flag+=(--no-enable-prefix-caching)
    echo "=== Prefix caching: DISABLED ==="
else
    prefix_flag+=(--enable-prefix-caching)
    echo "=== Prefix caching: ENABLED ==="
fi

python3 -m dynamo.frontend --router-mode round-robin --http-port 8000 &

exec python3 -m dynamo.vllm \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --connector kvbm \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${prefix_flag[@]}" \
    ${VLLM_EXTRA_ARGS:-}
