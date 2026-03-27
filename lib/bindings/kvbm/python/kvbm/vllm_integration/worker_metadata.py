# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM worker metadata for vLLM v0.18.0 worker->scheduler feedback channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorWorkerMetadata,
    )
except ImportError:

    class KVConnectorWorkerMetadata:
        def aggregate(self, other):
            raise NotImplementedError


class KvbmWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler metadata for KVBM.

    Carries per-request transfer completion and failure information
    from workers back to the scheduler-side connector each step.
    """

    def __init__(self):
        self.completed_onboard_ops: dict[str, set[str]] = {}
        self.completed_offload_ops: dict[str, set[str]] = {}
        self.failed_ops: dict[str, set[str]] = {}
        self.worker_id: int = 0

    @property
    def is_empty(self) -> bool:
        return (
            not self.completed_onboard_ops
            and not self.completed_offload_ops
            and not self.failed_ops
        )

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KvbmWorkerMetadata":
        """TP consensus: intersection for completions, union for failures."""
        assert isinstance(other, KvbmWorkerMetadata)
        result = KvbmWorkerMetadata()

        for req_id in self.completed_onboard_ops.keys() & other.completed_onboard_ops.keys():
            result.completed_onboard_ops[req_id] = (
                self.completed_onboard_ops[req_id] & other.completed_onboard_ops[req_id]
            )

        for req_id in self.completed_offload_ops.keys() & other.completed_offload_ops.keys():
            result.completed_offload_ops[req_id] = (
                self.completed_offload_ops[req_id] & other.completed_offload_ops[req_id]
            )

        all_fail_keys = self.failed_ops.keys() | other.failed_ops.keys()
        for req_id in all_fail_keys:
            result.failed_ops[req_id] = (
                self.failed_ops.get(req_id, set()) | other.failed_ops.get(req_id, set())
            )

        return result

    def completed_ops(self) -> dict[str, set[str]]:
        """All completed ops (onboard + offload) merged."""
        merged: dict[str, set[str]] = {}
        for req_id, ops in self.completed_onboard_ops.items():
            merged.setdefault(req_id, set()).update(ops)
        for req_id, ops in self.completed_offload_ops.items():
            merged.setdefault(req_id, set()).update(ops)
        return merged
