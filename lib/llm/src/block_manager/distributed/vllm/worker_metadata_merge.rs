// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust mirror of `kvbm/vllm_integration/worker_metadata.py` `KvbmWorkerMetadata.aggregate`
//! (TP consensus: intersection for completions, union for failures).
//! Keep logic aligned with Python when changing either side.

use std::collections::{HashMap, HashSet};

/// Test fixture: op ids as strings (UUID strings in production).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct KvbmWorkerMetadataFixture {
    pub completed_onboard_ops: HashMap<String, HashSet<String>>,
    pub completed_offload_ops: HashMap<String, HashSet<String>>,
    pub failed_ops: HashMap<String, HashSet<String>>,
}

pub(crate) fn aggregate_tp(
    a: &KvbmWorkerMetadataFixture,
    b: &KvbmWorkerMetadataFixture,
) -> KvbmWorkerMetadataFixture {
    let mut result = KvbmWorkerMetadataFixture::default();

    for req_id in a.completed_onboard_ops.keys() {
        if let Some(bset) = b.completed_onboard_ops.get(req_id) {
            let inter: HashSet<String> = a.completed_onboard_ops[req_id]
                .intersection(bset)
                .cloned()
                .collect();
            result.completed_onboard_ops.insert(req_id.clone(), inter);
        }
    }

    for req_id in a.completed_offload_ops.keys() {
        if let Some(bset) = b.completed_offload_ops.get(req_id) {
            let inter: HashSet<String> = a.completed_offload_ops[req_id]
                .intersection(bset)
                .cloned()
                .collect();
            result.completed_offload_ops.insert(req_id.clone(), inter);
        }
    }

    let all_fail_keys: HashSet<_> = a.failed_ops.keys().chain(b.failed_ops.keys()).cloned().collect();
    for req_id in all_fail_keys {
        let u: HashSet<String> = a
            .failed_ops
            .get(&req_id)
            .into_iter()
            .flatten()
            .chain(b.failed_ops.get(&req_id).into_iter().flatten())
            .cloned()
            .collect();
        result.failed_ops.insert(req_id, u);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setreq(
        onboard: &[(&str, &[&str])],
        offload: &[(&str, &[&str])],
        failed: &[(&str, &[&str])],
    ) -> KvbmWorkerMetadataFixture {
        let mut f = KvbmWorkerMetadataFixture::default();
        for (r, ops) in onboard {
            f.completed_onboard_ops.insert(
                (*r).to_string(),
                ops.iter().map(|s| (*s).to_string()).collect(),
            );
        }
        for (r, ops) in offload {
            f.completed_offload_ops.insert(
                (*r).to_string(),
                ops.iter().map(|s| (*s).to_string()).collect(),
            );
        }
        for (r, ops) in failed {
            f.failed_ops
                .insert((*r).to_string(), ops.iter().map(|s| (*s).to_string()).collect());
        }
        f
    }

    #[test]
    fn tp_intersection_same_ops_keeps_recv() {
        let a = setreq(&[("r1", &["o1", "o2"])], &[], &[]);
        let b = setreq(&[("r1", &["o1", "o2"])], &[], &[]);
        let m = aggregate_tp(&a, &b);
        assert_eq!(
            m.completed_onboard_ops.get("r1").unwrap(),
            &HashSet::from(["o1".into(), "o2".into()])
        );
    }

    #[test]
    fn tp_intersection_drops_mismatched_op() {
        let a = setreq(&[("r1", &["o1", "o2"])], &[], &[]);
        let b = setreq(&[("r1", &["o2"])], &[], &[]);
        let m = aggregate_tp(&a, &b);
        assert_eq!(m.completed_onboard_ops.get("r1").unwrap(), &HashSet::from(["o2".into()]));
    }

    #[test]
    fn tp_no_common_request_yields_empty_onboard() {
        let a = setreq(&[("r1", &["o1"])], &[], &[]);
        let b = setreq(&[("r2", &["o1"])], &[], &[]);
        let m = aggregate_tp(&a, &b);
        assert!(m.completed_onboard_ops.is_empty());
    }

    #[test]
    fn failures_union_across_workers() {
        let a = setreq(&[], &[], &[("r1", &["e1"])]);
        let b = setreq(&[], &[], &[("r1", &["e2"])]);
        let m = aggregate_tp(&a, &b);
        assert_eq!(
            m.failed_ops.get("r1").unwrap(),
            &HashSet::from(["e1".into(), "e2".into()])
        );
    }
}
