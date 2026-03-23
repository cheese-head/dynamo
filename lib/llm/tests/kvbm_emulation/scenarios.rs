// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioFixture {
    pub request_id: String,
    pub generation: u64,
    pub tp_bindings: usize,
    pub immediate_ops: usize,
    pub block_count: usize,
    pub block_size: usize,
}

pub fn load_fixture(name: &str) -> ScenarioFixture {
    let fixture = match name {
        "fixture-basic-offload-onboard" => include_str!("fixtures/basic_offload_onboard.yaml"),
        _ => panic!("unknown fixture: {name}"),
    };

    serde_yaml::from_str(fixture).expect("fixture must parse")
}
