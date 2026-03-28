// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use serde::Deserialize;

use crate::cli::Cli;

pub struct ResolvedLayout {
    pub num_layers: usize,
    pub outer_dim: usize,
    pub page_size: usize,
    pub inner_dim: usize,
    pub dtype_bytes: usize,
    pub source: String,
}

impl ResolvedLayout {
    pub fn block_bytes(&self) -> usize {
        self.num_layers * self.outer_dim * self.page_size * self.inner_dim * self.dtype_bytes
    }

    pub fn as_tuple(&self) -> (usize, usize, usize, usize, usize) {
        (self.num_layers, self.outer_dim, self.page_size, self.inner_dim, self.dtype_bytes)
    }
}

impl std::fmt::Display for ResolvedLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bb = self.block_bytes();
        write!(
            f,
            "{} × {} × {} × {} × {} = {} B ({:.2} MB)  [{}]",
            self.num_layers, self.outer_dim, self.page_size,
            self.inner_dim, self.dtype_bytes,
            bb, bb as f64 / 1e6, self.source,
        )
    }
}

pub fn effective_num_blocks(cli: &Cli) -> usize {
    match cli.isl {
        Some(isl) => {
            assert!(cli.page_size > 0, "--page-size must be > 0");
            (isl + cli.page_size - 1) / cli.page_size
        }
        None => cli.num_blocks,
    }
}

#[derive(Deserialize, Default, Clone, Debug)]
struct HfModelConfig {
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    hidden_size: Option<usize>,
    head_dim: Option<usize>,
    torch_dtype: Option<String>,
    text_config: Option<Box<HfModelConfig>>,
}

impl HfModelConfig {
    fn resolve(mut self) -> Self {
        if let Some(tc) = self.text_config.take() {
            let tc = *tc;
            self.num_hidden_layers = self.num_hidden_layers.or(tc.num_hidden_layers);
            self.num_attention_heads = self.num_attention_heads.or(tc.num_attention_heads);
            self.num_key_value_heads = self.num_key_value_heads.or(tc.num_key_value_heads);
            self.hidden_size = self.hidden_size.or(tc.hidden_size);
            self.head_dim = self.head_dim.or(tc.head_dim);
            self.torch_dtype = self.torch_dtype.or(tc.torch_dtype);
        }
        self
    }

    fn dtype_bytes(&self) -> usize {
        match self.torch_dtype.as_deref() {
            Some("float32") => 4,
            _ => 2,
        }
    }
}

fn download_model_config(model_id: &str) -> Result<HfModelConfig> {
    eprintln!("Downloading config.json for {model_id} ...");
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(model_id.to_string());
    let config_path = repo.get("config.json")?;
    let contents = std::fs::read_to_string(config_path)?;
    let config: HfModelConfig = serde_json::from_str(&contents)?;
    Ok(config.resolve())
}

pub fn resolve_layout(cli: &Cli) -> ResolvedLayout {
    if let Some(bb) = cli.block_bytes {
        let ps = cli.page_size;
        let dt = cli.dtype_bytes;
        assert!(ps > 0 && dt > 0, "--page-size and --dtype-bytes must be > 0");
        let inner = bb / (ps * dt);
        assert!(
            inner * ps * dt == bb,
            "--block-bytes ({bb}) must be divisible by page_size * dtype_bytes ({ps} * {dt} = {})",
            ps * dt
        );
        ResolvedLayout {
            num_layers: 1, outer_dim: 1, page_size: ps, inner_dim: inner, dtype_bytes: dt,
            source: format!("--block-bytes {bb}"),
        }
    } else if let Some(model_id) = &cli.model {
        let cfg = download_model_config(model_id)
            .unwrap_or_else(|e| panic!("Failed to download config for {model_id}: {e}"));

        let nl = cfg.num_hidden_layers
            .unwrap_or_else(|| panic!("{model_id}: missing num_hidden_layers"));
        let n_heads = cfg.num_attention_heads
            .unwrap_or_else(|| panic!("{model_id}: missing num_attention_heads"));
        let n_kv_heads = cfg.num_key_value_heads.unwrap_or(n_heads);
        let hs = cfg.hidden_size
            .unwrap_or_else(|| panic!("{model_id}: missing hidden_size"));
        let hd = cfg.head_dim.unwrap_or(hs / n_heads);
        let dt = cfg.dtype_bytes();

        assert!(
            n_kv_heads % cli.tp == 0,
            "{model_id}: num_key_value_heads ({n_kv_heads}) not divisible by --tp ({})",
            cli.tp
        );
        let kv_heads_per_tp = n_kv_heads / cli.tp;

        eprintln!(
            "Model {model_id}: {nl} layers, {n_kv_heads} KV heads ({kv_heads_per_tp}/tp), \
             head_dim={hd}, dtype={}",
            cfg.torch_dtype.as_deref().unwrap_or("bf16"),
        );

        ResolvedLayout {
            num_layers: nl, outer_dim: 2, page_size: cli.page_size,
            inner_dim: kv_heads_per_tp * hd, dtype_bytes: dt,
            source: format!("--model {model_id} --tp {}", cli.tp),
        }
    } else {
        ResolvedLayout {
            num_layers: cli.num_layers, outer_dim: cli.outer_dim,
            page_size: cli.page_size, inner_dim: cli.inner_dim,
            dtype_bytes: cli.dtype_bytes, source: "manual flags".into(),
        }
    }
}
