# KVBM G4 Remote Storage Configuration

G4 is the remote storage tier in KVBM — KV cache data stored on shared filesystems or object stores. This document covers all configuration for the G4 transfer pipeline.

## Storage Backend

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_REMOTE_STORAGE_TYPE` | `auto` | Backend type: `disk`, `object`, or `auto` (auto-detects from other env vars) |

### Disk Backend

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_REMOTE_DISK_PATH` | — | Single disk cache path (e.g., `/mnt/kvbm_cache`) |
| `DYN_KVBM_REMOTE_DISK_PATHS` | — | Comma-separated paths for multi-path striping. Takes precedence over `_PATH` if both set. |
| `DYN_KVBM_REMOTE_DISK_USE_GDS` | `true` | Enable GPU Direct Storage for disk I/O. Set `0` for POSIX fallback. |
| `DYN_KVBM_REMOTE_DISK_GDS_READS_ONLY` | `false` | Use GDS only for reads, POSIX for writes. |
| `DYN_KVBM_REMOTE_DISK_O_DIRECT` | `true` | Use O_DIRECT for bypassing page cache. Recommended for large sequential I/O. |
| `DYN_KVBM_REMOTE_DISK_FD_CACHE_MAX_ENTRIES` | `1024` | Max open file descriptors cached per worker for disk I/O. |
| `DYN_KVBM_LOCAL_DISK_USE_GDS` | `true` | Enable GDS for local disk cache (G3 tier). |
| `DYN_KVBM_DISK_ZEROFILL_FALLBACK` | `true` | Zerofill blocks on disk read failure instead of erroring. |

### Object Store Backend (S3 / MinIO)

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_OBJECT_BUCKET` | — | Bucket name template (e.g., `kvbm-cache-{worker_id}`) |
| `DYN_KVBM_OBJECT_ENDPOINT` | — | S3-compatible endpoint URL |
| `DYN_KVBM_OBJECT_REGION` | — | AWS region |
| `DYN_KVBM_OBJECT_ACCESS_KEY` | — | Access key |
| `DYN_KVBM_OBJECT_SECRET_KEY` | — | Secret key |
| `DYN_KVBM_OBJECT_SESSION_TOKEN` | — | Session token (for temporary credentials) |
| `DYN_KVBM_OBJECT_SCHEME` | — | `http` or `https` |
| `DYN_KVBM_OBJECT_USE_VIRTUAL_ADDRESSING` | `false` | Use virtual-hosted-style URLs |
| `DYN_KVBM_OBJECT_REQ_CHECKSUM` | — | Request checksum algorithm |
| `DYN_KVBM_OBJECT_CA_BUNDLE` | — | Path to CA bundle for TLS |

## Transfer Pipeline

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_G4_PIPELINE_CHUNK_SIZE` | `16` | Blocks per chunk in the R2H→H2D pipeline. Larger = fewer NIXL requests, less pipeline overlap. Set to 64 in production. |
| `DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS` | `120` | Timeout for a single G4 transfer. Requests exceeding this are failed. |
| `DYN_KVBM_PREFETCH_TIMEOUT_SECS` | `10` | Timeout for the prefetch phase (disk→host). If prefetch doesn't complete in time, the slot is preempted and falls back to full prefill. |
| `DYN_KVBM_TRANSFER_BATCH_SIZE` | `128` | Max block pairs per DMA batch. Splits large H2D/D2H into parallel batches. |
| `DYN_KVBM_MAX_CONCURRENT_TRANSFERS` | `16` | Max concurrent local (non-G4) transfers. Used by the offload path. |
| `DYN_KVBM_FLUSH_BATCH_SIZE` | `1024` | Max blocks per offload flush batch. |
| `DYN_KVBM_G4_CHECKSUM_VALIDATION` | `false` | Enable checksum validation on G4 offload writes. Debug only — adds overhead. |

## Cache Tiers

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_CPU_CACHE_GB` | `62` | Host (pinned RAM) cache size in GB. Used as bounce buffer for G4 and as L2 cache. |
| `DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS` | — | Override host cache size in blocks (instead of GB). |
| `DYN_KVBM_DISK_CACHE_GB` | — | Local disk (G3) cache size in GB. Empty = disabled. |
| `DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS` | — | Override disk cache size in blocks. |
| `DYN_KVBM_DISABLE_CPU_CACHE_LOOKUP` | `false` | Disable host cache lookups (forces all onboards to go through G4). Debug only. |
| `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` | `false` | Disable offload filtering (offload all blocks, not just changed ones). Debug only. |

## Leader / Worker

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_LEADER_ZMQ_HOST` | `0.0.0.0` | ZMQ bind address for leader PUB/ACK sockets |
| `DYN_KVBM_LEADER_ZMQ_PUB_PORT` | `5570` | ZMQ PUB port (leader → workers broadcast) |
| `DYN_KVBM_LEADER_ZMQ_ACK_PORT` | `5571` | ZMQ ACK port (workers → leader acknowledgment) |
| `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS` | `120` | Timeout for all workers to connect at startup |

## Observability

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_KVBM_METRICS` | — | Enable Prometheus metrics endpoint. Set to `true` or `1`. |
| `DYN_KVBM_METRICS_PORT` | `6880` | Port for Prometheus metrics. |
| `DYN_KVBM_CACHE_STATS_MAX_REQUESTS` | `100` | Sliding window size for cache hit rate tracking. |
| `DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS` | `30` | Interval for logging cache statistics. |
| `DYN_KVBM_ENABLE_RECORD` | `false` | Enable recording of connector operations for replay/debugging. |
| `DYN_KVBM_G4_MIN_CANDIDATE_BLOCKS` | — | Minimum blocks for a request to be considered for G4 onboard. |
| `OTEL_EXPORT_ENABLED` | — | Enable OpenTelemetry trace export. Set to `1`. |
| `OTEL_SERVICE_NAME` | — | Service name for OTEL spans. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | — | OTLP collector endpoint (e.g., `http://host.docker.internal:4317`) |

## Other

| Env Var | Default | Description |
|---------|---------|-------------|
| `KVBM_DEV_MODE` | `false` | Enable dev-only operations (clear_pool, cache reset). Required for management API destructive actions. |
| `DYN_KVBM_ENABLE_NUMA` | `false` | Enable NUMA-aware memory allocation for host cache. |
| `NIXL_PLUGIN_DIR` | `/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins` | Path to NIXL backend plugins. |

## Transfer Pipeline Architecture

```
Request arrives
  │
  ▼
get_num_new_matched_tokens (scheduler, per iteration)
  │ Returns (num_external_tokens, needs_async_onboard)
  ▼
update_state_after_alloc (scheduler, after block allocation)
  │ Triggers prefetch or onboard effects
  ▼
build_connector_meta (scheduler, end of schedule())
  │ Serializes ConnectorMetadata with completion snapshot
  │ Includes: loads_done, stores_done, failed
  ▼
LocalTransferEngine (leader process)
  ├── Onboard task: LocalOnboardRequest → process_onboard_request
  │     Host/Disk blocks → ZMQ → worker DMA (H2D)
  │
  ├── Remote task: RemoteTransferRequest → process_remote_transfer_request
  │     │
  │     ├── Prefetch (device_block_ids empty):
  │     │     Allocate host bounce → build pipeline → ZMQ to workers
  │     │     Workers: NIXL disk→host (chunked) → register in host pool
  │     │     No H2D — data stays in host cache
  │     │
  │     └── Onboard (device_block_ids set):
  │           Same pipeline but with H2D after each R2H chunk
  │           (Only runs if prefetch host resolution fails)
  │
  ├── Offload task: LocalOffloadRequest → process_offload_request
  │     Device → Host (CUDA D2H DMA)
  │
  └── Drain task: DrainItem → Host → Disk (NIXL write)

Workers report completion via:
  get_finished() → reads loads_done/stores_done from ConnectorMetadata
  vLLM KVOutputAggregator counts across TP workers
  Scheduler processes finished_recving / finished_sending
```


### For debugging:
- Set `OTEL_EXPORT_ENABLED=1` and point `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to a collector
- All transfer spans are correlated by request trace ID
- Key spans to look for: `kvbm.prefetch`, `kvbm.onboard.from_host`, `kvbm.nixl_read`, `kvbm.h2d`
- Set `KVBM_DEV_MODE=TRUE` to enable management API (clear_pool, cache status)
