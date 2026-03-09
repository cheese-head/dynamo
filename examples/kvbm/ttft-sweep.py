#!/usr/bin/env python3
"""
TTFT sweep with 2 scenarios and optional concurrency:
  1. Cold   — clear CPU pool before each ISL, but keep CPU lookup enabled
  2. Warm   — CPU pool has data (host cache hit, no G4 needed)

Each scenario sends N requests per ISL. The same prompt is used for all
requests at a given ISL so the cache can be hit.

Requires KVBM_DEV_MODE=TRUE on the server.

Usage:
    python3 scripts/ttft-sweep.py --url http://localhost:19000 --mgmt http://localhost:19881
    python3 scripts/ttft-sweep.py --url http://localhost:19000 --mgmt http://localhost:19881 -c 4
    python3 scripts/ttft-sweep.py --url http://localhost:19000 --mgmt http://localhost:19881 --isls 1000 10000 128000 -c 8
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request
import urllib.error
import urllib.parse


def _make_traceparent() -> str:
    """Generate a W3C traceparent header with a random trace ID and span ID."""
    trace_id = os.urandom(16).hex()
    span_id = os.urandom(8).hex()
    return f"00-{trace_id}-{span_id}-01"


def _trace_id_from_traceparent(traceparent: str | None) -> str | None:
    if not traceparent:
        return None
    parts = traceparent.split("-")
    if len(parts) != 4:
        return None
    return parts[1]


def _fragment_to_text(fragment):
    if fragment is None:
        return ""
    if isinstance(fragment, str):
        return fragment
    if isinstance(fragment, dict):
        text = fragment.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(fragment, "text", None)
    return text if isinstance(text, str) else ""


def _value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_fragment_to_text(fragment) for fragment in value)
    return _fragment_to_text(value)


def _extract_delta_text(delta: dict) -> tuple[str, str]:
    content_text = _value_to_text(delta.get("content"))
    reasoning_text = ""
    for field_name in ("reasoning_content", "reasoning", "thinking", "reasoning_text"):
        reasoning_text = _value_to_text(delta.get(field_name))
        if reasoning_text:
            break
    return content_text, reasoning_text


# ── Determinism tracking ──────────────────────────────────────────────────

class DeterminismTracker:
    """Collects completions per (scenario, ISL) and checks response quality."""

    def __init__(self, strict_text_match: bool = False):
        self.completions: dict[tuple[str, int], list[dict]] = {}
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.strict_text_match = strict_text_match

    @staticmethod
    def _looks_garbage(text: str) -> bool:
        # Conservative heuristic: catch obvious repeated-token loops.
        tokens = [t for t in text.lower().split() if t]
        if len(tokens) < 24:
            return False
        unique = len(set(tokens))
        if unique <= 3:
            return True
        dominant = max(tokens.count(t) for t in set(tokens))
        return (dominant / len(tokens)) > 0.7

    def record(self, scenario: str, isl: int, completion: str | None, **meta):
        key = (scenario, isl)
        self.completions.setdefault(key, []).append({"completion": completion, **meta})

    def check_isl(self, scenario: str, isl: int):
        """Check completions for a (scenario, ISL), grouped by prompt key."""
        key = (scenario, isl)
        entries = self.completions.get(key, [])
        if not entries:
            return
        by_prompt_key: dict[str, list[dict]] = {}
        for e in entries:
            by_prompt_key.setdefault(e.get("prompt_key", "v0"), []).append(e)

        if len(by_prompt_key) > 1:
            self.notes.append(
                f"{scenario} ISL={fmt_isl(isl)}: {len(by_prompt_key)} prompt variants used; determinism is checked per variant"
            )

        for prompt_key, grouped in by_prompt_key.items():
            texts = [self._response_text(e) for e in grouped]
            non_none = [t for t in texts if t is not None]

            if not non_none:
                self.failures.append(
                    f"{scenario} ISL={fmt_isl(isl)} prompt={prompt_key}: all {len(texts)} responses returned null content"
                )
                continue

            null_count = len(texts) - len(non_none)
            if null_count:
                self.failures.append(
                    f"{scenario} ISL={fmt_isl(isl)} prompt={prompt_key}: {null_count}/{len(texts)} responses returned null content"
                )

            empty = [t for t in non_none if not t.strip()]
            if empty:
                self.failures.append(
                    f"{scenario} ISL={fmt_isl(isl)} prompt={prompt_key}: {len(empty)}/{len(non_none)} responses were empty/whitespace"
                )

            for t in non_none:
                if self._looks_garbage(t):
                    self.failures.append(
                        f"{scenario} ISL={fmt_isl(isl)} prompt={prompt_key}: suspicious repetitive output detected"
                    )

            if self.strict_text_match and len(non_none) >= 2:
                unique = set(non_none)
                if len(unique) > 1:
                    samples = list(unique)[:3]
                    self.failures.append(
                        f"{scenario} ISL={fmt_isl(isl)} prompt={prompt_key}: {len(unique)} distinct completions across {len(non_none)} responses — {samples}"
                    )

    def check_cross_scenario(self):
        """For each (ISL, prompt_key) across scenarios, verify they all agree."""
        if not self.strict_text_match:
            return
        by_isl_prompt: dict[tuple[int, str], dict[str, str]] = {}
        for (scenario, isl), entries in self.completions.items():
            by_prompt_key: dict[str, list[str]] = {}
            for e in entries:
                t = self._response_text(e)
                if t is not None:
                    by_prompt_key.setdefault(e.get("prompt_key", "v0"), []).append(t)

            for prompt_key, texts in by_prompt_key.items():
                if texts:
                    majority = max(set(texts), key=texts.count)
                    by_isl_prompt.setdefault((isl, prompt_key), {})[scenario] = majority

        for (isl, prompt_key), scenario_map in sorted(by_isl_prompt.items()):
            unique_texts = set(scenario_map.values())
            if len(unique_texts) > 1:
                detail = ", ".join(f"{s}={repr(t)}" for s, t in scenario_map.items())
                self.failures.append(
                    f"CROSS-SCENARIO ISL={fmt_isl(isl)} prompt={prompt_key}: scenarios disagree — {detail}"
                )

    def dump(self, path: str):
        """Write all recorded completions to a JSON file."""
        output_path = Path(path)
        if output_path.is_dir():
            output_path = output_path / "responses.json"
        else:
            parent = output_path.parent
            if parent != Path("."):
                if parent.exists() and not parent.is_dir():
                    alt_path = Path(f"{path}.json")
                    print(
                        f"  Warning: parent path '{parent}' is a file; writing responses to '{alt_path}' instead"
                    )
                    output_path = alt_path
                    parent = output_path.parent
                parent.mkdir(parents=True, exist_ok=True)
        records = []
        for (scenario, isl), entries in sorted(self.completions.items()):
            for i, entry in enumerate(entries):
                records.append(
                    {
                    "scenario": scenario,
                    "isl": isl,
                    "request": i,
                    **entry,
                    }
                )
        with output_path.open("w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Responses written to {output_path}")

    @staticmethod
    def _response_text(entry: dict) -> str | None:
        return entry.get("completion") or entry.get("reasoning")

    def summary(self, dump_path: str | None = None):
        total_keys = len(self.completions)
        total_responses = sum(len(v) for v in self.completions.values())
        if total_responses == 0:
            return

        if dump_path:
            self.dump(dump_path)

        self.check_cross_scenario()

        if not self.failures:
            mode = "strict text match" if self.strict_text_match else "logic-focused"
            print(f"\n  ┌─ Response Quality: PASS ({total_responses} responses across {total_keys} groups, mode={mode})")
            if self.strict_text_match:
                print(f"  └─ ✓ All responses identical within each (scenario, ISL) and across scenarios")
            else:
                print(f"  └─ ✓ No null/empty/garbage outputs detected")
        else:
            label = "Determinism" if self.strict_text_match else "Response Quality"
            print(f"\n  ┌─ {label}: {len(self.failures)} ISSUE(S) ({total_responses} responses across {total_keys} groups)")
            for f in self.failures:
                print(f"  │  ✗ {f}")
            if self.strict_text_match:
                print(f"  └─ WARNING: non-deterministic or garbage responses detected")
            else:
                print(f"  └─ WARNING: null/empty/garbage responses detected")
        for n in self.notes:
            print(f"  · Note: {n}")


determinism = DeterminismTracker()


def _tempo_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _attr_map(attrs):
    out = {}
    for a in attrs or []:
        key = a.get("key")
        val = a.get("value", {})
        if "stringValue" in val:
            out[key] = val["stringValue"]
        elif "intValue" in val:
            out[key] = int(val["intValue"])
        elif "boolValue" in val:
            out[key] = bool(val["boolValue"])
        elif "doubleValue" in val:
            out[key] = float(val["doubleValue"])
    return out


def _flatten_trace_spans(trace: dict):
    spans = []
    for batch in trace.get("batches", []):
        resource_attrs = _attr_map(batch.get("resource", {}).get("attributes", []))
        service = resource_attrs.get("service.name")
        for scope_spans in batch.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                attrs = _attr_map(span.get("attributes", []))
                effective_name = attrs.get("otel.name") or span.get("name")
                spans.append(
                    {
                        "name": effective_name,
                        "raw_name": span.get("name"),
                        "start": int(span.get("startTimeUnixNano", "0")),
                        "end": int(span.get("endTimeUnixNano", "0")),
                        "attrs": attrs,
                        "service": service,
                    }
                )
    spans.sort(key=lambda s: s["start"])
    return spans


def _trace_summary(trace_id: str, trace: dict) -> dict:
    spans = _flatten_trace_spans(trace)
    if not spans:
        return {"trace_id": trace_id, "span_count": 0}

    names = {s["name"] for s in spans}
    attrs = {}
    for s in spans:
        attrs.update(s["attrs"])

    req_span = next((s for s in spans if s["name"] == "kvbm.request"), None)
    first_token = next((s for s in spans if s["name"] == "vllm.first_token"), None)
    prefill = next((s for s in spans if s["name"] == "vllm.prefill_scheduled"), None)
    worker_remote = next((s for s in spans if s["name"] == "kvbm.worker_remote_transfer"), None)
    remote_chunked = next((s for s in spans if s["name"] == "kvbm.remote_transfer_chunked"), None)
    trigger_onboarding = next((s for s in spans if s["name"] == "kvbm.trigger_onboarding"), None)
    onboard_from_g4 = next((s for s in spans if s["name"] == "kvbm.onboard_from_g4"), None)

    summary = {
        "trace_id": trace_id,
        "span_count": len(spans),
        "request_id": attrs.get("request_id") or attrs.get("gen_ai.request.id"),
        "prompt_tokens": attrs.get("num_prompt_tokens") or attrs.get("gen_ai.usage.prompt_tokens"),
        "completion_tokens": attrs.get("num_output_tokens") or attrs.get("gen_ai.usage.completion_tokens"),
        "host_matched": attrs.get("host_matched"),
        "disk_matched": attrs.get("disk_matched"),
        "g4_matched": attrs.get("g4_matched"),
        "num_matched": attrs.get("num_matched"),
        "finish_reason": attrs.get("finish_reason"),
        "has_remote_transfer_chunked": "kvbm.remote_transfer_chunked" in names,
    }

    if req_span and worker_remote:
        summary["worker_remote_transfer_start_ms"] = round(
            (worker_remote["start"] - req_span["start"]) / 1e6, 2
        )
    if trigger_onboarding and worker_remote:
        summary["trigger_to_worker_remote_start_ms"] = round(
            (worker_remote["start"] - trigger_onboarding["start"]) / 1e6, 2
        )
    if onboard_from_g4 and worker_remote:
        summary["onboard_to_worker_remote_start_ms"] = round(
            (worker_remote["start"] - onboard_from_g4["start"]) / 1e6, 2
        )
    if worker_remote:
        summary["worker_remote_transfer_dur_ms"] = round(
            (worker_remote["end"] - worker_remote["start"]) / 1e6, 2
        )
    if remote_chunked:
        summary["remote_transfer_chunked_dur_ms"] = round(
            (remote_chunked["end"] - remote_chunked["start"]) / 1e6, 2
        )
    if req_span and prefill:
        summary["prefill_scheduled_at_ms"] = round(
            (prefill["start"] - req_span["start"]) / 1e6, 2
        )
    if req_span and first_token:
        summary["first_token_at_ms"] = round(
            (first_token["start"] - req_span["start"]) / 1e6, 2
        )

    return summary


def _trace_has_required_spans(trace: dict) -> bool:
    spans = _flatten_trace_spans(trace)
    names = {s["name"] for s in spans}
    return (
        "kvbm.worker_remote_transfer" in names
        or "vllm.prefill_scheduled" in names
        or "vllm.first_token" in names
        or "vllm.finished" in names
    )


def _trace_timeline_rows(trace: dict) -> list[dict]:
    spans = _flatten_trace_spans(trace)
    if not spans:
        return []

    base_start = spans[0]["start"]
    interesting = {
        "kvbm.request",
        "kvbm.request_poll",
        "kvbm.get_matched_tokens",
        "kvbm.stage_local_matches",
        "kvbm.update_state_after_alloc",
        "kvbm.trigger_onboarding",
        "kvbm.onboard_from_g4",
        "kvbm.flush_onboarding",
        "kvbm.process_remote_transfer",
        "kvbm.remote_transfer_allocate",
        "kvbm.remote_transfer_build_pipeline",
        "kvbm.remote_transfer_dispatch",
        "kvbm.worker_remote_transfer",
        "kvbm.remote_transfer",
        "kvbm.remote_transfer_chunked",
        "kvbm.remote_transfer_chunk_r2h",
        "kvbm.remote_transfer_chunk_h2d",
        "kvbm.r2h",
        "kvbm.h2d",
        "vllm.prefill_scheduled",
        "vllm.first_token",
        "kvbm.request_finished",
        "vllm.finished",
    }

    rows = []
    for span in spans:
        if span["name"] not in interesting:
            continue
        start_ms = (span["start"] - base_start) / 1e6
        end_ms = (span["end"] - base_start) / 1e6
        dur_ms = (span["end"] - span["start"]) / 1e6
        rows.append(
            {
                "span": span["name"],
                "start_ms": round(start_ms, 2),
                "end_ms": round(end_ms, 2),
                "dur_ms": round(dur_ms, 2),
            }
        )
    return rows


def _print_trace_timeline_table(trace_id: str, rows: list[dict]):
    short_id = trace_id[:16]
    print(f"\n  Trace timeline {short_id}")
    hdr = f"  {'Span':<30} │ {'Start':>10} │ {'End':>10} │ {'Dur':>10}"
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    trigger = next((r for r in rows if r["span"] == "kvbm.trigger_onboarding"), None)
    onboard = next((r for r in rows if r["span"] == "kvbm.onboard_from_g4"), None)
    worker = next((r for r in rows if r["span"] == "kvbm.worker_remote_transfer"), None)
    if trigger and worker:
        print(
            f"  gap trigger_onboarding -> worker_remote_transfer: {worker['start_ms'] - trigger['start_ms']:.2f}ms"
        )
    if onboard and worker:
        print(
            f"  gap onboard_from_g4 -> worker_remote_transfer: {worker['start_ms'] - onboard['start_ms']:.2f}ms"
        )
    for row in rows:
        print(
            f"  {row['span']:<30} │ {row['start_ms']:>8.2f}ms │ {row['end_ms']:>8.2f}ms │ {row['dur_ms']:>8.2f}ms"
        )
    print(sep)


def _print_overlap_table(trace_rows: dict[str, list[dict]]):
    if len(trace_rows) < 2:
        return

    trace_ids = list(trace_rows.keys())
    span_names = []
    for rows in trace_rows.values():
        for row in rows:
            if row["span"] not in span_names:
                span_names.append(row["span"])

    print("\n  Trace overlap")
    hdr = f"  {'Span':<30}"
    for trace_id in trace_ids:
        hdr += f" │ {trace_id[:8]} start/end"
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for span_name in span_names:
        line = f"  {span_name:<30}"
        for trace_id in trace_ids:
            row = next((r for r in trace_rows[trace_id] if r["span"] == span_name), None)
            if row is None:
                line += f" │ {'-':>17}"
            else:
                line += f" │ {row['start_ms']:>6.0f}-{row['end_ms']:<6.0f}ms"
        print(line)
    print(sep)


def _print_global_overlap_summary(trace_rows: dict[str, list[dict]]):
    if len(trace_rows) < 2:
        return

    all_rows = [row for rows in trace_rows.values() for row in rows]
    if not all_rows:
        return

    global_min = min(row["start_ms"] for row in all_rows)

    def _find(rows: list[dict], span_name: str):
        return next((r for r in rows if r["span"] == span_name), None)

    ordered = []
    for trace_id, rows in trace_rows.items():
        worker = _find(rows, "kvbm.worker_remote_transfer")
        first_token = _find(rows, "vllm.first_token")
        sort_key = (
            worker["start_ms"] if worker else float("inf"),
            first_token["start_ms"] if first_token else float("inf"),
            trace_id,
        )
        ordered.append((sort_key, trace_id, rows))
    ordered.sort()

    print("\n  Trace overlap (global time)")
    hdr = (
        f"  {'Trace':<10} │ {'trigger':>8} │ {'worker_s':>8} │ {'worker_e':>8} │ "
        f"{'prefill':>8} │ {'first_tok':>9} │ {'xfer_ms':>8}"
    )
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for _, trace_id, rows in ordered:
        trigger = _find(rows, "kvbm.trigger_onboarding")
        worker = _find(rows, "kvbm.worker_remote_transfer")
        prefill = _find(rows, "vllm.prefill_scheduled")
        first_token = _find(rows, "vllm.first_token")

        def rel_start(row):
            return f"{row['start_ms'] - global_min:>6.0f}ms" if row else f"{'-':>8}"

        def rel_end(row):
            return f"{row['end_ms'] - global_min:>6.0f}ms" if row else f"{'-':>8}"

        xfer_ms = f"{worker['dur_ms']:>6.0f}ms" if worker else f"{'-':>8}"

        print(
            f"  {trace_id[:8]:<10} │ {rel_start(trigger)} │ {rel_start(worker)} │ {rel_end(worker)} │ "
            f"{rel_start(prefill)} │ {rel_start(first_token)} │ {xfer_ms}"
        )
    print(sep)


def _print_global_overlap_matrix(trace_rows: dict[str, list[dict]]):
    if len(trace_rows) < 2:
        return

    all_rows = [row for rows in trace_rows.values() for row in rows]
    if not all_rows:
        return

    global_min = min(row["start_ms"] for row in all_rows)
    trace_ids = list(trace_rows.keys())
    span_names = []
    for rows in trace_rows.values():
        for row in rows:
            if row["span"] not in span_names:
                span_names.append(row["span"])

    print("\n  Trace overlap (global span matrix)")
    hdr = f"  {'Span':<30}"
    for trace_id in trace_ids:
        hdr += f" │ {trace_id[:8]} start/end"
    sep = "  " + "─" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for span_name in span_names:
        line = f"  {span_name:<30}"
        for trace_id in trace_ids:
            row = next((r for r in trace_rows[trace_id] if r["span"] == span_name), None)
            if row is None:
                line += f" │ {'-':>17}"
            else:
                start = row["start_ms"] - global_min
                end = row["end_ms"] - global_min
                line += f" │ {start:>6.0f}-{end:<6.0f}ms"
        print(line)
    print(sep)


def fetch_tempo_traces(
    completions: dict,
    tempo_url: str,
    output_path: str,
    include_raw: bool = False,
    max_wait_seconds: float = 15.0,
    retry_interval_seconds: float = 1.0,
):
    trace_ids = []
    for entries in completions.values():
        for entry in entries:
            trace_id = entry.get("trace_id")
            if trace_id and trace_id not in trace_ids:
                trace_ids.append(trace_id)

    if not trace_ids:
        print("  Tempo trace fetch skipped: no trace IDs recorded")
        return

    base_output = Path(output_path)
    if base_output.is_dir():
        trace_dir = base_output / "traces"
    else:
        trace_dir = base_output.with_suffix("")
        trace_dir = trace_dir.parent / f"{trace_dir.name}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    timeline_index = {}
    for trace_id in trace_ids:
        trace = None
        try:
            deadline = time.time() + max_wait_seconds
            last_trace = None
            last_error = None
            while time.time() < deadline:
                try:
                    candidate = _tempo_get_json(f"{tempo_url}/api/traces/{trace_id}")
                    last_trace = candidate
                    if _trace_has_required_spans(candidate):
                        trace = candidate
                        break
                except Exception as e:
                    last_error = e
                time.sleep(retry_interval_seconds)

            if trace is None:
                if last_trace is not None:
                    trace = last_trace
                elif last_error is not None:
                    raise last_error
                else:
                    raise RuntimeError("Tempo trace fetch timed out")

            summary = _trace_summary(trace_id, trace)
            summary["complete_enough"] = _trace_has_required_spans(trace)
            summaries.append(summary)
            timeline_rows = _trace_timeline_rows(trace)
            timeline_index[trace_id] = timeline_rows
            summary_path = trace_dir / f"{trace_id}.summary.json"
            with summary_path.open("w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            timeline_path = trace_dir / f"{trace_id}.timeline.json"
            with timeline_path.open("w") as f:
                json.dump(timeline_rows, f, indent=2, ensure_ascii=False)
            if include_raw:
                raw_path = trace_dir / f"{trace_id}.trace.json"
                with raw_path.open("w") as f:
                    json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            summaries.append({"trace_id": trace_id, "error": str(e)})

    index_path = trace_dir / "trace_summaries.json"
    with index_path.open("w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"  Tempo traces written to {trace_dir}")

    for trace_id in trace_ids:
        rows = timeline_index.get(trace_id)
        if rows:
            _print_trace_timeline_table(trace_id, rows)
    incomplete = [s["trace_id"] for s in summaries if not s.get("complete_enough", True)]
    if incomplete:
        print(
            "  Warning: some traces were still partial after retry window: "
            + ", ".join(t[:16] for t in incomplete)
        )
    if timeline_index:
        _print_global_overlap_summary(timeline_index)
        _print_global_overlap_matrix(timeline_index)
        _print_overlap_table(timeline_index)


# ── Management API helpers ───────────────────────────────────────────────

def mgmt_post(mgmt_url: str, path: str, body: dict | None = None) -> dict | None:
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(
        f"{mgmt_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()) if resp.status == 200 else None
    except Exception as e:
        print(f"  [mgmt] {path} failed: {e}", file=sys.stderr)
        return None


def mgmt_get(mgmt_url: str, path: str) -> dict | None:
    req = urllib.request.Request(f"{mgmt_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()) if resp.status == 200 else None
    except Exception as e:
        print(f"  [mgmt] {path} failed: {e}", file=sys.stderr)
        return None


def clear_cpu_pool(mgmt_url: str):
    mgmt_post(mgmt_url, "/v1/cache/clear", {"pool": "cpu"})


def clear_all_pools(mgmt_url: str):
    mgmt_post(mgmt_url, "/v1/cache/clear_all")


def ensure_clear_cpu_pool(mgmt_url: str, retries: int = 3, delay_s: float = 2.0):
    last_error = None
    for attempt in range(1, retries + 1):
        result = mgmt_post(mgmt_url, "/v1/cache/clear", {"pool": "cpu"})
        if result is not None:
            return
        last_error = f"CPU cache clear failed on attempt {attempt}/{retries}"
        if attempt < retries:
            time.sleep(delay_s)
    raise RuntimeError(last_error or "CPU cache clear failed")


def check_health(mgmt_url: str) -> bool:
    for path in ("/v1/health",):
        r = mgmt_get(mgmt_url, path)
        if r is not None:
            return True
    return False


# ── Request helpers ──────────────────────────────────────────────────────

def make_prompt(num_tokens: int, variant: int = 0) -> list[dict]:
    """Build a chat message of approximately `num_tokens` tokens.

    Each `variant` value produces a deterministically different prompt of the
    same length. The prompt families are intentionally quite different so
    concurrent requests do not share most of their long prefix and collapse
    into the G4 inflight-dedupe path.
    """
    vocab_groups = [
        ["hi", "yo", "ok", "ah", "go", "up", "an", "we"],
        ["he", "if", "is", "it", "no", "my", "or", "us"],
        ["be", "by", "in", "me", "of", "on", "to", "so"],
        ["do", "at", "as", "oh", "um", "he", "go", "my"],
        ["we", "to", "an", "if", "up", "no", "by", "ok"],
        ["yo", "it", "me", "or", "so", "of", "ah", "us"],
        ["go", "be", "on", "my", "at", "we", "in", "hi"],
        ["ok", "no", "to", "as", "yo", "he", "of", "up"],
    ]
    group = vocab_groups[variant % len(vocab_groups)]
    rotated = group[variant % len(group):] + group[:variant % len(group)]

    # Use a variant-specific repeated sentence so the token stream diverges
    # across workers for the full prompt length, not just the first few tokens.
    sentence = " ".join(rotated)
    words = sentence.split()
    prompt_words = []
    while len(prompt_words) < num_tokens:
        prompt_words.extend(words)
    content = " ".join(prompt_words[:num_tokens])
    return [{"role": "user", "content": content}]


def send_request(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1,
    seed: int | None = None,
    stream: bool = False,
    ttft_mode: str = "either",
) -> dict:
    traceparent = _make_traceparent()
    headers = {"Content-Type": "application/json"}
    # Only send traceparent if OTEL is NOT enabled on the server.
    # When vLLM has --otlp-traces-endpoint, it creates its own root span;
    # sending a client traceparent makes the root a phantom (never exported).
    if os.environ.get("SEND_TRACEPARENT", "0") == "1":
        headers["traceparent"] = traceparent
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if seed is not None:
        payload["seed"] = seed
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if stream:
                return _read_streaming_response(resp, start, traceparent, ttft_mode)
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        return {
            "error": f"HTTP {e.code}: {error_body}",
            "ttft": None,
            "traceparent": traceparent,
            "trace_id": _trace_id_from_traceparent(traceparent),
        }
    except Exception as e:
        return {
            "error": str(e),
            "ttft": None,
            "traceparent": traceparent,
            "trace_id": _trace_id_from_traceparent(traceparent),
        }

    ttft = time.perf_counter() - start
    usage = body.get("usage", {})
    choices = body.get("choices", [])
    completion = None
    finish_reason = None
    stop_reason = None
    tool_calls = 0
    refusal = None
    reasoning = None
    if choices:
        c0 = choices[0]
        finish_reason = c0.get("finish_reason")
        stop_reason = c0.get("stop_reason")
        msg = c0.get("message") or c0.get("delta") or {}
        completion = msg.get("content") or c0.get("text")
        tool_calls = len(msg.get("tool_calls") or [])
        refusal = msg.get("refusal")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    return {
        "ttft": ttft,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "completion": completion,
        "response_id": body.get("id"),
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
        "refusal": refusal,
        "reasoning": reasoning,
        "seed": seed,
        "error": None,
        "traceparent": traceparent,
        "trace_id": _trace_id_from_traceparent(traceparent),
    }


def _select_stream_ttft(first_event_ttft, first_reasoning_ttft, first_content_ttft, ttft_mode: str):
    if ttft_mode == "event":
        return first_event_ttft
    if ttft_mode == "thinking":
        return first_reasoning_ttft
    if ttft_mode == "content":
        return first_content_ttft
    if ttft_mode == "either":
        candidates = [v for v in (first_reasoning_ttft, first_content_ttft) if v is not None]
        return min(candidates) if candidates else None
    raise ValueError(f"Unknown ttft_mode: {ttft_mode}")


def _read_streaming_response(resp, start: float, traceparent: str, ttft_mode: str) -> dict:
    first_event_ttft = None
    first_reasoning_ttft = None
    first_content_ttft = None
    prompt_tokens = None
    completion_tokens = None
    response_id = None
    finish_reason = None
    stop_reason = None
    tool_calls = 0
    refusal = None
    completion_parts = []
    reasoning_parts = []

    while True:
        raw_line = resp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break

        now = time.perf_counter()
        if first_event_ttft is None:
            first_event_ttft = now - start

        chunk = json.loads(data)
        response_id = response_id or chunk.get("id")
        usage = chunk.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        completion_tokens = usage.get("completion_tokens", completion_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason", finish_reason)
        stop_reason = choice.get("stop_reason", stop_reason)
        delta = choice.get("delta") or {}
        content_text, reasoning_text = _extract_delta_text(delta)

        if reasoning_text:
            reasoning_parts.append(reasoning_text)
            if first_reasoning_ttft is None:
                first_reasoning_ttft = now - start
        if content_text:
            completion_parts.append(content_text)
            if first_content_ttft is None:
                first_content_ttft = now - start

        tool_calls += len(delta.get("tool_calls") or [])
        refusal = refusal or delta.get("refusal")

    ttft = _select_stream_ttft(first_event_ttft, first_reasoning_ttft, first_content_ttft, ttft_mode)
    if ttft is None:
        ttft = first_event_ttft

    return {
        "ttft": ttft,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion": "".join(completion_parts) or None,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "tool_calls": tool_calls,
        "refusal": refusal,
        "reasoning": "".join(reasoning_parts) or None,
        "error": None,
        "traceparent": traceparent,
        "trace_id": _trace_id_from_traceparent(traceparent),
        "first_event_ttft": first_event_ttft,
        "first_reasoning_ttft": first_reasoning_ttft,
        "first_content_ttft": first_content_ttft,
    }


# ── Formatting ───────────────────────────────────────────────────────────

def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_isl(n: int) -> str:
    if n >= 1000:
        k = n / 1000
        return f"{k:.0f}K" if k == int(k) else f"{k:.1f}K"
    return str(n)


# ── Scenario runners ─────────────────────────────────────────────────────

def run_scenario(
    name: str,
    url: str,
    model: str,
    mgmt_url: str,
    isls: list[int],
    n: int,
    max_tokens: int,
    setup_fn,
    concurrency: int = 1,
    clear_between: bool = False,
    skip_cpu_flush: bool = False,
    seed: int | None = None,
    stream: bool = False,
    ttft_mode: str = "either",
    warmup_rounds: int = 0,
) -> list[dict]:
    """Run a scenario: for each ISL, call setup_fn then send n requests.

    When concurrency > 1, requests are sent in parallel using a thread pool.
    When clear_between is True, CPU pool is cleared after each request
    (and a short wait allows offload to G4 before clearing).
    When warmup_rounds > 0, that many untimed requests are sent first per ISL
    to burn off JIT/CUDA-graph warmup before the timed measurement begins.
    """
    results = []

    for isl in isls:
        # Create one prompt variant per concurrent worker to avoid inflight dedupe
        # while still repeating the same prompts across requests.
        variant_count = max(1, concurrency)
        all_messages = [make_prompt(isl, variant=i) for i in range(variant_count)]

        if setup_fn is not None:
            for v in range(min(concurrency, len(all_messages))):
                setup_fn(mgmt_url, url, model, all_messages[v], isl, skip_cpu_flush, seed)
            time.sleep(0.5)

        if warmup_rounds > 0:
            print(f"    [{name}] ISL={fmt_isl(isl)}: {warmup_rounds} warmup round(s)...")
            for w in range(warmup_rounds):
                wr = send_request(
                    url, model, all_messages[0], max_tokens,
                    seed=seed, stream=stream, ttft_mode=ttft_mode,
                )
                label = fmt_time(wr["ttft"]) if not wr["error"] else f"ERROR {wr['error']}"
                print(f"    [{name}] ISL={fmt_isl(isl)} warmup {w+1}/{warmup_rounds}: {label}")
            print(f"    [{name}] ISL={fmt_isl(isl)}: warmup done, starting timed runs")

        ttfts = []
        errors = 0

        if concurrency <= 1:
            for i in range(n):
                if stream:
                    r = send_request(
                        url, model, all_messages[0], max_tokens, seed=seed, stream=True, ttft_mode=ttft_mode
                    )
                else:
                    r = send_request(url, model, all_messages[0], max_tokens, seed=seed)
                if r["error"]:
                    errors += 1
                    print(
                        f"    [{name}] ISL={fmt_isl(isl)} run {i+1}/{n}: ERROR {r['error']} trace_id={r.get('trace_id')}",
                        file=sys.stderr,
                    )
                else:
                    ttfts.append(r["ttft"])
                    print(
                        f"    [{name}] ISL={fmt_isl(isl)} run {i+1}/{n}: TTFT={fmt_time(r['ttft'])} trace_id={r.get('trace_id')}"
                    )
                    determinism.record(
                        name,
                        isl,
                        r["completion"],
                        prompt_key="v0",
                        reasoning=r.get("reasoning"),
                        traceparent=r.get("traceparent"),
                        trace_id=r.get("trace_id"),
                        response_id=r.get("response_id"),
                        finish_reason=r.get("finish_reason"),
                        stop_reason=r.get("stop_reason"),
                        tool_calls=r.get("tool_calls"),
                        refusal=r.get("refusal"),
                        has_reasoning=bool(r.get("reasoning")),
                        first_event_ttft=r.get("first_event_ttft"),
                        first_reasoning_ttft=r.get("first_reasoning_ttft"),
                        first_content_ttft=r.get("first_content_ttft"),
                        prompt_tokens=r.get("prompt_tokens"),
                        completion_tokens=r.get("completion_tokens"),
                        seed=r.get("seed"),
                    )
                if clear_between and not skip_cpu_flush:
                    time.sleep(5)
                    ensure_clear_cpu_pool(mgmt_url)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_meta = {}
                for i in range(n):
                    variant = i % variant_count
                    fut = pool.submit(
                        send_request,
                        url,
                        model,
                        all_messages[variant],
                        max_tokens,
                        seed,
                        stream,
                        ttft_mode,
                    )
                    future_meta[fut] = variant
                for i, fut in enumerate(concurrent.futures.as_completed(future_meta)):
                    r = fut.result()
                    if r["error"]:
                        errors += 1
                        print(
                            f"    [{name}] ISL={fmt_isl(isl)} req {i+1}/{n}: ERROR {r['error']} trace_id={r.get('trace_id')}",
                            file=sys.stderr,
                        )
                    else:
                        ttfts.append(r["ttft"])
                        print(
                            f"    [{name}] ISL={fmt_isl(isl)} req {i+1}/{n}: TTFT={fmt_time(r['ttft'])} trace_id={r.get('trace_id')} prompt=v{future_meta[fut]}"
                        )
                        determinism.record(
                            name,
                            isl,
                            r["completion"],
                            prompt_key=f"v{future_meta[fut]}",
                            reasoning=r.get("reasoning"),
                            traceparent=r.get("traceparent"),
                            trace_id=r.get("trace_id"),
                            response_id=r.get("response_id"),
                            finish_reason=r.get("finish_reason"),
                            stop_reason=r.get("stop_reason"),
                            tool_calls=r.get("tool_calls"),
                            refusal=r.get("refusal"),
                            has_reasoning=bool(r.get("reasoning")),
                            first_event_ttft=r.get("first_event_ttft"),
                            first_reasoning_ttft=r.get("first_reasoning_ttft"),
                            first_content_ttft=r.get("first_content_ttft"),
                            prompt_tokens=r.get("prompt_tokens"),
                            completion_tokens=r.get("completion_tokens"),
                            seed=r.get("seed"),
                        )

        determinism.check_isl(name, isl)

        row = {
            "isl": isl,
            "n": len(ttfts),
            "errors": errors,
            "concurrency": concurrency,
            "min": min(ttfts) if ttfts else None,
            "p50": statistics.median(ttfts) if ttfts else None,
            "p95": sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) >= 2 else (ttfts[0] if ttfts else None),
            "max": max(ttfts) if ttfts else None,
            "mean": statistics.mean(ttfts) if ttfts else None,
        }
        results.append(row)
    return results


def setup_cold(
    mgmt_url,
    url,
    model,
    messages,
    isl,
    skip_cpu_flush: bool = False,
    seed: int | None = None,
):
    """Scenario 1: clear CPU pool but keep CPU lookup enabled."""
    if not skip_cpu_flush:
        ensure_clear_cpu_pool(mgmt_url)
    send_request(url, model, messages, 1, seed=seed)
    time.sleep(5)
    if not skip_cpu_flush:
        ensure_clear_cpu_pool(mgmt_url)


def setup_warm(
    mgmt_url,
    url,
    model,
    messages,
    isl,
    skip_cpu_flush: bool = False,
    seed: int | None = None,
):
    """Scenario 2: ensure CPU pool is populated (host cache hit)."""
    send_request(url, model, messages, 1, seed=seed)
    time.sleep(5)


# ── Output ───────────────────────────────────────────────────────────────

def print_table(name: str, results: list[dict], concurrency: int):
    c_label = f" (concurrency={concurrency})" if concurrency > 1 else ""
    hdr = f"  {'ISL':>7}  │  {'Min':>9}  │  {'P50':>9}  │  {'P95':>9}  │  {'Max':>9}  │  {'Mean':>9}  │  {'N':>3}  │  Err"
    sep = "  " + "─" * (len(hdr) - 2)

    print(f"\n  ┌─ {name}{c_label}")
    print(hdr)
    print(sep)

    for r in results:
        if r["n"] == 0:
            print(f"  {fmt_isl(r['isl']):>7}  │  {'ERROR':>9}  │  {'-':>9}  │  {'-':>9}  │  {'-':>9}  │  {'-':>9}  │  {0:>3}  │  {r['errors']}")
        else:
            print(
                f"  {fmt_isl(r['isl']):>7}  │  {fmt_time(r['min']):>9}  │  {fmt_time(r['p50']):>9}  │"
                f"  {fmt_time(r['p95']):>9}  │  {fmt_time(r['max']):>9}  │  {fmt_time(r['mean']):>9}  │  {r['n']:>3}  │  {r['errors']}"
            )

    print(sep)


def print_comparison(cold: list[dict], warm: list[dict]):
    hdr = f"  {'ISL':>7}  │  {'Cold':>9}  │  {'Warm':>9}  │  {'Cold/Warm':>9}"
    sep = "  " + "─" * (len(hdr) - 2)

    print(f"\n  ┌─ Comparison (P50 TTFT)")
    print(hdr)
    print(sep)

    for c, w in zip(cold, warm):
        cp = fmt_time(c["p50"])
        wp = fmt_time(w["p50"])

        cw_ratio = f"{c['p50'] / w['p50']:.1f}x" if (c["p50"] and w["p50"] and w["p50"] > 0) else "-"

        print(f"  {fmt_isl(c['isl']):>7}  │  {cp:>9}  │  {wp:>9}  │  {cw_ratio:>9}")

    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TTFT sweep")
    parser.add_argument("--url", default="http://localhost:19000", help="vLLM base URL")
    parser.add_argument("--mgmt", default="http://localhost:19881", help="KVBM management URL")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument(
        "--isls", type=int, nargs="+",
        default=[1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 120000],
    )
    parser.add_argument("-n", type=int, default=10, help="Requests per ISL per scenario")
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                        help="Concurrent requests per ISL (default: 1 = sequential)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming chat completions and measure streamed TTFT instead of full-response latency",
    )
    parser.add_argument(
        "--ttft-mode",
        choices=["event", "thinking", "content", "either"],
        default="either",
        help="When --stream is set, which streamed signal defines TTFT",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed sampling seed")
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Require exact text match across repeated requests (disabled by default)",
    )
    parser.add_argument(
        "--send-traceparent",
        action="store_true",
        help="Send W3C traceparent header so request rows can be correlated in Tempo",
    )
    parser.add_argument("--scenarios", nargs="+", default=["cold", "warm"],
                        choices=["cold", "warm"], help="Which scenarios to run")
    parser.add_argument(
        "--skip-cpu-flush",
        action="store_true",
        help="Skip CPU pool flush operations (useful for faster cold-scenario iteration)",
    )
    parser.add_argument("-o", "--output", default="responses.json",
                        help="Path to write all completions (default: responses.json)")
    parser.add_argument(
        "--fetch-traces",
        action="store_true",
        help="Fetch Tempo traces for recorded trace IDs and write summaries next to the output JSON",
    )
    parser.add_argument(
        "--tempo-url",
        default="http://localhost:3200",
        help="Tempo base URL used with --fetch-traces",
    )
    parser.add_argument(
        "--trace-raw",
        action="store_true",
        help="With --fetch-traces, also write full raw trace JSON for each trace ID",
    )
    parser.add_argument(
        "--trace-fetch-wait",
        type=float,
        default=15.0,
        help="Seconds to wait/retry for Tempo traces to become complete enough",
    )
    args = parser.parse_args()

    print(f"\n  TTFT Sweep")
    print(f"  Model:       {args.model}")
    print(f"  Endpoint:    {args.url}")
    print(f"  Mgmt:        {args.mgmt}")
    print(f"  ISLs:        {', '.join(fmt_isl(i) for i in args.isls)}")
    print(f"  N/ISL:       {args.n}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Scenarios:   {', '.join(args.scenarios)}")
    print(f"  Streaming:   {args.stream}")
    print(f"  TTFT mode:   {args.ttft_mode if args.stream else 'full-response'}")
    print(f"  Skip flush:  {args.skip_cpu_flush}")
    print(f"  Traceparent: {args.send_traceparent}")
    print(f"  Fetch traces: {args.fetch_traces}")
    if args.fetch_traces:
        print(f"  Tempo URL:   {args.tempo_url}")
    print(f"  Seed:        {args.seed if args.seed is not None else '-'}")
    print(f"  Strict det:  {args.strict_determinism}")

    if args.fetch_traces and not args.send_traceparent:
        print("  Note: enabling traceparent propagation because --fetch-traces was requested")
        args.send_traceparent = True

    if args.send_traceparent:
        os.environ["SEND_TRACEPARENT"] = "1"
    determinism.strict_text_match = args.strict_determinism

    if not check_health(args.mgmt):
        print(f"\n  ERROR: Management API not reachable at {args.mgmt}")
        print(f"  Make sure KVBM_DEV_MODE=TRUE is set and the port is correct.")
        sys.exit(1)

    print(f"\n  Management API: OK")

    scenario_map = {
        "cold": ("Scenario 1: Cold (CPU cache cleared)", setup_cold),
        "warm": ("Scenario 2: Warm (CPU cache hit)", setup_warm),
    }

    all_results = {}
    for scenario_key in args.scenarios:
        name, setup_fn = scenario_map[scenario_key]
        print(f"\n  ▸ Running: {name}...")

        clear_all_pools(args.mgmt)
        time.sleep(1)

        results = run_scenario(
            scenario_key, args.url, args.model, args.mgmt,
            args.isls, args.n, args.max_tokens, setup_fn,
            concurrency=args.concurrency,
            clear_between=(scenario_key == "cold"),
            skip_cpu_flush=args.skip_cpu_flush,
            seed=args.seed,
            stream=args.stream,
            ttft_mode=args.ttft_mode,
        )
        all_results[scenario_key] = results
        print_table(name, results, args.concurrency)
        sys.stdout.flush()

    if all(k in all_results for k in ("cold", "warm")):
        print_comparison(all_results["cold"], all_results["warm"])

    determinism.summary(dump_path=args.output)
    if args.fetch_traces:
        fetch_tempo_traces(
            determinism.completions,
            args.tempo_url,
            args.output,
            include_raw=args.trace_raw,
            max_wait_seconds=args.trace_fetch_wait,
        )
    print()


if __name__ == "__main__":
    main()
