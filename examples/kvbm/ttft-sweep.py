#!/usr/bin/env python3
"""
TTFT sweep with 2 scenarios and optional concurrency:
  1. Cold   — clear CPU pool before each ISL, but keep CPU lookup enabled
  2. Warm   — CPU pool has data (host cache hit, no G4 needed)

Each scenario sends N requests per ISL. By default the number of distinct
prompt shapes equals concurrency (each parallel slot gets its own variant).
With --prompt-variations V, measurement cycles through only V prompts
(`request_index % V`) while seeding still uses enough parallel lanes to match
concurrency. When V is set, the seed phase sends `concurrency * V` unique
full prompts (variant rotates `s % V`, plus a per-seed instance tag) so the
disk/KV tier is populated for all variation slots at full ISL.

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
import uuid


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
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()) if resp.status == 200 else None
    except Exception as e:
        print(f"  [mgmt] {path} failed: {e}", file=sys.stderr)
        return None


def mgmt_get(mgmt_url: str, path: str) -> dict | None:
    req = urllib.request.Request(f"{mgmt_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()) if resp.status == 200 else None
    except Exception as e:
        print(f"  [mgmt] {path} failed: {e}", file=sys.stderr)
        return None


def parse_mgmt_urls(mgmt_url: str | None) -> list[str]:
    """Split a comma-separated mgmt URL string into a list."""
    if not mgmt_url:
        return []
    return [u.strip() for u in mgmt_url.split(",") if u.strip()]


def clear_cpu_pool(mgmt_url: str):
    for url in parse_mgmt_urls(mgmt_url):
        mgmt_post(url, "/v1/cache/clear", {"pool": "cpu"})


def clear_all_pools(mgmt_url: str):
    for url in parse_mgmt_urls(mgmt_url):
        mgmt_post(url, "/v1/cache/clear_all")


def _extract_pool_info(status: dict, pool: str) -> tuple[int | None, int | None]:
    """Extract (total, available) from a /v1/cache/status response, trying multiple key formats."""
    pools_info = status.get("pools", {})
    aliases = [pool, {"cpu": "host", "gpu": "device"}.get(pool, pool)]
    for key in aliases:
        info = pools_info.get(key)
        if isinstance(info, dict):
            total = info.get("total_blocks") or info.get("total")
            available = info.get("available_blocks") or info.get("available")
            if total is not None:
                return total, available
    return None, None


def _wait_for_drain_single(
    url: str, pool: str = "cpu", max_wall_s: float | None = None
) -> dict:
    """Poll a single management endpoint until all pinned blocks have been offloaded."""
    start = time.perf_counter()
    delay = 0.5
    max_delay = 5.0
    polls = 0
    consecutive_failures = 0
    last_total = None
    last_available = None
    last_status_keys: str | None = None

    while True:
        elapsed = time.perf_counter() - start
        if max_wall_s is not None and elapsed >= max_wall_s:
            raise TimeoutError(
                f"drain for {url} exceeded {max_wall_s}s wall clock "
                f"(pool={pool!r}, last_total={last_total}, last_available={last_available}, "
                f"status_pools={last_status_keys})"
            )

        time.sleep(delay)
        polls += 1
        status = mgmt_get(url, "/v1/cache/status")

        if status:
            consecutive_failures = 0
            try:
                last_status_keys = ",".join(sorted(status.get("pools", {}).keys()))
            except Exception:
                last_status_keys = None
            total, available = _extract_pool_info(status, pool)
            if total is not None:
                last_total = total
                last_available = available

                if available is not None and total == available:
                    break

                pinned = total - (available or 0)
                elapsed = time.perf_counter() - start
                print(f"      drain({url}): {elapsed:.1f}s — {pinned} blocks still pinned ({available}/{total} available)")
            else:
                elapsed = time.perf_counter() - start
                if polls % 10 == 0:
                    print(
                        f"      drain({url}): {elapsed:.1f}s — waiting for pool {pool!r} "
                        f"totals in /v1/cache/status (pools={last_status_keys})"
                    )
        else:
            consecutive_failures += 1
            elapsed = time.perf_counter() - start
            print(f"      drain({url}): {elapsed:.1f}s — no response (attempt {polls}, {consecutive_failures} consecutive failures)")

        delay = min(delay * 1.5, max_delay)

    drain_time = time.perf_counter() - start
    print(f"      drain({url}): complete in {drain_time:.1f}s ({last_available}/{last_total} free) polls={polls}")

    return {
        "drain_time_s": round(drain_time, 3),
        "polls": polls,
        "drained": True,
        "total_blocks": last_total,
        "available_blocks": last_available,
    }


def wait_for_drain(
    mgmt_url: str, pool: str = "cpu", max_wall_s: float | None = None
) -> dict:
    """Poll all management endpoints until all pinned blocks have been offloaded.

    Drains all workers in parallel using threads.
    """
    urls = parse_mgmt_urls(mgmt_url)
    if not urls:
        return {"drain_time_s": 0, "polls": 0, "drained": True}

    if len(urls) == 1:
        return _wait_for_drain_single(urls[0], pool=pool, max_wall_s=max_wall_s)

    start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = {
            executor.submit(_wait_for_drain_single, url, pool, max_wall_s): url
            for url in urls
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    drain_time = time.perf_counter() - start
    total_blocks = sum(r.get("total_blocks") or 0 for r in results)
    available_blocks = sum(r.get("available_blocks") or 0 for r in results)
    print(f"      drain: all {len(urls)} workers drained in {drain_time:.1f}s ({available_blocks}/{total_blocks} free)")

    return {
        "drain_time_s": round(drain_time, 3),
        "polls": sum(r["polls"] for r in results),
        "drained": True,
        "total_blocks": total_blocks,
        "available_blocks": available_blocks,
    }


def drain_and_clear_pool(
    mgmt_url: str,
    pool: str = "cpu",
    h2r_settle: float = 5.0,
    max_drain_wall_s: float | None = None,
) -> dict:
    """Wait for all pinned blocks to be offloaded, then clear the pool safely.

    Never uses force. Polls until all blocks are free (or max_drain_wall_s exceeded).
    After drain, waits h2r_settle seconds for background host-to-remote (disk)
    writes to complete before clearing.
    Retries clear indefinitely with exponential backoff until every worker succeeds.
    Handles multiple management endpoints (comma-separated).
    """
    drain_result = wait_for_drain(mgmt_url, pool=pool, max_wall_s=max_drain_wall_s)

    if h2r_settle > 0:
        print(f"      waiting {h2r_settle:.0f}s for H2R disk writes to settle...")
        time.sleep(h2r_settle)

    urls = parse_mgmt_urls(mgmt_url)
    pending = set(urls)
    delay = 0.5
    max_delay = 5.0
    attempt = 0

    while pending:
        attempt += 1
        still_failed = set()
        for url in pending:
            result = mgmt_post(url, "/v1/cache/clear", {"pool": pool})
            if result is not None:
                print(f"      clear({url}): ok")
            else:
                still_failed.add(url)
                print(f"      clear({url}): failed (attempt {attempt}, retrying...)")
        pending = still_failed
        if pending:
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)

    drain_result["clear_ok"] = True
    return drain_result


def _worker_ready(url: str) -> tuple[bool, str]:
    """Check if a single worker is fully ready (mgmt up + model loaded)."""
    health = mgmt_get(url, "/v1/health")
    if health is None:
        return False, "mgmt unreachable"

    status = mgmt_get(url, "/v1/cache/status")
    if status is None:
        return False, "cache/status unreachable"

    pools = status.get("pools", {})
    has_blocks = any(
        (p.get("total_blocks") or p.get("total", 0)) > 0
        for p in pools.values() if isinstance(p, dict)
    )
    if not has_blocks:
        return False, "model not loaded (no blocks)"

    return True, "ok"


def _frontend_ready(frontend_url: str, expected_workers: int) -> tuple[bool, str]:
    """Check if the Dynamo frontend sees the expected number of generate endpoints."""
    try:
        req = urllib.request.Request(f"{frontend_url}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        return False, "frontend unreachable"

    instances = [i for i in data.get("instances", []) if i.get("endpoint") == "generate"]
    if len(instances) < expected_workers:
        return False, f"frontend sees {len(instances)}/{expected_workers} workers"
    return True, f"{len(instances)} workers registered"


def check_health(mgmt_url: str, frontend_url: str | None = None, wait: bool = True, timeout: float = 600.0) -> bool:
    urls = parse_mgmt_urls(mgmt_url)
    expected_workers = len(urls)

    if not wait:
        for url in urls:
            ok, reason = _worker_ready(url)
            if not ok:
                print(f"  Health check failed for {url}: {reason}")
                return False
        if frontend_url:
            ok, reason = _frontend_ready(frontend_url, expected_workers)
            if not ok:
                print(f"  Frontend check failed: {reason}")
                return False
        return True

    start = time.perf_counter()
    delay = 2.0
    while True:
        all_ok = True
        failed = []
        for url in urls:
            ok, reason = _worker_ready(url)
            if not ok:
                all_ok = False
                failed.append(f"{url} ({reason})")

        if frontend_url and all_ok:
            ok, reason = _frontend_ready(frontend_url, expected_workers)
            if not ok:
                all_ok = False
                failed.append(f"frontend ({reason})")

        if all_ok:
            print(f"  All {expected_workers} workers ready, frontend serving")
            return True
        elapsed = time.perf_counter() - start
        if elapsed >= timeout:
            print(f"  Health check timed out after {elapsed:.0f}s. Failed: {', '.join(failed)}")
            return False
        print(f"  Waiting for {len(failed)}/{expected_workers} workers... ({elapsed:.0f}s)")
        for f in failed:
            print(f"    - {f}")
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)


def get_tuning(mgmt_url: str) -> dict | None:
    urls = parse_mgmt_urls(mgmt_url)
    return mgmt_get(urls[0], "/v1/tuning") if urls else None


def set_tuning(mgmt_url: str, params: dict) -> dict | None:
    urls = parse_mgmt_urls(mgmt_url)
    result = None
    for url in urls:
        result = mgmt_post(url, "/v1/tuning", params)
    return result


# ── Prometheus metrics helpers ───────────────────────────────────────────

def scrape_prometheus(metrics_url: str) -> dict[str, float]:
    """Scrape a Prometheus /metrics endpoint and return a flat dict of metric values."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except Exception:
        return {}
    result = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return result


def parse_histogram_from_prom(
    metrics: dict[str, float], base_name: str, label_filter: dict[str, str]
) -> dict:
    """Extract histogram stats from scraped Prometheus metrics.

    Returns dict with count, sum, avg, and estimated percentiles from bucket boundaries.
    """
    def matches_labels(metric_name: str) -> bool:
        for k, v in label_filter.items():
            if f'{k}="{v}"' not in metric_name:
                return False
        return True

    count_key = None
    sum_key = None
    buckets = []

    for key, val in metrics.items():
        if not matches_labels(key):
            continue
        if f"{base_name}_count" in key:
            count_key = key
        elif f"{base_name}_sum" in key:
            sum_key = key
        elif f"{base_name}_bucket" in key and 'le="' in key:
            le_str = key.split('le="')[1].split('"')[0]
            try:
                le = float(le_str) if le_str != "+Inf" else float("inf")
                buckets.append((le, val))
            except ValueError:
                pass

    count = metrics.get(count_key, 0) if count_key else 0
    total = metrics.get(sum_key, 0) if sum_key else 0
    buckets.sort(key=lambda x: x[0])

    result = {"count": int(count), "sum": total, "avg": total / count if count > 0 else None}

    for pct_name, pct_val in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]:
        target = count * pct_val
        for le, cum_count in buckets:
            if cum_count >= target and le != float("inf"):
                result[pct_name] = le
                break
        else:
            result[pct_name] = None

    return result


def get_transfer_latency_stats(
    metrics_url: str,
) -> dict[str, dict]:
    """Get offload and onboard latency stats from KVBM Prometheus metrics."""
    metrics = scrape_prometheus(metrics_url)
    if not metrics:
        return {}

    base = "kvbm_remote_transfer_latency_seconds"
    result = {
        "offload": parse_histogram_from_prom(metrics, base, {"direction": "offload", "result": "success"}),
        "onboard": parse_histogram_from_prom(metrics, base, {"direction": "onboard", "result": "success"}),
        "offload_failed": parse_histogram_from_prom(metrics, base, {"direction": "offload", "result": "failure"}),
        "onboard_failed": parse_histogram_from_prom(metrics, base, {"direction": "onboard", "result": "failure"}),
    }
    result["offload_bytes"] = metrics.get("kvbm_offload_bytes_remote", 0)
    result["onboard_bytes"] = metrics.get("kvbm_onboard_bytes_remote", 0)
    return result


# ── ClickHouse helpers ────────────────────────────────────────────────

def query_clickhouse(ch_url: str, sql: str) -> str | None:
    """Run a SQL query against ClickHouse HTTP API and return raw text."""
    try:
        encoded = urllib.parse.quote(sql + " FORMAT TabSeparatedWithNames")
        req = urllib.request.Request(f"{ch_url}/?query={encoded}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [clickhouse] query failed: {e}", file=sys.stderr)
        return None


def exec_clickhouse(ch_url: str, sql: str) -> bool:
    """Execute a ClickHouse statement (INSERT, CREATE, etc.)."""
    try:
        data = sql.encode("utf-8")
        req = urllib.request.Request(ch_url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            _ = resp.read()
        return True
    except Exception as e:
        print(f"  [clickhouse] exec failed: {e}", file=sys.stderr)
        return False


def insert_kvbm_run(
    ch_url: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    scenario: str,
    tuning_label: str,
    tuning_config: dict,
    client_config: dict,
    system_config: dict | None,
    isls: list[int],
    concurrency: int,
    n: int,
    results: list[dict],
    model: str,
    name: str = "",
    benchmark: str = "ttft-sweep",
    notes: str = "",
    tags: dict[str, str] | None = None,
) -> bool:
    """Insert a row into otel_traces.kvbm_runs."""
    results_json_str = json.dumps(results).replace("'", "\\'")
    tuning_json = json.dumps(tuning_config).replace("'", "\\'")
    client_json = json.dumps(client_config).replace("'", "\\'")
    system_json = json.dumps(system_config or {}).replace("'", "\\'")
    isls_arr = "[" + ",".join(str(i) for i in isls) + "]"

    seed_ttft = 0.0
    p50 = p95 = mean = 0.0
    if results:
        r = results[0]
        seed_ttft = (r.get("offload_seed_ttft") or 0) * 1000
        p50 = (r.get("p50") or 0) * 1000
        p95 = (r.get("p95") or 0) * 1000
        mean = (r.get("mean") or 0) * 1000

    tags_map = "map()"
    if tags:
        pairs = ",".join(f"'{k}','{v}'" for k, v in tags.items())
        tags_map = f"map({pairs})"

    sql = f"""INSERT INTO otel_traces.kvbm_runs (
        run_id, started_at, finished_at, name, benchmark, scenario,
        tuning_label, tuning_config, client_config, system_config,
        isls, concurrency, n,
        results_json, offload_seed_ttft_ms, onboard_p50_ms,
        onboard_p95_ms, onboard_mean_ms, model, notes, tags
    ) VALUES (
        '{run_id}', '{started_at}', '{finished_at}', '{name}', '{benchmark}',
        '{scenario}', '{tuning_label}', '{tuning_json}', '{client_json}',
        '{system_json}', {isls_arr}, {concurrency}, {n},
        '{results_json_str}', {seed_ttft:.1f}, {p50:.1f},
        {p95:.1f}, {mean:.1f}, '{model}', '{notes}', {tags_map}
    )"""
    return exec_clickhouse(ch_url, sql)


def _build_system_config(args) -> dict:
    """Collect system/runtime config for the system_config column."""
    cfg: dict = {}

    # KVBM tuning from management API
    if args.mgmt:
        t = get_tuning(args.mgmt)
        if t and "tuning" in t:
            cfg["kvbm_tuning"] = t["tuning"]

    # Key env vars
    for key in [
        "DYN_KVBM_CPU_CACHE_GB", "DYN_KVBM_DISK_CACHE_GB",
        "DYN_KVBM_REMOTE_STORAGE_TYPE", "DYN_KVBM_REMOTE_DISK_USE_GDS",
        "DYN_KVBM_REMOTE_DISK_O_DIRECT", "DYN_KVBM_REMOTE_TRANSFER_CONTEXT_POOL_SIZE",
        "DYN_KVBM_G4_MAX_REMOTE_INFLIGHT", "DYN_KVBM_G4_DRAIN_QUEUE_CAP",
        "DYN_KVBM_REMOTE_DISK_FD_CACHE_MAX_ENTRIES",
        "TP_SIZE", "GPU_MEMORY_UTILIZATION",
    ]:
        val = os.environ.get(key)
        if val:
            cfg[key] = val

    # GPU info (best-effort)
    try:
        import subprocess
        gpu_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,count", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if gpu_out.returncode == 0:
            lines = [l.strip() for l in gpu_out.stdout.strip().split("\n") if l.strip()]
            if lines:
                parts = lines[0].split(", ")
                cfg["gpu_name"] = parts[0] if len(parts) > 0 else ""
                cfg["gpu_memory_mb"] = parts[1] if len(parts) > 1 else ""
                cfg["gpu_count"] = str(len(lines))
    except Exception:
        pass

    # CPU info (best-effort)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cfg["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    cfg["ram_gb"] = str(round(kb / (1024 * 1024)))
                    break
    except Exception:
        pass

    return cfg


def clickhouse_span_stats(ch_url: str, since_seconds: int = 300) -> list[dict]:
    """Get per-span-name P50/P95/P99/avg from ClickHouse OTEL spans."""
    sql = f"""
SELECT
    SpanName,
    count() AS ops,
    round(quantile(0.50)(Duration / 1000000), 2) AS p50_ms,
    round(quantile(0.95)(Duration / 1000000), 2) AS p95_ms,
    round(quantile(0.99)(Duration / 1000000), 2) AS p99_ms,
    round(avg(Duration / 1000000), 2) AS avg_ms,
    round(min(Duration / 1000000), 2) AS min_ms,
    round(max(Duration / 1000000), 2) AS max_ms
FROM otel_traces.otel_spans
WHERE Timestamp > now() - INTERVAL {since_seconds} SECOND
  AND SpanName LIKE 'kvbm.%'
GROUP BY SpanName
ORDER BY avg_ms DESC
"""
    raw = query_clickhouse(ch_url, sql)
    if not raw or not raw.strip():
        return []
    lines = raw.strip().split("\n")
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        vals = line.split("\t")
        row = dict(zip(headers, vals))
        rows.append(row)
    return rows


def print_clickhouse_span_table(rows: list[dict], label: str = ""):
    """Print ClickHouse span stats using tabulate."""
    if not rows:
        return
    try:
        from tabulate import tabulate
    except ImportError:
        return

    title = f"  ClickHouse Span Analytics"
    if label:
        title += f" [{label}]"

    headers = ["Span", "Ops", "Avg ms", "P50 ms", "P95 ms", "P99 ms", "Min ms", "Max ms"]
    table_rows = []
    for r in rows:
        table_rows.append([
            r.get("SpanName", ""),
            r.get("ops", ""),
            r.get("avg_ms", ""),
            r.get("p50_ms", ""),
            r.get("p95_ms", ""),
            r.get("p99_ms", ""),
            r.get("min_ms", ""),
            r.get("max_ms", ""),
        ])
    print(f"\n{title}")
    print(tabulate(table_rows, headers=headers, tablefmt="simple_outline", stralign="right"))


def build_tuning_matrix(matrix_spec: dict) -> list[dict]:
    """Expand a matrix spec into a list of tuning configurations.

    Input format (each key maps to a list of values to sweep):
        {"flush_batch_size": [512, 1024], "g4_pipeline_chunk_size": [16, 64]}

    Output: cartesian product of all combinations:
        [{"flush_batch_size": 512, "g4_pipeline_chunk_size": 16},
         {"flush_batch_size": 512, "g4_pipeline_chunk_size": 64},
         {"flush_batch_size": 1024, "g4_pipeline_chunk_size": 16},
         {"flush_batch_size": 1024, "g4_pipeline_chunk_size": 64}]
    """
    import itertools

    keys = list(matrix_spec.keys())
    value_lists = [matrix_spec[k] if isinstance(matrix_spec[k], list) else [matrix_spec[k]] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos


SERVER_TUNING_PARAMS = {
    "transfer_batch_size", "max_concurrent_transfers",
    "flush_batch_size", "g4_pipeline_chunk_size", "g4_transfer_timeout_secs",
}

CLIENT_SWEEP_PARAMS = {"concurrency", "n", "isls", "prompt_variations"}


def split_matrix_config(config: dict) -> tuple[dict, dict]:
    """Split a matrix config into (server_tuning, client_overrides)."""
    server = {k: v for k, v in config.items() if k in SERVER_TUNING_PARAMS}
    client = {k: v for k, v in config.items() if k in CLIENT_SWEEP_PARAMS}
    return server, client


def _normalize_isls_list(isls, fallback: list[int]) -> list[int]:
    if isinstance(isls, (int, float)):
        return [int(isls)]
    if isinstance(isls, list):
        return [int(x) for x in isls]
    return fallback


def _effective_client_banner(
    tuning_configs: list, args,
) -> tuple[list[int], int, int, str]:
    """ISLs / n / concurrency for startup banner (first matrix row + CLI fallback)."""
    if not tuning_configs or tuning_configs == [{}]:
        return args.isls, args.n, args.concurrency, ""
    _, cp0 = split_matrix_config(tuning_configs[0])
    isls = _normalize_isls_list(cp0.get("isls", args.isls), args.isls)
    n = int(cp0.get("n", args.n))
    c = int(cp0.get("concurrency", args.concurrency))
    note = ""
    if len(tuning_configs) > 1:
        for cfg in tuning_configs[1:]:
            _, cp = split_matrix_config(cfg)
            o_isls = _normalize_isls_list(cp.get("isls", isls), isls)
            if (
                o_isls != isls
                or int(cp.get("n", n)) != n
                or int(cp.get("concurrency", c)) != c
            ):
                note = " (config 1 shown; later rows may override n/c/isls)"
                break
    return isls, n, c, note


def expand_name_template(template: str, **kwargs) -> str:
    """Expand a benchmark name template with config values.

    Built-in placeholders: {uuid} (short 8-char), {salt} (run salt).
    Unresolved placeholders are left as-is (no KeyError).
    """
    if not template:
        return ""
    kwargs.setdefault("uuid", uuid.uuid4().hex[:8])
    kwargs.setdefault("salt", _RUN_SALT)
    try:
        return template.format_map(collections.defaultdict(str, **kwargs))
    except Exception:
        return template


import collections


SHORT_PARAM_NAMES = {
    "transfer_batch_size": "batch",
    "max_concurrent_transfers": "conc",
    "flush_batch_size": "flush",
    "g4_pipeline_chunk_size": "chunk",
    "g4_transfer_timeout_secs": "timeout",
    "concurrency": "c",
    "n": "n",
    "isls": "isl",
    "prompt_variations": "pv",
}


def _format_param_value(k, v):
    if k == "isls" and isinstance(v, list):
        return "+".join(fmt_isl(i) for i in v)
    return v


def tuning_label(config: dict) -> str:
    """Full label for a tuning configuration (used for logging/display)."""
    parts = []
    for k, v in sorted(config.items()):
        parts.append(f"{SHORT_PARAM_NAMES.get(k, k)}={_format_param_value(k, v)}")
    return " ".join(parts)


def compact_tuning_labels(configs: list[dict]) -> list[str]:
    """Compute short labels showing only parameters that vary across configs."""
    if len(configs) <= 1:
        return [tuning_label(cfg) if cfg else "baseline" for cfg in configs]

    all_keys = sorted({k for cfg in configs for k in cfg})
    varying_keys = [
        k for k in all_keys
        if len({str(cfg.get(k)) for cfg in configs}) > 1
    ]
    if not varying_keys:
        varying_keys = all_keys

    labels = []
    for cfg in configs:
        parts = []
        for k in varying_keys:
            v = cfg.get(k)
            if v is not None:
                parts.append(f"{SHORT_PARAM_NAMES.get(k, k)}={_format_param_value(k, v)}")
        labels.append(" ".join(parts) if parts else tuning_label(cfg))
    return labels


# ── Request helpers ──────────────────────────────────────────────────────

# Per-run salt so each sweep invocation produces unique sequence hashes,
# forcing fresh offload+onboard even when ISL/variant match a prior run.
# Override with --prompt-salt for reproducible prompts across runs.
_RUN_SALT: str = os.urandom(4).hex()


def set_run_salt(salt: str | None):
    global _RUN_SALT
    if salt is not None:
        _RUN_SALT = salt


def make_prompt(num_tokens: int, variant: int = 0, instance: int | None = None) -> list[dict]:
    """Build a chat message of approximately `num_tokens` tokens.

    Each `variant` value produces a deterministically different prompt of the
    same length. A per-run random salt ensures sequence hashes differ across
    sweep invocations so disk-cached data from prior runs is not reused.

    When ``instance`` is set, the first token embeds it so two prompts with the
    same variant still produce distinct full sequences (used for multi-seed
    population with a small number of prompt variations).
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

    sentence = " ".join(rotated)
    words = sentence.split()
    salt0 = f"{_RUN_SALT}-i{instance}" if instance is not None else _RUN_SALT
    prompt_words = [salt0]
    while len(prompt_words) < num_tokens:
        prompt_words.extend(words)
    content = " ".join(prompt_words[:num_tokens])
    return [{"role": "user", "content": content}]


_BAGGAGE: str = ""


def set_baggage(baggage: str):
    global _BAGGAGE
    _BAGGAGE = baggage


def send_request(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1,
    seed: int | None = None,
    stream: bool = False,
    ttft_mode: str = "either",
    send_note: str | None = None,
) -> dict:
    traceparent = _make_traceparent()
    trace_id = _trace_id_from_traceparent(traceparent)
    note = send_note if send_note else "send"
    tp_on = os.environ.get("SEND_TRACEPARENT", "0") == "1"
    print(
        f"    [{note}] trace_id={trace_id} traceparent={tp_on} sending...",
        flush=True,
    )
    headers = {"Content-Type": "application/json"}
    if tp_on:
        headers["traceparent"] = traceparent
    if _BAGGAGE:
        headers["baggage"] = _BAGGAGE
        headers["tracestate"] = f"kvbm={urllib.parse.quote(_BAGGAGE, safe='')}"
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
        # Prefer first token in reasoning or content; fall back to first SSE `data:` line.
        # GPT-OSS (and others) may emit initial chunks without delta.reasoning/content filled.
        candidates = [v for v in (first_reasoning_ttft, first_content_ttft) if v is not None]
        if candidates:
            return min(candidates)
        return first_event_ttft
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

    end_mono = time.perf_counter()
    ttft = _select_stream_ttft(first_event_ttft, first_reasoning_ttft, first_content_ttft, ttft_mode)
    if ttft is None:
        ttft = first_event_ttft
    if ttft is None:
        # Empty or non-standard stream (no data: lines) — use full read duration.
        ttft = end_mono - start

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


def _fmt_throughput(total_bytes: float, total_seconds: float) -> str:
    if total_seconds <= 0 or total_bytes <= 0:
        return "-"
    gbps = total_bytes / (1024**3) / total_seconds
    if gbps >= 1.0:
        return f"{gbps:.1f}G/s"
    mbps = total_bytes / (1024**2) / total_seconds
    if mbps >= 1.0:
        return f"{mbps:.0f}M/s"
    return f"{mbps:.1f}M/s"


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
    flush_wait: float = 10.0,
    variant_offset: int = 0,
    benchmark_name: str = "",
    tuning_cfg: dict | None = None,
    prompt_variations: int | None = None,
    max_drain_wall_s: float | None = None,
) -> list[dict]:
    """Run a scenario: for each ISL, call setup_fn then send n requests.

    When concurrency > 1, requests are sent in parallel using a thread pool.
    When clear_between is True, CPU pool is cleared after each request
    (drain waits for all offloads to complete before clearing).
    When variant_offset > 0, prompt variants are shifted so each tuning config
    produces unique sequence hashes, forcing fresh offload+onboard cycles.

    When ``prompt_variations`` is set to V, only V distinct prompts are used
    during measurement (``request_index % V``). The seed phase then issues
    ``concurrency * V`` unique prompts (see ``make_prompt`` ``instance``) in
    waves of up to ``concurrency`` parallel setup calls. When unset, behavior
    matches the original sweep: ``variant_count == concurrency`` and the seed
    phase issues one parallel wave of ``concurrency`` seeds.
    """
    results = []

    for isl in isls:
        isl_run_id = str(uuid.uuid4())

        cfg = tuning_cfg or {}
        expanded_name = expand_name_template(
            benchmark_name,
            uuid=isl_run_id[:8],
            isl=fmt_isl(isl), chunk=cfg.get("g4_pipeline_chunk_size", ""),
            batch=cfg.get("transfer_batch_size", ""), c=cfg.get("concurrency", concurrency),
            n=cfg.get("n", n), scenario=name, config=tuning_label(cfg) if cfg else "baseline",
        )

        baggage_parts = [f"run_id={isl_run_id}", f"run_salt={_RUN_SALT}", f"isl={isl}"]
        if expanded_name:
            baggage_parts.append(f"benchmark={urllib.parse.quote(expanded_name)}")
        set_baggage(",".join(baggage_parts))

        if prompt_variations is not None:
            variant_count = max(1, int(prompt_variations))
        else:
            variant_count = max(1, concurrency)
        all_messages = [make_prompt(isl, variant=i + variant_offset) for i in range(variant_count)]

        setup_timing = {}
        seed_trace_ids = []
        if setup_fn is not None:
            if prompt_variations is not None:
                v = variant_count
                num_seeds = concurrency * v
                seed_indices = list(range(num_seeds))
            else:
                num_seeds = min(concurrency, len(all_messages))
                seed_indices = list(range(num_seeds))

            if len(seed_indices) <= 1:
                s = seed_indices[0]
                if prompt_variations is not None:
                    pv = s % variant_count
                    msgs = make_prompt(isl, variant=pv + variant_offset, instance=s)
                else:
                    msgs = all_messages[s]
                st = setup_fn(
                    mgmt_url, url, model, msgs, isl, skip_cpu_flush, seed, flush_wait,
                    seed_index=s + 1, seed_total=len(seed_indices),
                    max_drain_wall_s=max_drain_wall_s,
                )
                if isinstance(st, dict):
                    setup_timing = st
                    if st.get("seed_trace_id"):
                        seed_trace_ids.append(st["seed_trace_id"])
            else:
                workers = min(concurrency, len(seed_indices))
                for batch_start in range(0, len(seed_indices), workers):
                    batch = seed_indices[batch_start : batch_start + workers]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                        futures = {}
                        for s in batch:
                            if prompt_variations is not None:
                                pv = s % variant_count
                                msgs = make_prompt(isl, variant=pv + variant_offset, instance=s)
                            else:
                                msgs = all_messages[s]
                            fut = pool.submit(
                                setup_fn,
                                mgmt_url,
                                url,
                                model,
                                msgs,
                                isl,
                                skip_cpu_flush,
                                seed,
                                flush_wait,
                                s + 1,
                                len(seed_indices),
                                max_drain_wall_s=max_drain_wall_s,
                            )
                            futures[fut] = s
                        for fut in concurrent.futures.as_completed(futures):
                            st = fut.result()
                            if isinstance(st, dict):
                                setup_timing = st
                                if st.get("seed_trace_id"):
                                    seed_trace_ids.append(st["seed_trace_id"])
            print(
                f"    [{name}] ISL={fmt_isl(isl)}: all {len(seed_indices)} seed request(s) complete, draining..."
            )

            if not skip_cpu_flush and mgmt_url:
                drain_and_clear_pool(
                    mgmt_url, pool="cpu", max_drain_wall_s=max_drain_wall_s
                )

        ttfts = []
        trace_ids = []
        errors = 0

        if concurrency <= 1:
            for i in range(n):
                variant = i % variant_count
                sn = f"{name} ISL={fmt_isl(isl)} run {i + 1}/{n} prompt=v{variant}"
                if stream:
                    r = send_request(
                        url,
                        model,
                        all_messages[variant],
                        max_tokens,
                        seed=seed,
                        stream=True,
                        ttft_mode=ttft_mode,
                        send_note=sn,
                    )
                else:
                    r = send_request(
                        url,
                        model,
                        all_messages[variant],
                        max_tokens,
                        seed=seed,
                        send_note=sn,
                    )
                if r["error"]:
                    errors += 1
                    print(
                        f"    [{name}] ISL={fmt_isl(isl)} run {i+1}/{n}: ERROR {r['error']} trace_id={r.get('trace_id')}",
                        file=sys.stderr,
                    )
                else:
                    ttfts.append(r["ttft"])
                    if r.get("trace_id"):
                        trace_ids.append(r["trace_id"])
                    print(
                        f"    [{name}] ISL={fmt_isl(isl)} run {i+1}/{n}: TTFT={fmt_time(r['ttft'])} trace_id={r.get('trace_id')} prompt=v{variant}"
                    )
                    determinism.record(
                        name,
                        isl,
                        r["completion"],
                        prompt_key=f"v{variant}",
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
                if clear_between and not skip_cpu_flush and mgmt_url:
                    drain_and_clear_pool(
                        mgmt_url, pool="cpu", max_drain_wall_s=max_drain_wall_s
                    )
        else:
            req_counter = 0
            for batch_start in range(0, n, concurrency):
                batch_size = min(concurrency, n - batch_start)
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    future_meta = {}
                    for j in range(batch_size):
                        i = batch_start + j
                        variant = i % variant_count
                        sn = f"{name} ISL={fmt_isl(isl)} run {i + 1}/{n} prompt=v{variant}"
                        fut = pool.submit(
                            send_request,
                            url,
                            model,
                            all_messages[variant],
                            max_tokens,
                            seed,
                            stream,
                            ttft_mode,
                            sn,
                        )
                        future_meta[fut] = (i, variant)
                    for fut in concurrent.futures.as_completed(future_meta):
                        i, variant = future_meta[fut]
                        req_counter += 1
                        r = fut.result()
                        if r["error"]:
                            errors += 1
                            print(
                                f"    [{name}] ISL={fmt_isl(isl)} req {req_counter}/{n}: ERROR {r['error']} trace_id={r.get('trace_id')}",
                                file=sys.stderr,
                            )
                        else:
                            ttfts.append(r["ttft"])
                            if r.get("trace_id"):
                                trace_ids.append(r["trace_id"])
                            print(
                                f"    [{name}] ISL={fmt_isl(isl)} req {req_counter}/{n}: TTFT={fmt_time(r['ttft'])} trace_id={r.get('trace_id')} prompt=v{variant}"
                            )
                            determinism.record(
                                name,
                                isl,
                                r["completion"],
                                prompt_key=f"v{variant}",
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
                if clear_between and not skip_cpu_flush and mgmt_url:
                    drain_and_clear_pool(
                        mgmt_url, pool="cpu", max_drain_wall_s=max_drain_wall_s
                    )

        determinism.check_isl(name, isl)

        row = {
            "run_id": isl_run_id,
            "name": expanded_name,
            "isl": isl,
            "n": len(ttfts),
            "errors": errors,
            "concurrency": concurrency,
            "min": min(ttfts) if ttfts else None,
            "p50": statistics.median(ttfts) if ttfts else None,
            "p95": sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) >= 2 else (ttfts[0] if ttfts else None),
            "max": max(ttfts) if ttfts else None,
            "mean": statistics.mean(ttfts) if ttfts else None,
            "offload_seed_ttft": setup_timing.get("offload_seed_ttft"),
            "offload_total": setup_timing.get("offload_total"),
            "trace_ids": seed_trace_ids + trace_ids,
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
    flush_wait: float = 10.0,
    seed_index: int | None = None,
    seed_total: int | None = None,
    max_drain_wall_s: float | None = None,
) -> dict:
    """Send a seed request to populate KV context. No drain or clear here.

    max_drain_wall_s is accepted for API parity with setup_warm / executor.submit;
    post-seed drain uses the value passed to run_scenario instead.
    """
    tag = ""
    if seed_index is not None and seed_total is not None:
        tag = f" [{seed_index}/{seed_total}]"
    seed_note = f"setup_cold ISL={fmt_isl(isl)}{tag}"
    offload_start = time.perf_counter()
    r = send_request(url, model, messages, 1, seed=seed, send_note=seed_note)
    offload_ttft = time.perf_counter() - offload_start
    err = r.get("error")
    err_note = f" ERROR={err}" if err else ""
    print(
        f"    [setup_cold] ISL={fmt_isl(isl)}{tag}: seed request TTFT={fmt_time(offload_ttft)} "
        f"trace_id={r.get('trace_id')}{err_note}"
    )
    return {
        "offload_seed_ttft": offload_ttft,
        "offload_total": time.perf_counter() - offload_start,
        "seed_trace_id": r.get("trace_id"),
    }


def setup_warm(
    mgmt_url,
    url,
    model,
    messages,
    isl,
    skip_cpu_flush: bool = False,
    seed: int | None = None,
    flush_wait: float = 10.0,
    seed_index: int | None = None,
    seed_total: int | None = None,
    max_drain_wall_s: float | None = None,
) -> dict:
    """Scenario 2: ensure CPU pool is populated (host cache hit).

    Sends a seed request, drains and clears safely, then re-populates.
    """
    tag = ""
    if seed_index is not None and seed_total is not None:
        tag = f" [{seed_index}/{seed_total}]"
    seed_note = f"setup_warm ISL={fmt_isl(isl)}{tag}"
    offload_start = time.perf_counter()
    r = send_request(url, model, messages, 1, seed=seed, send_note=seed_note)
    offload_ttft = time.perf_counter() - offload_start
    print(
        f"    [setup_warm] ISL={fmt_isl(isl)}{tag}: seed request TTFT={fmt_time(offload_ttft)} trace_id={r.get('trace_id')}"
    )

    if mgmt_url:
        drain_and_clear_pool(
            mgmt_url, pool="cpu", max_drain_wall_s=max_drain_wall_s
        )
        send_request(
            url,
            model,
            messages,
            1,
            seed=seed,
            send_note=f"{seed_note} repopulate",
        )
        time.sleep(2)
    else:
        time.sleep(flush_wait)

    return {"offload_seed_ttft": offload_ttft, "seed_trace_id": r.get("trace_id")}


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


def print_matrix_comparison(
    matrix_results: dict[str, dict[str, list[dict]]],
    tuning_configs: list[dict] | None = None,
    prom_snapshots: dict[str, dict] | None = None,
):
    """Print a single comparison table per scenario with config params as columns.

    matrix_results: {tuning_label: {scenario_key: [row_per_isl]}}
    tuning_configs: original config dicts (used to extract individual param columns)
    prom_snapshots: {tuning_label: {offload_delta: {...}, onboard_delta: {...}}}
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("\n  WARNING: 'tabulate' not installed. Install with: pip install tabulate")
        print("  Skipping matrix comparison tables.")
        return

    labels = list(matrix_results.keys())
    if not labels:
        return

    configs = tuning_configs if tuning_configs and len(tuning_configs) == len(labels) else None
    has_prom = prom_snapshots and any(prom_snapshots.get(l) for l in labels)

    param_columns: list[str] = []
    if configs:
        all_keys = sorted({k for cfg in configs for k in cfg})
        param_columns = [
            k for k in all_keys
            if len({str(cfg.get(k)) for cfg in configs}) > 1
        ]
        if not param_columns:
            param_columns = all_keys

    for scenario_key in next(iter(matrix_results.values())).keys():
        param_headers = [SHORT_PARAM_NAMES.get(k, k) for k in param_columns]
        headers = param_headers + [
            "ISL", "N", "Min", "P50", "P95", "Max", "Mean", "Seed", "Err",
        ]
        if has_prom:
            headers += ["Off#", "Off Avg", "Off Tput", "On#", "On Avg", "On Tput"]

        first_results = matrix_results[labels[0]].get(scenario_key, [])
        has_multiple = len(labels) >= 2
        if has_multiple:
            headers.append("P50 vs #1")

        rows = []
        for cfg_idx, label in enumerate(labels):
            results = matrix_results[label].get(scenario_key, [])
            cfg = configs[cfg_idx] if configs else {}
            snap = prom_snapshots.get(label, {}) if prom_snapshots else {}
            for isl_idx, r in enumerate(results):
                row = [_format_param_value(k, cfg.get(k, "")) for k in param_columns]
                row += [
                    fmt_isl(r["isl"]),
                    r["n"],
                    fmt_time(r["min"]),
                    fmt_time(r["p50"]),
                    fmt_time(r["p95"]),
                    fmt_time(r["max"]),
                    fmt_time(r["mean"]),
                    fmt_time(r.get("offload_seed_ttft")),
                    r["errors"],
                ]
                if has_prom:
                    off = snap.get("offload_delta", {})
                    on = snap.get("onboard_delta", {})
                    row += [
                        off.get("count", 0),
                        off.get("avg", "-"),
                        off.get("throughput", "-"),
                        on.get("count", 0),
                        on.get("avg", "-"),
                        on.get("throughput", "-"),
                    ]
                if has_multiple:
                    base_p50 = first_results[isl_idx]["p50"] if isl_idx < len(first_results) else None
                    cur_p50 = r["p50"]
                    if cfg_idx == 0:
                        row.append("-")
                    elif base_p50 and cur_p50 and cur_p50 > 0:
                        row.append(f"{base_p50 / cur_p50:.2f}x")
                    else:
                        row.append("-")
                rows.append(row)

        print(f"\n  Matrix Results: {scenario_key}")
        print(tabulate(rows, headers=headers, tablefmt="simple_outline", stralign="right"))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TTFT sweep")
    parser.add_argument("--url", default="http://localhost:19000", help="vLLM base URL")
    parser.add_argument("--mgmt", default=None, help="KVBM management URL (omit to skip all CPU pool management)")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument(
        "--isls", type=int, nargs="+",
        default=[1000, 5000, 10000, 20000, 40000, 60000, 80000, 100000, 120000],
    )
    parser.add_argument("-n", type=int, default=10, help="Requests per ISL per scenario")
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                        help="Concurrent requests per ISL (default: 1 = sequential)")
    parser.add_argument(
        "--prompt-variations",
        type=int,
        default=None,
        metavar="V",
        help="Use only V distinct prompts during measurement (round-robin by request index). "
             "When set, the seed phase sends concurrency*V unique full prompts in waves of "
             "size concurrency (default: unset = legacy, one prompt shape per lane, V=concurrency).",
    )
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
    parser.add_argument(
        "--skip-clear-all",
        action="store_true",
        help="Skip clear_all_pools at scenario start (avoids cancel token bug with multi-instance KVBM)",
    )
    parser.add_argument(
        "--flush-wait",
        type=float,
        default=10.0,
        help="Seconds to wait for offload to complete before clearing CPU pool (default: 10)",
    )
    parser.add_argument(
        "--max-drain-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="Abort if a CPU pool drain (waiting for pinned blocks to reach 0) exceeds SEC "
        "seconds. Default: no limit — a stuck pin can hang the sweep forever after seeding.",
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
    parser.add_argument(
        "--tuning-matrix",
        default=None,
        help='JSON file or inline JSON with tuning param matrix, e.g. '
             '\'{"flush_batch_size": [512, 1024], "g4_pipeline_chunk_size": [16, 64]}\'',
    )
    parser.add_argument(
        "--metrics-url",
        default="http://localhost:6880/metrics",
        help="KVBM Prometheus metrics URL for offload/onboard latency histograms (default: http://localhost:6880/metrics)",
    )
    parser.add_argument(
        "--clickhouse-url",
        default="http://localhost:8123",
        help="ClickHouse HTTP API URL for SQL trace analytics and run storage (default: http://localhost:8123)",
    )
    parser.add_argument(
        "--prompt-salt",
        default=None,
        help="Fixed salt for prompt generation (default: random per run). "
             "Use a fixed value for reproducible prompts across runs.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Benchmark name template for labeling runs. Supports placeholders: "
             "{isl}, {chunk}, {batch}, {c}, {n}, {scenario}, {config}. "
             "Example: --name 'VAST TTFT - {isl} chunk={chunk}' "
             "produces 'VAST TTFT - 120K chunk=32' per config.",
    )
    args = parser.parse_args()

    if args.prompt_variations is not None and args.prompt_variations < 1:
        print("  ERROR: --prompt-variations must be >= 1 when set")
        sys.exit(1)

    set_run_salt(args.prompt_salt)

    tuning_configs: list = [{}]
    if args.tuning_matrix:
        try:
            if os.path.isfile(args.tuning_matrix):
                with open(args.tuning_matrix) as f:
                    matrix_spec = json.load(f)
            else:
                matrix_spec = json.loads(args.tuning_matrix)

            if isinstance(matrix_spec, list):
                tuning_configs = matrix_spec
            else:
                tuning_configs = build_tuning_matrix(matrix_spec)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"\n  ERROR: Failed to parse --tuning-matrix: {e}")
            sys.exit(1)

    banner_isls, banner_n, banner_c, banner_mtx_note = _effective_client_banner(
        tuning_configs, args
    )

    print(f"\n  TTFT Sweep")
    print(f"  Model:       {args.model}")
    print(f"  Endpoint:    {args.url}")
    print(f"  Mgmt:        {args.mgmt or '(disabled)'}")
    print(
        f"  ISLs:        {', '.join(fmt_isl(i) for i in banner_isls)}{banner_mtx_note}"
    )
    print(f"  N/ISL:       {banner_n}")
    print(f"  Concurrency: {banner_c}")
    print(
        f"  Prompt var.: {args.prompt_variations if args.prompt_variations is not None else '(lanes = concurrency)'}"
    )
    print(f"  Scenarios:   {', '.join(args.scenarios)}")
    print(f"  Streaming:   {args.stream}")
    print(f"  TTFT mode:   {args.ttft_mode if args.stream else 'full-response'}")
    print(f"  Skip flush:  {args.skip_cpu_flush}")
    print(f"  Flush wait:  {args.flush_wait:.0f}s")
    if args.max_drain_seconds is not None:
        print(f"  Max drain:   {args.max_drain_seconds:.0f}s (CPU pool drain watchdog)")
    else:
        print("  Max drain:   (none — drain waits until all pins drop; can hang if KVBM stuck)")
    print(f"  Traceparent: {args.send_traceparent}")
    print(f"  Fetch traces: {args.fetch_traces}")
    if args.fetch_traces:
        print(f"  Tempo URL:   {args.tempo_url}")
    print(f"  Seed:        {args.seed if args.seed is not None else '-'}")
    print(f"  Strict det:  {args.strict_determinism}")

    if tuning_configs != [{}]:
        print(f"\n  Tuning matrix: {len(tuning_configs)} configurations")
        for i, cfg in enumerate(tuning_configs):
            print(f"    [{i+1}] {tuning_label(cfg)}")

    if args.fetch_traces and not args.send_traceparent:
        print("  Note: enabling traceparent propagation because --fetch-traces was requested")
        args.send_traceparent = True

    if args.send_traceparent:
        os.environ["SEND_TRACEPARENT"] = "1"
    determinism.strict_text_match = args.strict_determinism

    if args.mgmt:
        if not check_health(args.mgmt, frontend_url=args.url):
            print(f"\n  ERROR: Workers not ready at {args.mgmt}")
            print(
                "  Make sure KVBM_DEV_MODE=TRUE, the stack is up, and model is loaded. "
                "Connection refused on some ports usually means those workers are not running "
                "or your --mgmt list has more URLs than running replicas (e.g. 8 URLs vs 4x compose).",
            )
            sys.exit(1)
        print(f"\n  Management API: OK")
        tuning_status = get_tuning(args.mgmt)
        if tuning_status and "tuning" in tuning_status:
            print(f"  Current tuning: {tuning_status['tuning']}")
    else:
        print(f"\n  Management API: disabled (CPU pool operations skipped)")

    scenario_map = {
        "cold": ("Scenario 1: Cold (CPU cache cleared)", setup_cold),
        "warm": ("Scenario 2: Warm (CPU cache hit)", setup_warm),
    }

    # matrix_results[tuning_label][scenario_key] = [row_per_isl]
    matrix_results: dict[str, dict[str, list[dict]]] = {}
    original_tuning = None
    if args.mgmt and len(tuning_configs) > 1:
        t = get_tuning(args.mgmt)
        if t and "tuning" in t:
            original_tuning = t["tuning"]

    # prom_snapshots[label] = {"before": {...}, "after": {...}, "delta": {...}}
    prom_snapshots: dict[str, dict] = {}

    for cfg_idx, tuning_cfg in enumerate(tuning_configs):
        import datetime as _dt
        cfg_started_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        label = tuning_label(tuning_cfg) if tuning_cfg else "baseline"
        server_params, client_params = split_matrix_config(tuning_cfg)

        if args.prompt_salt is None:
            set_run_salt(os.urandom(4).hex())
        print(f"  Prompt salt: {_RUN_SALT}")

        # Reset Prometheus histograms so per-config metrics are clean
        if args.mgmt and args.metrics_url:
            for url in parse_mgmt_urls(args.mgmt):
                mgmt_post(url, "/v1/metrics/reset")

        # Baggage is updated per-ISL inside run_scenario with a unique run_id.
        # Set a config-level fallback here for seed requests.
        baggage_parts = [f"run_salt={_RUN_SALT}", f"config_idx={cfg_idx}"]
        if args.name:
            baggage_parts.append(f"benchmark={urllib.parse.quote(args.name)}")
        if label != "baseline":
            baggage_parts.append(f"config={urllib.parse.quote(label)}")
        set_baggage(",".join(baggage_parts))

        # Apply server-side tuning via management API
        if server_params and args.mgmt:
            print(f"\n  ━━━ Tuning config [{cfg_idx+1}/{len(tuning_configs)}]: {label} ━━━")
            result = set_tuning(args.mgmt, server_params)
            if result and "tuning" in result:
                print(f"  Applied server: {result['tuning']}")
            elif result and "errors" in result:
                print(f"  WARNING: {result['errors']}")
            time.sleep(1)
        elif len(tuning_configs) > 1:
            print(f"\n  ━━━ Tuning config [{cfg_idx+1}/{len(tuning_configs)}]: {label} ━━━")

        # Resolve client-side overrides (fall back to CLI args)
        cfg_concurrency = int(client_params.get("concurrency", args.concurrency))
        cfg_n = int(client_params.get("n", args.n))
        cfg_isls = client_params.get("isls", args.isls)
        if isinstance(cfg_isls, (int, float)):
            cfg_isls = [int(cfg_isls)]
        elif isinstance(cfg_isls, list):
            cfg_isls = [int(i) for i in cfg_isls]

        cfg_prompt_variations = client_params.get("prompt_variations", args.prompt_variations)
        if cfg_prompt_variations is not None:
            cfg_prompt_variations = int(cfg_prompt_variations)
            if cfg_prompt_variations < 1:
                print("  ERROR: prompt_variations in tuning matrix must be >= 1")
                sys.exit(1)

        ch_client_config = dict(client_params)
        if cfg_prompt_variations is not None:
            ch_client_config["prompt_variations"] = cfg_prompt_variations

        if client_params:
            pv_note = (
                f" prompt_variations={cfg_prompt_variations}"
                if cfg_prompt_variations is not None
                else ""
            )
            print(
                f"  Client overrides: concurrency={cfg_concurrency} n={cfg_n} "
                f"isls={[fmt_isl(i) for i in cfg_isls]}{pv_note}"
            )
        elif cfg_prompt_variations is not None:
            print(f"  Client overrides: prompt_variations={cfg_prompt_variations}")

        prom_before = {}
        if args.metrics_url:
            prom_before = get_transfer_latency_stats(args.metrics_url)

        all_results = {}
        for scenario_key in args.scenarios:
            name, setup_fn = scenario_map[scenario_key]
            display_name = f"{name}" if not tuning_cfg else f"{name} [{label}]"
            print(f"\n  ▸ Running: {display_name}...")

            if args.mgmt and not args.skip_clear_all:
                clear_all_pools(args.mgmt)
                time.sleep(1)

            try:
                results = run_scenario(
                    scenario_key, args.url, args.model, args.mgmt,
                    cfg_isls, cfg_n, args.max_tokens, setup_fn,
                    concurrency=cfg_concurrency,
                    clear_between=(scenario_key == "cold"),
                    skip_cpu_flush=args.skip_cpu_flush,
                    seed=args.seed,
                    stream=args.stream,
                    ttft_mode=args.ttft_mode,
                    flush_wait=args.flush_wait,
                    variant_offset=cfg_idx * cfg_concurrency,
                    benchmark_name=args.name,
                    tuning_cfg=tuning_cfg,
                    prompt_variations=cfg_prompt_variations,
                    max_drain_wall_s=args.max_drain_seconds,
                )
            except TimeoutError as e:
                print(f"\n  ERROR: {e}", file=sys.stderr)
                print(
                    "  Hint: pinned blocks never reached 0 on one worker — check KVBM / disk / logs, "
                    "or use --skip-cpu-flush for a quicker (less strict) iteration.",
                    file=sys.stderr,
                )
                sys.exit(1)
            all_results[scenario_key] = results
            print_table(display_name, results, cfg_concurrency)
            sys.stdout.flush()

        if all(k in all_results for k in ("cold", "warm")):
            print_comparison(all_results["cold"], all_results["warm"])

        if args.metrics_url:
            prom_after = get_transfer_latency_stats(args.metrics_url)
            snap = {"before": prom_before, "after": prom_after}
            for direction in ("offload", "onboard"):
                before = prom_before.get(direction, {})
                after = prom_after.get(direction, {})
                delta_count = after.get("count", 0) - before.get("count", 0)
                delta_sum = after.get("sum", 0) - before.get("sum", 0)
                delta_bytes = (
                    prom_after.get(f"{direction}_bytes", 0)
                    - prom_before.get(f"{direction}_bytes", 0)
                )
                snap[f"{direction}_delta"] = {
                    "count": delta_count,
                    "sum_s": round(delta_sum, 3),
                    "avg": fmt_time(delta_sum / delta_count) if delta_count > 0 else "-",
                    "p50": fmt_time(after.get("p50")),
                    "p95": fmt_time(after.get("p95")),
                    "p99": fmt_time(after.get("p99")),
                    "bytes": delta_bytes,
                    "throughput": _fmt_throughput(delta_bytes, delta_sum),
                }
            prom_snapshots[label] = snap

            if len(tuning_configs) <= 1:
                print(f"\n  Prometheus transfer latency ({label}):")
                for direction in ("offload", "onboard"):
                    d = snap[f"{direction}_delta"]
                    print(f"    {direction}: count={d['count']} avg={d['avg']} p50={d['p50']} p95={d['p95']} p99={d['p99']}")

        if args.clickhouse_url:
            ch_rows = clickhouse_span_stats(args.clickhouse_url, since_seconds=300)
            print_clickhouse_span_table(ch_rows, label)

        matrix_results[label] = all_results

        # Insert per-ISL run rows into ClickHouse
        if args.clickhouse_url:
            cfg_finished_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            for scenario_key, scenario_results in all_results.items():
                for r in scenario_results:
                    system_cfg = _build_system_config(args)
                    ok = insert_kvbm_run(
                        ch_url=args.clickhouse_url,
                        run_id=r.get("run_id", str(uuid.uuid4())),
                        started_at=cfg_started_at,
                        finished_at=cfg_finished_at,
                        scenario=scenario_key,
                        tuning_label=label,
                        tuning_config=server_params,
                        client_config=ch_client_config,
                        system_config=system_cfg,
                        isls=[r["isl"]],
                        concurrency=cfg_concurrency,
                        n=r["n"],
                        results=[r],
                        model=args.model,
                        name=r.get("name", args.name),
                        tags={"config_idx": str(cfg_idx), "prompt_salt": _RUN_SALT},
                    )
                    if ok:
                        rid = r.get("run_id", "unknown")[:8]
                        print(f"  Inserted kvbm_run (run_id={rid}..., ISL={fmt_isl(r['isl'])}, config={label}, scenario={scenario_key})")

    # Restore original tuning if we changed it
    if original_tuning and args.mgmt:
        set_tuning(args.mgmt, original_tuning)
        print(f"\n  Restored original tuning: {original_tuning}")

    # Print matrix comparison if we ran multiple configs
    if len(matrix_results) > 1:
        print_matrix_comparison(matrix_results, tuning_configs, prom_snapshots)

    # Prom latency data is included in the matrix table when multiple configs are present.
    # For single-config runs, it was already printed inline above.

    if args.clickhouse_url:
        print("\n  ━━━ ClickHouse: Full Session Span Analytics ━━━")
        ch_rows = clickhouse_span_stats(args.clickhouse_url, since_seconds=7200)
        print_clickhouse_span_table(ch_rows, "all configs")

    if args.output:
        determinism.dump(args.output)
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
