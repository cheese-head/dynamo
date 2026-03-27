# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KVBM Management HTTP Server.

Provides a lightweight REST API for managing KVBM pools when KVBM_DEV_MODE=TRUE.

The server runs in a **separate child process** so that its HTTP handling and
any blocking operations (e.g. clear_pool -> blocking_recv) never hold the
Python GIL in the vLLM EngineCore process.  Communication between the HTTP
process and the parent EngineCore uses a multiprocessing Pipe.

Endpoints:
    POST /v1/cache/clear          Clear a specific pool (body: {"pool": "cpu"})
    POST /v1/cache/clear_all      Clear all pools (device, host, disk)
    POST /v1/cache/flush          Wait for offloads to drain, then clear (body: {"pool": "cpu", "max_wait": 60})
    POST /v1/cache/cpu_lookup     Enable/disable CPU cache lookup (body: {"disabled": true})
    GET  /v1/cache/status          Get pool status summary
    GET  /v1/cache/cpu_lookup     Get CPU cache lookup status
    GET  /v1/tuning               Get current tuning parameters
    POST /v1/tuning               Update tuning parameters (body: {"param": "value", ...})
    GET  /v1/health                Health check
    POST /v1/metrics/reset        Reset Prometheus histograms

Environment variables:
    KVBM_DEV_MODE               Must be TRUE/true/1 to enable destructive operations.
    KVBM_MANAGEMENT_PORT        HTTP port for the management server (default: 6881).
    KVBM_MANAGEMENT_ENABLED     Set to TRUE/true/1 to auto-start the server (default: same as KVBM_DEV_MODE).
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time as _time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MANAGEMENT_PORT = 6881

VALID_POOLS = {"gpu", "device", "cpu", "host", "disk"}


def _is_dev_mode() -> bool:
    return os.environ.get("KVBM_DEV_MODE", "").lower() in ("true", "1")


def _is_management_enabled() -> bool:
    val = os.environ.get("KVBM_MANAGEMENT_ENABLED", "")
    if val:
        return val.lower() in ("true", "1")
    return _is_dev_mode()


def _get_management_port() -> int:
    try:
        return int(os.environ.get("KVBM_MANAGEMENT_PORT", str(DEFAULT_MANAGEMENT_PORT)))
    except ValueError:
        return DEFAULT_MANAGEMENT_PORT


# ── IPC message types ────────────────────────────────────────────────
# Each request is a tuple: (command: str, args: dict)
# Each response is a tuple: (ok: bool, result: Any)
# "result" is either the return value (ok=True) or an error string (ok=False).

_CMD_CLEAR_POOL = "clear_pool"
_CMD_GET_STATUS = "get_status"
_CMD_SET_CPU_LOOKUP = "set_cpu_lookup"
_CMD_GET_CPU_LOOKUP = "get_cpu_lookup"
_CMD_FLUSH_AND_CLEAR = "flush_and_clear"
_CMD_RESET_METRICS = "reset_metrics"
_CMD_GET_TUNING = "get_tuning"
_CMD_SET_TUNING = "set_tuning"


# ── Pool operations (parent-side, holds real callbacks) ──────────────

class _PoolOpsLocal:
    """Holds the real callable references registered by the KVBM leader.

    Lives in the parent (EngineCore) process only.
    """

    def __init__(self):
        self._clear_pool: Optional[Callable[[str], None]] = None
        self._get_status: Optional[Callable[[], dict[str, Any]]] = None
        self._set_cpu_lookup_disabled: Optional[Callable[[bool], None]] = None
        self._get_cpu_lookup_status: Optional[Callable[[], dict[str, Any]]] = None
        self._reset_metrics: Optional[Callable[[], None]] = None

    def register_clear_pool(self, fn: Callable[[str], None]) -> None:
        self._clear_pool = fn

    def register_get_status(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._get_status = fn

    def register_set_cpu_lookup_disabled(self, fn: Callable[[bool], None]) -> None:
        self._set_cpu_lookup_disabled = fn

    def register_get_cpu_lookup_status(self, fn: Callable[[], dict[str, Any]]) -> None:
        self._get_cpu_lookup_status = fn

    def register_reset_metrics(self, fn: Callable[[], None]) -> None:
        self._reset_metrics = fn

    def handle_command(self, cmd: str, args: dict) -> tuple[bool, Any]:
        """Execute a management command and return (ok, result)."""
        try:
            if cmd == _CMD_CLEAR_POOL:
                if self._clear_pool is None:
                    raise RuntimeError("No clear_pool handler registered")
                self._clear_pool(args["pool"])
                return (True, None)

            elif cmd == _CMD_GET_STATUS:
                if self._get_status is None:
                    return (True, {"status": "no status handler registered"})
                return (True, self._get_status())

            elif cmd == _CMD_SET_CPU_LOOKUP:
                if self._set_cpu_lookup_disabled is None:
                    raise RuntimeError("No cpu_lookup setter registered")
                self._set_cpu_lookup_disabled(args["disabled"])
                return (True, None)

            elif cmd == _CMD_GET_CPU_LOOKUP:
                if self._get_cpu_lookup_status is None:
                    return (True, {"status": "no cpu_lookup status handler registered"})
                return (True, self._get_cpu_lookup_status())

            elif cmd == _CMD_FLUSH_AND_CLEAR:
                return (True, self._flush_and_clear(args["pool"]))

            elif cmd == _CMD_RESET_METRICS:
                if self._reset_metrics is None:
                    raise RuntimeError("No reset_metrics handler registered")
                self._reset_metrics()
                return (True, None)

            elif cmd == _CMD_GET_TUNING:
                from kvbm._core import get_tuning_params
                return (True, get_tuning_params())

            elif cmd == _CMD_SET_TUNING:
                from kvbm._core import set_tuning_param, get_tuning_params
                errors: dict[str, str] = {}
                updated: dict[str, int] = {}
                valid_params = {
                    "transfer_batch_size", "max_concurrent_transfers",
                    "flush_batch_size", "g4_pipeline_chunk_size",
                    "g4_transfer_timeout_secs",
                }
                for key, value in args.items():
                    if key not in valid_params:
                        errors[key] = f"unknown parameter (valid: {sorted(valid_params)})"
                        continue
                    try:
                        int_val = int(value)
                        if int_val < 1:
                            errors[key] = "value must be >= 1"
                            continue
                        set_tuning_param(key, int_val)
                        updated[key] = int_val
                    except (ValueError, TypeError) as e:
                        errors[key] = f"invalid value: {e}"
                    except Exception as e:
                        errors[key] = str(e)
                result = {"updated": updated, "tuning": get_tuning_params()}
                if errors:
                    result["errors"] = errors
                return (True, result)

            else:
                return (False, f"unknown command: {cmd}")

        except Exception as e:
            return (False, str(e))

    def _flush_and_clear(self, pool: str) -> dict[str, Any]:
        start = _time.monotonic()
        clear_error = None
        try:
            if self._clear_pool:
                self._clear_pool(pool)
        except Exception as e:
            clear_error = str(e)

        result: dict[str, Any] = {
            "pool": pool,
            "total_time_s": round(_time.monotonic() - start, 3),
        }
        if clear_error:
            result["clear_error"] = clear_error
        return result


# Parent-side singleton holding the real callbacks.
_local_ops = _PoolOpsLocal()


# ── IPC proxy (child-side, used by HTTP handlers) ────────────────────

class _PoolOpsProxy:
    """Proxy that sends commands over a pipe to the parent process.

    Used inside the child HTTP server process.  Every method blocks until
    the parent executes the command and returns a result.
    """

    def __init__(self, conn: multiprocessing.connection.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def _call(self, cmd: str, args: dict | None = None) -> Any:
        with self._lock:
            self._conn.send((cmd, args or {}))
            ok, result = self._conn.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def clear_pool(self, pool: str) -> None:
        self._call(_CMD_CLEAR_POOL, {"pool": pool})

    def get_status(self) -> dict[str, Any]:
        return self._call(_CMD_GET_STATUS)

    def set_cpu_lookup_disabled(self, disabled: bool) -> None:
        self._call(_CMD_SET_CPU_LOOKUP, {"disabled": disabled})

    def get_cpu_lookup_status(self) -> dict[str, Any]:
        return self._call(_CMD_GET_CPU_LOOKUP)

    def flush_and_clear(self, pool: str) -> dict[str, Any]:
        return self._call(_CMD_FLUSH_AND_CLEAR, {"pool": pool})

    def reset_metrics(self) -> None:
        self._call(_CMD_RESET_METRICS)

    def get_tuning(self) -> dict:
        return self._call(_CMD_GET_TUNING)

    def set_tuning(self, params: dict) -> dict:
        return self._call(_CMD_SET_TUNING, params)


# The _pool_ops variable is set to _PoolOpsProxy in the child process
# and stays None in the parent.  The HTTP handler always uses _pool_ops.
_pool_ops: Optional[_PoolOpsProxy] = None


# ── Public registration API (unchanged, called in parent process) ────

def register_clear_pool(fn: Callable[[str], None]) -> None:
    _local_ops.register_clear_pool(fn)

def register_get_status(fn: Callable[[], dict[str, Any]]) -> None:
    _local_ops.register_get_status(fn)

def register_set_cpu_lookup_disabled(fn: Callable[[bool], None]) -> None:
    _local_ops.register_set_cpu_lookup_disabled(fn)

def register_get_cpu_lookup_status(fn: Callable[[], dict[str, Any]]) -> None:
    _local_ops.register_get_cpu_lookup_status(fn)

def register_reset_metrics(fn: Callable[[], None]) -> None:
    _local_ops.register_reset_metrics(fn)


# ── HTTP handler (runs in child process) ─────────────────────────────

class _ManagementHandler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, body: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, indent=2).encode())

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/health":
            self._send_json(200, {"status": "ok", "dev_mode": _is_dev_mode()})
        elif self.path == "/v1/cache/status":
            try:
                status = _pool_ops.get_status()
                self._send_json(200, {"status": "ok", "pools": status})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path == "/v1/cache/cpu_lookup":
            try:
                status = _pool_ops.get_cpu_lookup_status()
                self._send_json(200, {"status": "ok", **status})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.path == "/v1/tuning":
            self._handle_get_tuning()
        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    def do_POST(self):  # noqa: N802
        if self.path == "/v1/cache/clear":
            self._handle_clear_pool()
        elif self.path == "/v1/cache/clear_all":
            self._handle_clear_all()
        elif self.path == "/v1/cache/cpu_lookup":
            self._handle_cpu_lookup_toggle()
        elif self.path == "/v1/cache/flush":
            self._handle_flush()
        elif self.path == "/v1/tuning":
            self._handle_set_tuning()
        elif self.path == "/v1/metrics/reset":
            self._handle_metrics_reset()
        else:
            self._send_json(404, {"error": f"not found: {self.path}"})

    def _handle_clear_pool(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {"error": "KVBM_DEV_MODE is not enabled."})
            return
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        pool = body.get("pool", "").strip().lower()
        if pool not in VALID_POOLS:
            self._send_json(400, {"error": f"invalid pool '{pool}'"})
            return
        try:
            _pool_ops.clear_pool(pool)
            self._send_json(200, {"status": "ok", "pool": pool, "action": "cleared"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_clear_all(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {"error": "KVBM_DEV_MODE is not enabled."})
            return
        errors: dict[str, str] = {}
        cleared: list[str] = []
        for pool in ["gpu", "cpu", "disk"]:
            try:
                _pool_ops.clear_pool(pool)
                cleared.append(pool)
            except Exception as e:
                errors[pool] = str(e)
        status = 200 if not errors else 207
        self._send_json(status, {
            "status": "partial" if errors else "ok",
            "cleared": cleared, "errors": errors,
        })

    def _handle_cpu_lookup_toggle(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {"error": "KVBM_DEV_MODE is not enabled."})
            return
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        if "disabled" not in body:
            self._send_json(400, {"error": "missing required field: disabled"})
            return
        disabled = bool(body.get("disabled"))
        if disabled:
            self._send_json(400, {"error": "disabling CPU lookup is no longer supported"})
            return
        try:
            _pool_ops.set_cpu_lookup_disabled(disabled)
            status = _pool_ops.get_cpu_lookup_status()
            self._send_json(200, {"status": "ok", **status})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_flush(self) -> None:
        if not _is_dev_mode():
            self._send_json(403, {"error": "KVBM_DEV_MODE is not enabled."})
            return
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        pool = body.get("pool", "cpu").strip().lower()
        if pool not in VALID_POOLS:
            self._send_json(400, {"error": f"invalid pool '{pool}'"})
            return
        try:
            result = _pool_ops.flush_and_clear(pool)
            status_code = 200 if "clear_error" not in result else 207
            self._send_json(status_code, {"status": "ok" if "clear_error" not in result else "partial", **result})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_metrics_reset(self) -> None:
        try:
            _pool_ops.reset_metrics()
            self._send_json(200, {"status": "ok", "action": "metrics_reset"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_get_tuning(self) -> None:
        try:
            params = _pool_ops.get_tuning()
            self._send_json(200, {"status": "ok", "tuning": params})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_set_tuning(self) -> None:
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        if not body:
            self._send_json(400, {"error": "body must contain at least one tuning parameter"})
            return
        try:
            result = _pool_ops.set_tuning(body)
            has_errors = "errors" in result and result["errors"]
            status_code = 200 if not has_errors else 207
            self._send_json(status_code, {"status": "partial" if has_errors else "ok", **result})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, format: str, *args) -> None:
        logger.info("[kvbm-mgmt] %s", format % args)


# ── Child process entrypoint ─────────────────────────────────────────

def _run_management_server(conn: multiprocessing.connection.Connection, port: int) -> None:
    """Entry point for the management HTTP server child process."""
    global _pool_ops

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _pool_ops = _PoolOpsProxy(conn)

    try:
        server = HTTPServer(("0.0.0.0", port), _ManagementHandler)
    except OSError as e:
        logger.error("Management server failed to bind port %d: %s", port, e)
        return

    logger.info("Management server child process started on port %d", port)
    server.serve_forever()


# ── Parent-side pipe handler ─────────────────────────────────────────

def _handle_pipe_requests(conn: multiprocessing.connection.Connection) -> None:
    """Runs in a daemon thread in the parent process.

    Reads (cmd, args) from the pipe, dispatches to _local_ops, and sends
    (ok, result) back.  This thread acquires the GIL only briefly per
    command -- it does not hold it while waiting for pipe data.
    """
    while True:
        try:
            if not conn.poll(timeout=1.0):
                continue
            cmd, args = conn.recv()
            ok, result = _local_ops.handle_command(cmd, args)
            conn.send((ok, result))
        except EOFError:
            logger.info("Management pipe closed, stopping handler")
            break
        except Exception as e:
            logger.error("Pipe handler error: %s", e)
            try:
                conn.send((False, str(e)))
            except Exception:
                break


# ── Server lifecycle ─────────────────────────────────────────────────

_server_process: Optional[multiprocessing.Process] = None
_pipe_thread: Optional[threading.Thread] = None


def start_management_server(port: Optional[int] = None) -> Optional[int]:
    """Start the management HTTP server in a separate child process.

    The child process handles HTTP requests independently of the parent's
    GIL.  Operations that need the KVBM leader are dispatched via a pipe
    back to the parent and executed in a lightweight handler thread.
    """
    global _server_process, _pipe_thread

    if _server_process is not None and _server_process.is_alive():
        logger.warning("KVBM management server is already running")
        return port or _get_management_port()

    if not _is_management_enabled():
        logger.debug("KVBM management server not enabled")
        return None

    if port is None:
        port = _get_management_port()

    parent_conn, child_conn = multiprocessing.Pipe()

    _server_process = multiprocessing.Process(
        target=_run_management_server,
        args=(child_conn, port),
        daemon=True,
        name="kvbm-management",
    )
    _server_process.start()

    _pipe_thread = threading.Thread(
        target=_handle_pipe_requests,
        args=(parent_conn,),
        daemon=True,
        name="kvbm-mgmt-pipe",
    )
    _pipe_thread.start()

    logger.info(
        "KVBM management server started (pid=%d, port=%d, dev_mode=%s)",
        _server_process.pid, port, _is_dev_mode(),
    )
    return port


def stop_management_server() -> None:
    """Stop the management server if running."""
    global _server_process, _pipe_thread

    if _server_process is not None:
        _server_process.terminate()
        _server_process.join(timeout=5)
        _server_process = None

    _pipe_thread = None

    logger.info("KVBM management server stopped")


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import signal
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="KVBM Management HTTP Server")
    parser.add_argument("--port", type=int, default=_get_management_port(), help="Port to listen on")
    args = parser.parse_args()

    if not _is_dev_mode():
        logger.warning("KVBM_DEV_MODE is not set. Destructive operations will be rejected.")

    os.environ["KVBM_MANAGEMENT_ENABLED"] = "TRUE"

    port = start_management_server(args.port)
    if port is None:
        logger.error("Failed to start management server")
        sys.exit(1)

    print(f"KVBM Management Server listening on http://0.0.0.0:{port}")
    print(f"  POST /v1/cache/clear      - Clear a pool")
    print(f"  POST /v1/cache/clear_all   - Clear all pools")
    print(f"  GET  /v1/cache/status      - Pool status")
    print(f"  GET  /v1/health            - Health check")

    evt = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: evt.set())
    signal.signal(signal.SIGTERM, lambda *_: evt.set())
    evt.wait()

    stop_management_server()
