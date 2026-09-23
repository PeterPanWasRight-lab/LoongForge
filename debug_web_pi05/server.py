#!/usr/bin/env python
"""LoongForge PI05 Console: stdlib HTTP, SSE logs and ranged video files.

Start: bash debug_web_pi05/run.sh
Foreground: python debug_web_pi05/server.py --port 8901
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEBUG_WEB_DIR = Path(__file__).resolve().parent

# A crash inside a C extension (pyarrow, for the dataset scan) kills the process
# without a Python traceback, which is indistinguishable from being killed. This
# dumps every Python stack to a file when that happens, so the culprit is
# recorded instead of guessed at.
_CRASH_LOG = DEBUG_WEB_DIR / "runtime" / "server.crash.log"
_CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
try:
    _crash_handle = open(_CRASH_LOG, "a", buffering=1)  # noqa: SIM115 - lives for the process
    faulthandler.enable(file=_crash_handle, all_threads=True)
except OSError:
    faulthandler.enable(all_threads=True)

sys.path.insert(0, str(DEBUG_WEB_DIR))

from app import api  # noqa: E402
from app.jobs import registry  # noqa: E402
from app.responses import File, Json, Raw, Response, Stream, Text  # noqa: E402

MAX_BODY_BYTES = 4 * 1024 * 1024

#: Block size used when streaming a file body. Large enough that syscalls stay
#: rare, small enough that three concurrent camera streams stay cheap.
_FILE_CHUNK = 256 * 1024

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _parse_range(value: str, size: int):
    """Interpret one ``Range`` header.

    Returns ``(start, end)`` when satisfiable, ``"unsatisfiable"`` when the range
    is well-formed but lies past the end of the file (a 416), and ``None`` when
    the header should simply be ignored — RFC 9110 requires falling back to a
    full response rather than failing on a header we do not understand.
    """
    match = _RANGE_RE.fullmatch((value or "").strip())
    if not match:
        return None

    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:  # "bytes=-N" → the trailing N bytes
        suffix = int(raw_end)
        if suffix <= 0 or size == 0:
            return "unsatisfiable"
        return max(0, size - suffix), size - 1

    start = int(raw_start)
    end = int(raw_end) if raw_end else size - 1
    if start >= size or start > end:
        return "unsatisfiable"
    return start, min(end, size - 1)


class RequestBodyError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "PI05Console/1.0"

    # ── verbs ────────────────────────────────────────────────────────────────
    def do_GET(self):  # noqa: N802
        self._handle("GET")

    def do_POST(self):  # noqa: N802
        self._handle("POST")

    def do_DELETE(self):  # noqa: N802
        self._handle("DELETE")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ── plumbing ─────────────────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")

    def _read_body(self) -> dict | None:
        if self.headers.get("Transfer-Encoding"):
            raise RequestBodyError("Transfer-Encoding is not supported")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise RequestBodyError("invalid Content-Length") from None
        if length < 0:
            raise RequestBodyError("invalid Content-Length")
        if length == 0:
            return None
        if length > MAX_BODY_BYTES:
            raise RequestBodyError("request body too large", 413)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RequestBodyError("request body must be valid JSON") from None
        if not isinstance(body, dict):
            raise RequestBodyError("request body must be a JSON object")
        return body

    def _handle(self, method: str) -> None:
        origin=self.headers.get('Origin')
        if method in {'POST','DELETE'} and origin and urlsplit(origin).netloc != self.headers.get('Host'):
            self.close_connection=True
            self._send(Json({'error':'禁止跨站修改'},status=403))
            return
        path, _, query = self.path.partition("?")
        try:
            body = self._read_body() if method in {"POST", "DELETE"} else None
        except RequestBodyError as exc:
            self.close_connection = True
            self._send(Json({"ok": False, "error": str(exc)}, status=exc.status))
            return

        started = time.time()
        try:
            response = api.dispatch(method, path, query, body)
        except KeyError:
            response = Json({"ok": False, "error": "not found"}, status=404)
        except Exception:  # noqa: BLE001 - a broken handler must not kill the server
            traceback.print_exc()
            response = Json({"ok": False, "error": traceback.format_exc(limit=3)}, status=500)

        if isinstance(response, Stream):
            self._send_stream(response)
        else:
            self._send(response)

        if not path.startswith(("/api/jobs/", "/api/scan")) or response.status >= 400:
            print(f"  {method} {path} -> {response.status} ({(time.time() - started) * 1000:.0f} ms)")

    def _send(self, response: Response) -> None:
        # Files get their own path because they are the only responses that
        # support byte ranges, and the only ones big enough to matter: a LeRobot
        # v3.0 dataset stores every episode of a camera in one ~30 MB mp4, so
        # "play episode 37" is a seek into a shared file.
        if isinstance(response, File):
            self._send_file(response)
            return

        try:
            payload = response.body()
        except OSError:
            self.send_error(404, "file missing")
            return
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in getattr(response, "headers", {}).items():
            self.send_header(key, value)
        if isinstance(response, Raw):
            self._cors()
            self.send_header("Accept-Ranges", "none")
        self.end_headers()
        if payload:
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_file(self, response: File) -> None:
        try:
            stat = response.path.stat()
        except OSError:
            self.send_error(404, "file missing")
            return
        size = stat.st_size

        start, end, status = 0, max(0, size - 1), int(getattr(response, "status", 200) or 200)
        requested = self.headers.get("Range")
        if requested:
            parsed = _parse_range(requested, size)
            if parsed == "unsatisfiable":
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self._cors()
                self.end_headers()
                return
            if parsed is not None:
                start, end, status = parsed[0], parsed[1], 206

        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        for key, value in getattr(response, "headers", {}).items():
            self.send_header(key, value)
        self._cors()
        self.end_headers()

        if length:
            self._pipe_file(response.path, start, length)

    def _pipe_file(self, path: Path, start: int, length: int) -> None:
        """Copy ``length`` bytes from ``start`` in fixed blocks.

        Deliberately not ``read_bytes()``: three camera streams of ~30 MB each
        would otherwise land in RAM simultaneously, on a box that is also
        running training.
        """
        remaining = length
        try:
            with open(path, "rb", buffering=0) as handle:
                handle.seek(start)
                while remaining > 0:
                    block = handle.read(min(_FILE_CHUNK, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The player seeked away or the tab was closed.
            self.close_connection = True
            return

        if remaining > 0:
            # The file shrank underneath us. We have already promised
            # Content-Length bytes, so the framing is broken and the only safe
            # move on a keep-alive connection is to drop it.
            self.close_connection = True

    def _send_stream(self, response: Stream) -> None:
        """Chunked transfer encoding, so the browser renders frames as they come."""
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Transfer-Encoding", "chunked")
        for key, value in response.headers.items():
            self.send_header(key, value)
        self._cors()
        self.end_headers()

        generator = response.generator
        try:
            for chunk in generator:
                if not chunk:
                    continue
                self.wfile.write(f"{len(chunk):X}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client navigated away / closed the tab. The framing is now broken,
            # so this connection must not be reused for a following request.
            self.close_connection = True
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                self.close_connection = True

    def log_message(self, fmt, *args):  # noqa: A002 - silence per-request noise
        return


class Server(ThreadingHTTPServer):
    """Threaded HTTP server with a backlog that matches how the console is used.

    ``http.server`` accepts a queue of only 5 pending connections. Opening the
    inference page bursts 3 camera streams + an event stream + the usual API
    calls at once, and anything past the backlog is refused outright instead of
    being queued — which the console reports as a dropped connection.
    """

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


def detect_public_host() -> str:
    """First real IPv4 of this machine, for the URLs we print.

    Skips loopback, the Clash TUN range (198.18/15) and the docker bridge
    (172.17/16) so the banner shows the address users should actually open.
    """
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "127.0.0.1"
    for token in out.split():
        if ":" in token or token.startswith(("127.", "198.18.", "172.17.")):
            continue
        return token
    return "127.0.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument(
        "--public-host",
        default=None,
        help="host to advertise in the banner (default: auto-detected LAN IP)",
    )
    parser.add_argument(
        "--stop-jobs-on-exit",
        action="store_true",
        help="also kill running jobs on shutdown (default: leave them running; they are re-attached on restart)",
    )
    args = parser.parse_args()

    server = Server((args.host, args.port), Handler)
    server.daemon_threads = True
    server.allow_reuse_address = True

    def shutdown(_signum, _frame):
        print("\n[server] shutting down…")
        if args.stop_jobs_on_exit:
            print("[server] stopping running jobs (--stop-jobs-on-exit)")
            registry.stop_all()
        else:
            running = [j["id"] for j in registry.list() if j["status"] == "running"]
            if running:
                # Jobs write to their own log file and keep their own pid, so a
                # restart re-attaches to them instead of orphaning them.
                print(f"[server] leaving {len(running)} job(s) running; they resume on restart:")
                for job_id in running:
                    print(f"           {job_id}")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    public_host = args.public_host or detect_public_host()
    print("=" * 68)
    print("  LoongForge PI05 Console")
    print(f"  url     : http://{public_host}:{args.port}/")
    print(f"  bind    : {args.host}:{args.port}")
    print(f"  runtime : {api.runtime_paths()['jobs_dir']}")
    print("=" * 68)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if args.stop_jobs_on_exit:
            registry.stop_all()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
