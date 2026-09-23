#!/usr/bin/env python
"""Job registry: subprocess lifecycle, log buffering and structured events.

Each job owns a directory under ``debug_web/runtime/jobs/<id>`` holding

    meta.json     job description + final status (survives a server restart)
    log.txt       raw stdout/stderr, appended live
    events.jsonl  parsed worker events (metrics, curves, frame notices)
    frames/       latest JPEG per camera, written by the worker

Workers speak one protocol: every stdout line is either plain text (kept as a
log line) or a JSON object ``{"t": "<type>", ...}``.  That keeps the transport
trivial to inspect with ``tail -f`` while still giving the UI structure.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .settings import JOBS_DIR, settings, subprocess_env

TERMINAL_STATES = {"succeeded", "failed", "stopped"}

#: ``lerobot-train`` prints one of these per logging step:
#:   INFO ... step:4 smpl:8 ep:0 epch:0.00 loss:0.221 grdn:6.950 lr:8.1e-06 ...
_TRAIN_STEP_RE = re.compile(
    r"step:\s*(\d+).*?"
    r"(?:smpl:\s*(\d+).*?)?"
    r"loss:\s*([-\d.eE+]+).*?"
    r"grdn:\s*([-\d.eE+]+).*?"
    r"lr:\s*([-\d.eE+]+)",
)
_LOG_LEVEL_RE = re.compile(r"\b(INFO|WARNING|ERROR|CRITICAL|DEBUG)\b")
#: Cap the in-memory curve so a long run does not grow the metrics payload forever.
_MAX_LOSS_POINTS = 4000


def _now() -> float:
    return time.time()


def _stamp(ts: float | None = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts or _now()))


def _split_line(raw: str) -> list[str]:
    """Turn one stdout line into the lines a human would actually want to see.

    tqdm redraws its bar with ``\\r`` and no newline, so a single read can hold
    dozens of progress frames glued to the real log record. Keep the records and
    drop the intermediate bar frames.
    """
    text = raw.rstrip("\n")
    if "\r" not in text:
        return [text]
    parts = [p for p in text.split("\r") if p.strip()]
    if not parts:
        return []
    keep = [p for p in parts if _LOG_LEVEL_RE.search(p)]
    if keep:
        return keep
    return [parts[-1]]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except OSError:
            pass
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_matches(pid: int, argv: list[str]) -> bool:
    """Guard against PID reuse: the live process must be the job we think it is."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            actual = [part.decode("utf-8", "replace") for part in handle.read().split(b"\0") if part]
    except OSError:
        return False
    if not actual or not argv:
        return False
    # Compare the interpreter and the script/module that identifies the job.
    return actual == argv


_LAUNCHER_EXIT_RE = re.compile(r"\[launcher\] exited with code (-?\d+) \((\w+)\)")
_WORKER_EXIT_RE = re.compile(r'\{\s*"t"\s*:\s*"exit"\s*,\s*"code"\s*:\s*(-?\d+)\s*\}')
#: ``lerobot-train`` prints this only when a run completes normally.
_TRAIN_DONE_RE = re.compile(r"End of training")


def _status_from_log(job: Job) -> tuple[str, int | None] | None:
    """Recover a job's final status from its own log.

    Needed whenever a run finished while no console was watching (server
    restarted / killed), because a re-parented process cannot be reaped and its
    exit code is therefore unavailable to us. Three independent records are
    consulted, most explicit first:

    1. ``[launcher] exited with code N (status)`` — written by whichever console
       was present when the process ended.
    2. ``{"t": "exit", "code": N}`` — written by the worker itself.
    3. ``End of training`` — the trainer's own completion marker.
    """
    if not job.log_path.is_file():
        return None
    try:
        with job.log_path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = deque(handle, maxlen=400)
    except OSError:
        return None

    for line in reversed(tail):
        match = _LAUNCHER_EXIT_RE.search(line)
        if match:
            return match.group(2), int(match.group(1))

    for line in reversed(tail):
        match = _WORKER_EXIT_RE.search(line)
        if match:
            code = int(match.group(1))
            return ("succeeded" if code == 0 else "failed"), code

    if job.kind == "train" and any(_TRAIN_DONE_RE.search(line) for line in tail):
        return "succeeded", 0
    return None


def parse_train_progress(line: str) -> dict | None:
    """Extract step/loss/grad-norm/lr from a training log record."""
    matches = _TRAIN_STEP_RE.findall(line)
    if not matches:
        return None
    step, samples, loss, grad_norm, lr = matches[-1]
    out: dict = {"step": int(step)}
    if samples:
        out["samples"] = int(samples)
    for key, value in (("loss", loss), ("grad_norm", grad_norm), ("lr", lr)):
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


class Job:
    """One child process (a training run, a rollout, a dataset replay)."""

    def __init__(
        self,
        kind: str,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict | None = None,
        params: dict | None = None,
        title: str = "",
        cams: list[str] | None = None,
        job_id: str | None = None,
        write_meta: bool = True,
    ):
        self.id = job_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.kind = kind
        self.argv = list(argv)
        self.cwd = cwd or str(settings.get("repo_root"))
        self.env = dict(env or {})
        self.params = dict(params or {})
        self.title = title or kind
        self.cams = list(cams or [])

        self.status = "pending"
        self.returncode: int | None = None
        self.error: str | None = None
        self.created_at = _now()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        # Latest metric snapshot so a page reload can render without replaying.
        # ``metrics_rev`` lets the SSE feed skip re-sending an unchanged payload
        # (curve updates can be a few hundred KB).
        self.metrics: dict = {}
        self.summary: dict = {}
        self.metrics_rev = 0

        self.dir = JOBS_DIR / self.id
        self.frames_dir = self.dir / "frames"
        self.log_path = self.dir / "log.txt"
        self.events_path = self.dir / "events.jsonl"
        self.meta_path = self.dir / "meta.json"

        max_lines = int(settings.get("max_log_lines", 20000) or 20000)
        self._lines: deque[tuple[float, str]] = deque(maxlen=max_lines)
        self._log_total = 0
        self._restore_offset = 0
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        #: Set when the process was started by a *previous* server run: we can
        #: watch it, but we cannot `wait()` on it.
        self._foreign_pid: int | None = None
        self._log_offset = 0

        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()
        if write_meta:
            # Never on restore: the file on disk already holds the real status,
            # and rewriting it here would reset a finished job to "pending".
            self._write_meta()

    # ── persistence ──────────────────────────────────────────────────────────
    def _write_meta(self) -> None:
        with self._lock:
            self._write_meta_locked()

    def _write_meta_locked(self) -> None:
        meta = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "argv": self.argv,
            "cmdline": shlex.join(self.argv),
            "cwd": self.cwd,
            "params": self.params,
            "cams": self.cams,
            "status": self.status,
            "returncode": self.returncode,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metrics": self.metrics,
            "summary": self.summary,
            "pid": self._proc.pid if self._proc is not None else self._foreign_pid,
        }
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.meta_path)

    def to_dict(self, *, with_lines: bool = False) -> dict:
        with self._lock:
            data = {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "returncode": self.returncode,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "duration": ((self.ended_at or _now()) - (self.started_at or self.created_at))
                if self.ended_at is not None or self.status in {"pending", "running", "stopping"} else None,
                "pid": self._proc.pid if self._proc is not None else self._foreign_pid,
                "cmdline": shlex.join(self.argv),
                "cwd": self.cwd,
                "params": self.params,
                "cams": self.cams,
                "metrics": dict(self.metrics),
                "summary": dict(self.summary),
                "metrics_rev": self.metrics_rev,
                "log_lines": len(self._lines),
                "dir": str(self.dir),
                "log_path": str(self.log_path),
            }
            if with_lines:
                data["lines"] = [{"ts": ts, "text": text} for ts, text in list(self._lines)]
            return data

    # ── log plumbing ─────────────────────────────────────────────────────────
    def _append_line(self, text: str) -> None:
        with self._lock:
            self._lines.append((_now(), text))
            self._log_total += 1

    def log_since(self, offset: int, limit: int = 2000) -> dict:
        with self._lock:
            total = self._log_total
            first = total - len(self._lines)
            start = max(first, min(max(0, offset), total))
            chunk = list(self._lines)[start - first : start - first + max(1, min(limit, 20000))]
            return {
                "offset": start + len(chunk), "total": total, "first": first,
                "truncated": offset < first,
                "lines": [{"ts": ts, "text": text} for ts, text in chunk],
                "status": self.status, "returncode": self.returncode,
            }

    def log_snapshot(self, count: int = 10000) -> dict:
        with self._lock:
            return self.log_since(max(0, self._log_total - count), count)

    def log_tail(self, count: int = 400) -> list[dict]:
        with self._lock:
            snapshot = list(self._lines)[-count:]
        return [{"ts": ts, "text": text} for ts, text in snapshot]

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                return
            self._append_line(f"[launcher] $ {' '.join(self.argv)}")
            self._append_line(f"[launcher] cwd={self.cwd}")
            env = subprocess_env(self.env)
            try:
                # The child writes straight into its log file rather than into a
                # pipe. That way the record survives this server process: after a
                # restart the job can still be followed (and a long training run
                # does not become an unobservable orphan).
                with self.log_path.open("ab", buffering=0) as log_handle:
                    self._proc = subprocess.Popen(  # noqa: S603 - argv is built by the server
                        self.argv,
                        cwd=self.cwd,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        # Own process group: torchrun spawns children, and killing only
                        # the parent would leave GPU-resident orphans behind.
                        start_new_session=True,
                    )
            except (OSError, FileNotFoundError) as exc:
                self.status = "failed"
                self.error = str(exc)
                self.ended_at = _now()
                self._append_line(f"[launcher] failed to start: {exc}")
                self._write_meta()
                return

            self.status = "running"
            self.started_at = _now()
            self._append_line(f"[launcher] pid={self._proc.pid}")
            self._write_meta()
            self._reader = threading.Thread(target=self._pump_output, name=f"job-{self.id}", daemon=True)
            self._reader.start()

    def _is_running(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        if self._foreign_pid is not None:
            return _pid_alive(self._foreign_pid) and _pid_matches(self._foreign_pid, self.argv)
        return False

    def _pump_output(self, *, from_start: bool = True) -> None:
        """Tail the job's log file, folding new lines into memory and metrics.

        Tailing (rather than reading a pipe) is what lets a job stay observable
        after a server restart — see :meth:`Job.resume_tailing`.
        """
        try:
            with self.log_path.open("rb") as handle:
                if not from_start:
                    handle.seek(self._restore_offset)
                pending = b""
                while True:
                    block = handle.read(65536)
                    if block:
                        pending += block
                        records = pending.split(b"\n")
                        pending = records.pop()
                        for raw in records:
                            self._consume_raw(raw.decode("utf-8", "replace"))
                        continue
                    if not self._is_running():
                        # Re-read once after exit: the last write can race our EOF.
                        rest = handle.read()
                        pending += rest
                        for raw in pending.splitlines():
                            self._consume_raw(raw.decode("utf-8", "replace"))
                        break
                    time.sleep(0.12)
        except Exception as exc:  # noqa: BLE001 - never let the reader die silently
            self._append_line(f"[launcher] log reader error: {exc}")
        finally:
            self._finalize()

    def resume_tailing(self, pid: int) -> None:
        """Re-attach to a job whose process outlived a server restart."""
        self._foreign_pid = pid
        self.status = "running"
        if self.started_at is None:
            self.started_at = self.created_at
        self._append_line(f"[launcher] re-attached to pid {pid} after a server restart")
        self._write_meta()
        self._reader = threading.Thread(
            target=self._pump_output, kwargs={"from_start": False},
            name=f"job-{self.id}", daemon=True,
        )
        self._reader.start()

    def _consume_raw(self, raw: str) -> None:
        """Handle one raw line (or tqdm burst) coming out of the child."""
        for text in _split_line(raw):
            self._append_line(text)
            stripped = text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    self._handle_event(event)
                continue
            self._handle_plain_line(text)

    def _finalize(self) -> None:
        """Record the exit status once the child is gone."""
        proc = self._proc
        code = self.returncode
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            code = proc.returncode

        # The child always writes its own verdict as the last log line, whichever
        # server was watching at the time. It is the most reliable source, and the
        # only one available for a job we re-attached to after a restart.
        recorded = _status_from_log(self)

        with self._lock:
            if code is None and recorded:
                code = recorded[1]
            self.returncode = code
            if self.status == "stopping":
                self.status = "stopped"
            elif recorded and recorded[0] in TERMINAL_STATES:
                self.status = recorded[0]
            elif code == 0:
                self.status = "succeeded"
            else:
                self.status = "failed"
            self.ended_at = _now()
            marker = f"[launcher] exited with code {code} ({self.status})"
            self._append_line(marker)
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(marker + "\n")
        except OSError:
            pass
        self._write_meta()

    def _handle_plain_line(self, line: str) -> None:
        """Fold recognised plain-text progress (``lerobot-train``) into metrics.

        ``loss_history.csv`` is not always produced, so the console reads the run
        from the log instead — which also means the chart fills in live rather
        than only after the run ends.
        """
        progress = parse_train_progress(line)
        if not progress:
            return
        with self._lock:
            self.metrics.update(progress)
            step = progress.get("step")
            loss = progress.get("loss")
            if step is not None and loss is not None:
                curve = self.metrics.setdefault("loss_curve", [])
                if not curve or curve[-1][0] != step:
                    curve.append([step, loss])
                    if len(curve) > _MAX_LOSS_POINTS:
                        del curve[: len(curve) - _MAX_LOSS_POINTS]
            self.metrics_rev += 1

    def _handle_event(self, event: dict, *, persist: bool = True) -> None:
        """Fold a worker JSON event into metrics / summary and persist it."""
        kind = event.get("t") or event.get("type")
        with self._lock:
            if persist:
                try:
                    with self.events_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                except OSError:
                    pass
            if kind == "metric":
                self.metrics.update({k: v for k, v in event.items() if k != "t"})
                self.metrics_rev += 1
            elif kind == "summary":
                self.summary.update({k: v for k, v in event.items() if k != "t"})
                self.metrics_rev += 1
            elif kind == "curves":
                # Curves are large; keep only the newest payload.
                self.metrics["curves"] = event.get("series", [])
                self.metrics_rev += 1
            elif kind == "status":
                self.metrics["status_text"] = event.get("text", "")
                self.metrics_rev += 1
            elif kind == "episode":
                rows = self.metrics.setdefault("episodes", [])
                rows.append({k: v for k, v in event.items() if k != "t"})
                self.metrics_rev += 1
            elif kind == "exit":
                # The worker's own verdict, emitted just before it exits. This is
                # what lets a console that re-attached after a restart still know
                # whether the run succeeded.
                self.returncode = event.get("code")

    def stop(self, *, timeout: float = 20.0, block: bool = True) -> dict:
        """SIGINT -> SIGTERM -> SIGKILL, always against the whole process group.

        ``block=False`` returns as soon as the stop has been requested and lets a
        background thread do the escalating. The HTTP handler must use that: a
        worker that ignores SIGINT costs the full ``timeout`` (20 s) before
        SIGTERM, so a synchronous stop can pin one of the browser's few
        connections open for half a minute. The job's own status transitions to
        ``stopped`` when the process actually dies, and the event stream reports
        it.
        """
        with self._lock:
            pid = self._proc.pid if self._proc is not None else self._foreign_pid
            if self.status == "stopping":
                return {"stopped": True, "pid": pid, "async": True}
            if pid is None or not self._is_running():
                if self.status not in TERMINAL_STATES:
                    self.status = "stopped"
                    self.ended_at = _now()
                    self._write_meta()
                return {"stopped": False, "reason": "not running"}
            self.status = "stopping"
            self._append_line("[launcher] stop requested (SIGINT to process group)")
        self._write_meta()

        if block:
            self._escalate(pid, timeout)
            return {"stopped": True, "pid": pid}

        threading.Thread(
            target=self._escalate, args=(pid, timeout), name=f"stop-{self.id}", daemon=True
        ).start()
        return {"stopped": True, "pid": pid, "async": True}

    def _escalate(self, pid: int, timeout: float) -> None:
        """Walk SIGINT -> SIGTERM -> SIGKILL until the process group is gone."""
        for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, max(timeout / 2, 5.0)),
                          (signal.SIGKILL, 5.0)):
            if self._signal_group(pid, sig, wait):
                return

    def _signal_group(self, pid: int, sig: int, timeout: float) -> bool:
        """Signal the process group and wait; returns True once it is gone.

        Works for both jobs we launched and jobs inherited from a previous
        server run (for those we only have a pid, not a ``Popen``).
        """
        try:
            if self._foreign_pid is not None and not _pid_matches(pid, self.argv):
                return True
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, sig)
            except OSError:
                return True
        deadline = time.time() + max(timeout, 0.1)
        while time.time() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.2)
        return not _pid_alive(pid)

    def is_alive(self) -> bool:
        return self._is_running()


class JobRegistry:
    """In-memory registry; jobs started by a previous server run are re-listed from disk."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._load_previous()

    def _load_previous(self) -> None:
        """Surface old runs as read-only history entries."""
        for meta_path in sorted(JOBS_DIR.glob("*/meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job = Job(
                kind=meta.get("kind", "unknown"),
                argv=meta.get("argv") or [],
                cwd=meta.get("cwd"),
                params=meta.get("params") or {},
                title=meta.get("title", ""),
                cams=meta.get("cams") or [],
                job_id=meta.get("id"),
                write_meta=False,
            )
            job.status = meta.get("status", "unknown")
            job.returncode = meta.get("returncode")
            job.error = meta.get("error")
            job.created_at = meta.get("created_at", job.created_at)
            job.started_at = meta.get("started_at")
            job.ended_at = meta.get("ended_at")
            job.metrics = meta.get("metrics") or {}
            job.summary = meta.get("summary") or {}
            self._restore_lines(job)
            if job.status not in TERMINAL_STATES:
                recorded = _status_from_log(job)
                pid = meta.get("pid")
                if recorded is not None:
                    # It finished while the server was down; the log says how.
                    job.status, job.returncode = recorded
                elif pid and _pid_alive(int(pid)) and _pid_matches(int(pid), job.argv):
                    # Still running and the server is back: keep watching it.
                    job.resume_tailing(int(pid))
                else:
                    job.status = "unknown"
                    job.error = "server restarted while the job was running, and its process is gone"
            self._jobs[job.id] = job

    @staticmethod
    def _restore_lines(job: Job) -> None:
        """Replay a finished job's log so its metrics and curve survive a restart."""
        if not job.log_path.is_file():
            return
        job.metrics = {}
        job.summary = {}
        try:
            with job.log_path.open("rb") as handle:
                while True:
                    start = handle.tell()
                    raw = handle.readline()
                    if not raw or not raw.endswith(b"\n"):
                        job._restore_offset = start
                        break
                    for text in _split_line(raw.decode("utf-8", "replace")):
                        job._append_line(text)
                        try:
                            event = json.loads(text)
                        except (ValueError, TypeError):
                            job._handle_plain_line(text)
                        else:
                            if isinstance(event, dict):
                                job._handle_event(event, persist=False)
        except OSError:
            return

    # ── public api ───────────────────────────────────────────────────────────
    def create(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def stop(self, job_id: str, *, block: bool = True) -> dict:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.stop(block=block)

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.is_alive() or (job._reader and job._reader.is_alive()):
                raise ValueError("任务仍在运行或收尾，请停止后稍候再删除")
            if job.dir.exists():
                shutil.rmtree(job.dir)
            del self._jobs[job_id]

    def stop_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.is_alive():
                try:
                    job.stop(timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass


registry = JobRegistry()
