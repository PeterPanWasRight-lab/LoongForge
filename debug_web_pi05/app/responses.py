#!/usr/bin/env python
"""Tiny response objects so API handlers stay transport-agnostic."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class Response:
    status = 200
    content_type = "text/plain; charset=utf-8"

    def body(self) -> bytes:
        raise NotImplementedError


class Json(Response):
    content_type = "application/json; charset=utf-8"

    def __init__(self, data: Any, status: int = 200):
        self.data = data
        self.status = status

    def body(self) -> bytes:
        return json.dumps(self.data, ensure_ascii=False, default=str).encode("utf-8")


class Text(Response):
    def __init__(self, text: str, status: int = 200, content_type: str | None = None):
        self.text = text
        self.status = status
        if content_type:
            self.content_type = content_type

    def body(self) -> bytes:
        return self.text.encode("utf-8")


class Error(Json):
    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__({"ok": False, "error": message, **extra}, status=status)


class Raw(Response):
    """A single in-memory blob (a JPEG frame, a file we already read)."""

    def __init__(self, data: bytes, content_type: str, status: int = 200, headers: dict | None = None):
        self.data = data
        self.content_type = content_type
        self.status = status
        self.headers = headers or {}

    def body(self) -> bytes:
        return self.data


class File(Response):
    """Served with a size so the browser can handle range requests / caching."""

    def __init__(self, path: Path, content_type: str | None = None, status: int = 200):
        self.path = Path(path)
        self.status = status
        self.content_type = content_type or guess_content_type(self.path)

    def body(self) -> bytes:
        return self.path.read_bytes()

    def size(self) -> int:
        return self.path.stat().st_size


class Stream(Response):
    """Chunked stream: MJPEG camera feeds and SSE event feeds.

    ``generator`` yields already-framed byte chunks, so the HTTP layer only has
    to write them out and flush.
    """

    def __init__(
        self,
        generator: Iterator[bytes],
        content_type: str,
        status: int = 200,
        headers: dict | None = None,
    ):
        self.generator = generator
        self.content_type = content_type
        self.status = status
        self.headers = headers or {}


_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".gif": "image/gif",
    ".webm": "video/webm",
    ".csv": "text/csv; charset=utf-8",
}


def guess_content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
