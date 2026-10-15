#!/usr/bin/env python3
"""极简 policy 客户端 —— 与 LoongForge `eval/transport/client.py` 协议对齐。

为什么自己写一份：边端只装了 websockets/msgpack，**不装 LoongForge 整个仓库**
（也不该装 —— 那边是训练环境）。协议很短，复制四行消息格式即可，避免把训练侧
依赖树拖到采集机上。

协议（与 LoongForge `protocol/schema.py` 对齐）
---------
1. 客户端连上后，**server 先发一条 metadata**（msgpack），客户端必须先收；
2. 之后每条请求 = ``{"type": ..., "request_id": ..., "protocol_version": "1.0",
   "payload": {...}}``，msgpack 打包；
3. 响应 = msgpack dict；``type == "inference_result"`` 时动作在 ``resp["data"]``。

关键 payload 字段（对应 server 侧 ``GenericPredictActionPolicy.predict_action`` 的签名）：
``images``（batch-of-view-list，如 ``[[front, wrist]]``）、``instructions``（list-of-str）、
``state``（(B, state_dim) 或 None）、以及两个 RPC 控制开关：
``return_action_chunk``（True=返回整段 chunk，而不是只回一步）、
``disable_action_cache``（True=强制重新推理，不复用 server 端 chunk 缓存）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import msgpack
import numpy as np
import websockets.sync.client

# ★ 必须用 LoongForge **自带的** msgpack_numpy（已 vendor 到本目录）：
#   第三方 PyPI 版 msgpack_numpy 的 ndarray 编码格式与 LoongForge 不同
#   ⇒ server 端 unpack 后拿到 dict，报 "pic should be PIL Image or ndarray. Got <class 'dict'>"。
#   本文件是 loongforge/embodied/eval/transport/msgpack_numpy.py 的逐字副本（Apache-2.0）。
from . import msgpack_numpy

PROTOCOL_VERSION = "1.0"

#: ⚠️ 铁律：websockets/urllib 认 HTTP_PROXY **且不豁免局域网 IP** ⇒ 若边端 shell 里
#: 有代理变量，连 ws://172.16.107.56 会被劫持到代理上（表现为超时或莫名其妙的重定向）。
#: LoongForge 的 client 也是这么处理的，这里保持一致。
_PROXY_ENV = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
              "NO_PROXY", "no_proxy")


def _clear_proxy_env() -> None:
    for k in _PROXY_ENV:
        os.environ.pop(k, None)


class PolicyClient:
    """连到 LoongForge policy server 的同步 WebSocket 客户端。"""

    def __init__(self, host: str = "172.16.107.56", port: int = 10093, timeout: float = 300.0) -> None:
        self._uri = f"ws://{host}:{port}"
        self._timeout = timeout
        self._ws = None
        self._metadata: Dict[str, Any] = {}
        # ★ 用 LoongForge 自带的 Packer（无参构造，pack() 每次返回 bytes）
        self._packer = msgpack_numpy.Packer()

    # ------------------------------------------------------------------ 连接
    def connect(self) -> Dict[str, Any]:
        _clear_proxy_env()
        self._ws = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None, open_timeout=60
        )
        raw = self._ws.recv(timeout=self._timeout)
        self._metadata = msgpack.unpackb(raw, raw=False)
        got = self._metadata.get("protocol_version")
        if got != PROTOCOL_VERSION:
            raise RuntimeError(f"protocol_version 不匹配: {got} != {PROTOCOL_VERSION}")
        return self._metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    # ------------------------------------------------------------------ 请求
    def request(self, message_type: str, payload: Optional[Dict[str, Any]] = None,
                request_id: str = "default") -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("未连接；先调 connect()")
        msg = {
            "type": message_type,
            "request_id": request_id,
            "protocol_version": PROTOCOL_VERSION,
            "payload": payload or {},
        }
        self._ws.send(self._packer.pack(msg))
        raw = self._ws.recv(timeout=self._timeout)
        if isinstance(raw, str):
            raise RuntimeError(raw)
        return msgpack_numpy.unpackb(raw)

    def ping(self) -> Dict[str, Any]:
        return self.request("ping")

    def reset(self, episode_id: str = "default") -> Dict[str, Any]:
        return self.request("reset", {"episode_id": episode_id})

    def predict_action(self, **payload: Any) -> Dict[str, Any]:
        """发一次 infer；返回 ``resp["data"]``（含 ``actions`` / ``inference_latency_ms``）。"""
        resp = self.request("infer", payload)
        if not resp.get("ok", False):
            raise RuntimeError(f"infer 失败: {resp}")
        return resp.get("data", {})


def as_chunk(data: Dict[str, Any]) -> np.ndarray:
    """把响应里的 actions 规范成 ``(horizon, dim)``。"""
    arr = np.asarray(data["actions"], dtype=np.float32)
    if arr.ndim == 3:      # (B, horizon, dim)
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        pass
    return arr
