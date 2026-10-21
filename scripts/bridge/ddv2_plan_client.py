"""TCP client for DDV2 plan server (usable from chrono9)."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))

from ddv2_plan_protocol import recv_npz, send_npz  # noqa: E402


class DDV2PlanClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self._sock = sock
        print(f"[ddv2_client] connected {self.host}:{self.port}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                send_npz(self._sock, {"shutdown": np.array([1], dtype=np.int32)})
                recv_npz(self._sock)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def infer(self, obs: Dict[str, np.ndarray]) -> tuple:
        """Returns (trajectory (8,3), meta dict with infer_ms / rtt_ms)."""
        import time as _time

        self.connect()
        assert self._sock is not None
        payload = {
            "camera_feature": np.asarray(obs["camera_feature"], dtype=np.float32),
            "lidar_feature": np.asarray(obs["lidar_feature"], dtype=np.float32),
            "status_feature": np.asarray(obs["status_feature"], dtype=np.float32),
        }
        t0 = _time.perf_counter()
        send_npz(self._sock, payload)
        out = recv_npz(self._sock)
        rtt_ms = (_time.perf_counter() - t0) * 1000.0
        traj = np.asarray(out["trajectory"], dtype=np.float32)
        infer_ms = float(np.asarray(out["infer_ms"]).reshape(-1)[0]) if "infer_ms" in out else float("nan")
        return traj, {"infer_ms": infer_ms, "rtt_ms": rtt_ms}
