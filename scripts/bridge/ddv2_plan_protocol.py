"""Length-prefixed npz IPC between chrono9 client and ddv2 plan server."""

from __future__ import annotations

import io
import socket
import struct
from typing import Dict

import numpy as np

HEADER = struct.Struct("!Q")  # payload byte length


def send_npz(sock: socket.socket, arrays: Dict[str, np.ndarray]) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    payload = buf.getvalue()
    sock.sendall(HEADER.pack(len(payload)) + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = sock.recv(min(65536, n - got))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv_npz(sock: socket.socket) -> Dict[str, np.ndarray]:
    raw_len = recv_exact(sock, HEADER.size)
    (nbytes,) = HEADER.unpack(raw_len)
    payload = recv_exact(sock, nbytes)
    data = np.load(io.BytesIO(payload), allow_pickle=False)
    return {k: data[k] for k in data.files}
