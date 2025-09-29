import json
import socket
import struct
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class MessageType(Enum):
    DISCOVERY_HELLO = "discovery_hello"
    DISCOVERY_PONG = "discovery_pong"
    USER_MESSAGE = "user_message"
    SYSTEM = "system"


@dataclass
class P2PMessage:
    id: str
    sender_id: str
    type: str
    payload: Dict[str, Any]
    ts: str

    @staticmethod
    def create(sender_id: str, type: MessageType, payload: Dict[str, Any]) -> "P2PMessage":
        return P2PMessage(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            type=type.value,
            payload=payload,
            ts=datetime.utcnow().isoformat(),
        )

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")


def pack_length_prefixed(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        buf += chunk
    return buf


def recv_length_prefixed(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length > 10 * 1024 * 1024:
        raise ValueError("Frame too large")
    return recv_exact(sock, length)


