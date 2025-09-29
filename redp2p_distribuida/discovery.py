import socket
import threading
import time
from typing import Callable, Dict

from .protocol import P2PMessage, MessageType


class LANDiscovery:
    def __init__(self, node_id: str, discovery_port: int = 45055, broadcast_interval: float = 3.0):
        self.node_id = node_id
        self.discovery_port = discovery_port
        self.broadcast_interval = broadcast_interval
        self._stop_event = threading.Event()
        self._on_peer: Callable[[str, str, int], None] | None = None

    def on_peer(self, handler: Callable[[str, str, int], None]) -> None:
        self._on_peer = handler

    def start(self) -> None:
        self._stop_event.clear()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop_event.set()

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            while not self._stop_event.is_set():
                msg = P2PMessage.create(
                    sender_id=self.node_id,
                    type=MessageType.DISCOVERY_HELLO,
                    payload={"port": self.discovery_port},
                )
                sock.sendto(msg.to_json_bytes(), ("255.255.255.255", self.discovery_port))
                time.sleep(self.broadcast_interval)
        finally:
            sock.close()

    def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.discovery_port))
            while not self._stop_event.is_set():
                data, addr = sock.recvfrom(65535)
                try:
                    text = data.decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if MessageType.DISCOVERY_HELLO.value in text or MessageType.DISCOVERY_PONG.value in text:
                    if self._on_peer is not None:
                        self._on_peer(addr[0], addr[0], self.discovery_port)
        finally:
            sock.close()


