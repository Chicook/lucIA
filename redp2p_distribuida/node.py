import json
import logging
import socket
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .protocol import (
    P2PMessage,
    MessageType,
    pack_length_prefixed,
    recv_length_prefixed,
)
from .discovery import LANDiscovery


logger = logging.getLogger("redp2p")


@dataclass
class P2PConfig:
    tcp_port: int = 45056
    discovery_port: int = 45055
    broadcast_interval: float = 3.0


class P2PNode:
    def __init__(self, config: Optional[P2PConfig] = None) -> None:
        self.config = config or P2PConfig()
        self.node_id = str(uuid.uuid4())
        self.peers: Dict[str, Tuple[str, int]] = {}
        self._stop_event = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._discovery = LANDiscovery(self.node_id, self.config.discovery_port, self.config.broadcast_interval)
        self._discovery.on_peer(self._handle_discovered_peer)

    def start(self) -> None:
        logger.info("Starting P2P node %s", self.node_id[:8])
        self._stop_event.clear()
        self._server_thread = threading.Thread(target=self._tcp_server_loop, daemon=True)
        self._server_thread.start()
        self._discovery.start()

    def stop(self) -> None:
        logger.info("Stopping P2P node %s", self.node_id[:8])
        self._stop_event.set()
        self._discovery.stop()

    def _handle_discovered_peer(self, peer_id: str, host: str, port: int) -> None:
        # We use host as peer_id proxy; could be extended by challenge/response
        if host == self._get_local_ip():
            return
        if host not in self.peers:
            self.peers[host] = (host, self.config.tcp_port)
            logger.info("Discovered peer: %s", host)

    def _tcp_server_loop(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("", self.config.tcp_port))
            srv.listen(50)
            while not self._stop_event.is_set():
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
        finally:
            srv.close()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        with conn:
            try:
                frame = recv_length_prefixed(conn)
                obj = json.loads(frame.decode("utf-8"))
                logger.info("Received from %s: %s", addr, obj.get("type"))
            except Exception as exc:
                logger.error("Error handling client %s: %s", addr, exc)

    def send_text(self, host: str, text: str) -> bool:
        peer = self.peers.get(host)
        if not peer:
            peer = (host, self.config.tcp_port)
        try:
            sock = socket.create_connection(peer, timeout=3.0)
            try:
                msg = P2PMessage.create(
                    sender_id=self.node_id,
                    type=MessageType.USER_MESSAGE,
                    payload={"text": text},
                )
                data = pack_length_prefixed(msg.to_json_bytes())
                sock.sendall(data)
                return True
            finally:
                sock.close()
        except Exception as exc:
            logger.error("Failed to send to %s: %s", host, exc)
            return False

    @staticmethod
    def _get_local_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            s.close()


