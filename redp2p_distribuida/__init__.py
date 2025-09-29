"""
Paquete: redp2p_distribuida
Descripción: Infraestructura básica para red P2P distribuida (PC a PC) en LAN.

Componentes:
- protocol: Definiciones de protocolo, mensajes y utilidades de serialización.
- discovery: Descubrimiento de peers vía UDP broadcast/multicast.
- node: Nodo P2P con servidor TCP y cliente para mensajería JSON con framing.
"""

from .node import P2PNode, P2PConfig
from .protocol import P2PMessage, MessageType

__all__ = [
    "P2PNode",
    "P2PConfig",
    "P2PMessage",
    "MessageType",
]


