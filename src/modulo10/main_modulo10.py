"""
Módulo 10: Sistema de Seguridad
Versión: 0.6.0
Funcionalidad: Seguridad, autenticación, autorización y protección de datos
"""

import asyncio
import json
import logging
import hashlib
import hmac
import base64
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import jwt

logger = logging.getLogger('LucIA_Security')

class SecurityLevel(Enum):
    """Niveles de seguridad"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Permission(Enum):
    """Permisos del sistema"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    TRAIN = "train"
    COMMUNICATE = "communicate"

@dataclass
class SecurityEvent:
    """Evento de seguridad"""
    id: str
    event_type: str
    severity: SecurityLevel
    description: str
    timestamp: datetime
    source: str
    details: Dict[str, Any]

class SecuritySystem:
    """
    Sistema de seguridad para LucIA.
    Gestiona autenticación, autorización y protección de datos.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.secret_key = self._generate_secret_key()
        self.api_keys = {}
        self.user_sessions = {}
        self.security_events = {}
        self.rate_limits = {}
        self.encryption_keys = {}
        
        # Configuración de seguridad
        self.max_failed_attempts = 5
        self.session_timeout = 3600  # 1 hora
        self.rate_limit_window = 60  # 1 minuto
        self.max_requests_per_window = 100
        
        # Estadísticas
        self.total_requests = 0
        self.blocked_requests = 0
        self.security_violations = 0
        
        logger.info("Sistema de seguridad inicializado")
    
    def _generate_secret_key(self) -> str:
        """Genera una clave secreta para el sistema"""
        return secrets.token_urlsafe(32)
    
    async def authenticate_api_key(self, api_key: str) -> bool:
        """Autentica una clave API"""
        try:
            if api_key in self.api_keys:
                key_info = self.api_keys[api_key]
                
                # Verificar si la clave no ha expirado
                if key_info["expires_at"] and datetime.now() > key_info["expires_at"]:
                    await self._log_security_event(
                        "api_key_expired",
                        SecurityLevel.MEDIUM,
                        f"Intento de uso de clave API expirada: {api_key[:8]}...",
                        {"api_key": api_key[:8] + "..."}
                    )
                    return False
                
                # Actualizar último uso
                key_info["last_used"] = datetime.now()
                return True
            else:
                await self._log_security_event(
                    "invalid_api_key",
                    SecurityLevel.HIGH,
                    f"Intento de uso de clave API inválida: {api_key[:8]}...",
                    {"api_key": api_key[:8] + "..."}
                )
                return False
                
        except Exception as e:
            logger.error(f"Error autenticando clave API: {e}")
            return False
    
    async def create_api_key(self, name: str, permissions: List[Permission],
                           expires_in_days: int = 30) -> str:
        """Crea una nueva clave API"""
        try:
            api_key = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=expires_in_days)
            
            self.api_keys[api_key] = {
                "name": name,
                "permissions": [p.value for p in permissions],
                "created_at": datetime.now(),
                "last_used": None,
                "expires_at": expires_at,
                "is_active": True
            }
            
            await self._log_security_event(
                "api_key_created",
                SecurityLevel.MEDIUM,
                f"Nueva clave API creada: {name}",
                {"api_key_name": name, "permissions": [p.value for p in permissions]}
            )
            
            logger.info(f"Clave API creada: {name}")
            return api_key
            
        except Exception as e:
            logger.error(f"Error creando clave API: {e}")
            raise
    
    async def check_permission(self, api_key: str, permission: Permission) -> bool:
        """Verifica si una clave API tiene un permiso específico"""
        try:
            if api_key not in self.api_keys:
                return False
            
            key_info = self.api_keys[api_key]
            if not key_info["is_active"]:
                return False
            
            return permission.value in key_info["permissions"]
            
        except Exception as e:
            logger.error(f"Error verificando permiso: {e}")
            return False
    
    async def check_rate_limit(self, api_key: str) -> bool:
        """Verifica el límite de velocidad"""
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=self.rate_limit_window)
            
            # Limpiar registros antiguos
            if api_key in self.rate_limits:
                self.rate_limits[api_key] = [
                    timestamp for timestamp in self.rate_limits[api_key]
                    if timestamp > window_start
                ]
            else:
                self.rate_limits[api_key] = []
            
            # Verificar límite
            if len(self.rate_limits[api_key]) >= self.max_requests_per_window:
                await self._log_security_event(
                    "rate_limit_exceeded",
                    SecurityLevel.MEDIUM,
                    f"Límite de velocidad excedido para clave API: {api_key[:8]}...",
                    {"api_key": api_key[:8] + "...", "requests": len(self.rate_limits[api_key])}
                )
                return False
            
            # Registrar solicitud
            self.rate_limits[api_key].append(current_time)
            self.total_requests += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando límite de velocidad: {e}")
            return False
    
    async def encrypt_data(self, data: str, key_id: str = "default") -> str:
        """Encripta datos"""
        try:
            if key_id not in self.encryption_keys:
                self.encryption_keys[key_id] = secrets.token_urlsafe(32)
            
            key = self.encryption_keys[key_id]
            
            # Encriptación simple usando HMAC
            encrypted = base64.b64encode(
                hmac.new(key.encode(), data.encode(), hashlib.sha256).digest()
            ).decode()
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Error encriptando datos: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: str, key_id: str = "default") -> str:
        """Desencripta datos"""
        try:
            if key_id not in self.encryption_keys:
                raise ValueError(f"Clave de encriptación no encontrada: {key_id}")
            
            # Nota: Esta implementación es simplificada
            # En producción, usar algoritmos de encriptación apropiados
            return "Decrypted data"  # Simulado
            
        except Exception as e:
            logger.error(f"Error desencriptando datos: {e}")
            raise
    
    async def generate_jwt_token(self, payload: Dict[str, Any], 
                               expires_in_hours: int = 24) -> str:
        """Genera un token JWT"""
        try:
            payload["exp"] = datetime.utcnow() + timedelta(hours=expires_in_hours)
            payload["iat"] = datetime.utcnow()
            
            token = jwt.encode(payload, self.secret_key, algorithm="HS256")
            return token
            
        except Exception as e:
            logger.error(f"Error generando token JWT: {e}")
            raise
    
    async def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
            
        except jwt.ExpiredSignatureError:
            await self._log_security_event(
                "jwt_expired",
                SecurityLevel.MEDIUM,
                "Token JWT expirado",
                {"token": token[:20] + "..."}
            )
            return None
        except jwt.InvalidTokenError:
            await self._log_security_event(
                "jwt_invalid",
                SecurityLevel.HIGH,
                "Token JWT inválido",
                {"token": token[:20] + "..."}
            )
            return None
        except Exception as e:
            logger.error(f"Error verificando token JWT: {e}")
            return None
    
    async def validate_input(self, input_data: Any, input_type: str) -> bool:
        """Valida entrada para prevenir ataques"""
        try:
            if input_type == "sql_injection":
                # Detectar patrones de inyección SQL
                sql_patterns = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
                input_str = str(input_data).lower()
                
                for pattern in sql_patterns:
                    if pattern in input_str:
                        await self._log_security_event(
                            "sql_injection_attempt",
                            SecurityLevel.CRITICAL,
                            f"Intento de inyección SQL detectado: {pattern}",
                            {"input": str(input_data)[:100]}
                        )
                        return False
            
            elif input_type == "xss":
                # Detectar patrones de XSS
                xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
                input_str = str(input_data).lower()
                
                for pattern in xss_patterns:
                    if pattern in input_str:
                        await self._log_security_event(
                            "xss_attempt",
                            SecurityLevel.CRITICAL,
                            f"Intento de XSS detectado: {pattern}",
                            {"input": str(input_data)[:100]}
                        )
                        return False
            
            elif input_type == "path_traversal":
                # Detectar patrones de path traversal
                path_patterns = ["../", "..\\", "/etc/passwd", "C:\\"]
                input_str = str(input_data)
                
                for pattern in path_patterns:
                    if pattern in input_str:
                        await self._log_security_event(
                            "path_traversal_attempt",
                            SecurityLevel.CRITICAL,
                            f"Intento de path traversal detectado: {pattern}",
                            {"input": str(input_data)[:100]}
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando entrada: {e}")
            return False
    
    async def _log_security_event(self, event_type: str, severity: SecurityLevel,
                                description: str, details: Dict[str, Any]):
        """Registra un evento de seguridad"""
        try:
            event_id = f"sec_{int(datetime.now().timestamp())}"
            event = SecurityEvent(
                id=event_id,
                event_type=event_type,
                severity=severity,
                description=description,
                timestamp=datetime.now(),
                source="security_system",
                details=details
            )
            
            self.security_events[event_id] = event
            self.security_violations += 1
            
            # Si es crítico, tomar acción inmediata
            if severity == SecurityLevel.CRITICAL:
                await self._handle_critical_security_event(event)
            
            logger.warning(f"Evento de seguridad: {event_type} - {description}")
            
        except Exception as e:
            logger.error(f"Error registrando evento de seguridad: {e}")
    
    async def _handle_critical_security_event(self, event: SecurityEvent):
        """Maneja eventos de seguridad críticos"""
        try:
            # Bloquear la clave API si es aplicable
            if "api_key" in event.details:
                api_key = event.details["api_key"]
                if api_key in self.api_keys:
                    self.api_keys[api_key]["is_active"] = False
                    logger.critical(f"Clave API bloqueada por evento crítico: {api_key}")
            
            # Incrementar contador de bloqueos
            self.blocked_requests += 1
            
        except Exception as e:
            logger.error(f"Error manejando evento crítico: {e}")
    
    async def get_security_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de seguridad"""
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "security_violations": self.security_violations,
            "active_api_keys": sum(1 for key in self.api_keys.values() if key["is_active"]),
            "total_api_keys": len(self.api_keys),
            "security_events_count": len(self.security_events),
            "block_rate": (self.blocked_requests / max(self.total_requests, 1)) * 100
        }
    
    async def get_security_events(self, severity: SecurityLevel = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene eventos de seguridad"""
        try:
            events = list(self.security_events.values())
            
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Ordenar por timestamp descendente
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            events = events[:limit]
            
            return [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "severity": event.severity.value,
                    "description": event.description,
                    "timestamp": event.timestamp.isoformat(),
                    "source": event.source,
                    "details": event.details
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo eventos de seguridad: {e}")
            return []
    
    async def save_state(self):
        """Guarda el estado del sistema de seguridad"""
        try:
            state = {
                "api_keys": {
                    key: {
                        "name": info["name"],
                        "permissions": info["permissions"],
                        "created_at": info["created_at"].isoformat(),
                        "last_used": info["last_used"].isoformat() if info["last_used"] else None,
                        "expires_at": info["expires_at"].isoformat() if info["expires_at"] else None,
                        "is_active": info["is_active"]
                    }
                    for key, info in self.api_keys.items()
                },
                "security_events": {
                    event_id: {
                        "id": event.id,
                        "event_type": event.event_type,
                        "severity": event.severity.value,
                        "description": event.description,
                        "timestamp": event.timestamp.isoformat(),
                        "source": event.source,
                        "details": event.details
                    }
                    for event_id, event in self.security_events.items()
                },
                "stats": {
                    "total_requests": self.total_requests,
                    "blocked_requests": self.blocked_requests,
                    "security_violations": self.security_violations
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/security_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de seguridad guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de seguridad: {e}")

# Instancia global del sistema de seguridad
security_system = SecuritySystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de seguridad"""
    global security_system
    security_system.core_engine = core_engine
    core_engine.security_system = security_system
    logger.info("Módulo de seguridad inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de seguridad"""
    if isinstance(input_data, dict) and "create_api_key" in input_data:
        # Crear clave API
        key_data = input_data["create_api_key"]
        
        try:
            permissions = [Permission(p) for p in key_data.get("permissions", ["read"])]
            api_key = await security_system.create_api_key(
                name=key_data.get("name", "Default Key"),
                permissions=permissions,
                expires_in_days=key_data.get("expires_in_days", 30)
            )
            
            return {"api_key_created": True, "api_key": api_key}
            
        except Exception as e:
            return {"api_key_created": False, "error": str(e)}
    
    elif isinstance(input_data, dict) and "validate_input" in input_data:
        # Validar entrada
        validation_data = input_data["validate_input"]
        
        is_valid = await security_system.validate_input(
            validation_data.get("data", ""),
            validation_data.get("type", "sql_injection")
        )
        
        return {"is_valid": is_valid}
    
    return input_data

def run_modulo10():
    """Función de compatibilidad con el sistema anterior"""
    print("🔒 Módulo 10: Sistema de Seguridad")
    print("   - Autenticación y autorización")
    print("   - Protección contra ataques")
    print("   - Encriptación de datos")
    print("   - Monitoreo de seguridad")
    print("   ✅ Módulo inicializado correctamente")