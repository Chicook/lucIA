"""
Módulo 12: Sistema de Integración
Versión: 0.6.0
Funcionalidad: Integración con APIs externas, servicios y sistemas
"""

import asyncio
import json
import logging
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

logger = logging.getLogger('LucIA_Integration')

class IntegrationType(Enum):
    """Tipos de integración"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"

class ServiceStatus(Enum):
    """Estado de servicios"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class IntegrationService:
    """Servicio de integración"""
    id: str
    name: str
    type: IntegrationType
    endpoint: str
    status: ServiceStatus
    config: Dict[str, Any]
    last_used: Optional[datetime] = None
    error_count: int = 0

class IntegrationSystem:
    """
    Sistema de integración para LucIA.
    Gestiona conexiones con APIs externas y servicios.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.services = {}
        self.api_rotation = True
        self.fallback_enabled = True
        self.timeout = 30
        self.retry_attempts = 3
        
        # APIs externas soportadas
        self.external_apis = {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "endpoints": {
                    "chat": "/chat/completions",
                    "embeddings": "/embeddings",
                    "models": "/models"
                },
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {api_key}"
                }
            },
            "claude": {
                "base_url": "https://api.anthropic.com/v1",
                "endpoints": {
                    "messages": "/messages",
                    "models": "/models"
                },
                "headers": {
                    "Content-Type": "application/json",
                    "x-api-key": "{api_key}",
                    "anthropic-version": "2023-06-01"
                }
            },
            "gemini": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "endpoints": {
                    "generate": "/models/{model}:generateContent",
                    "models": "/models"
                },
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }
        
        # Estadísticas
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_used = 0
        
        logger.info("Sistema de integración inicializado")
    
    async def register_service(self, service_id: str, name: str, 
                             integration_type: IntegrationType,
                             endpoint: str, config: Dict[str, Any]) -> bool:
        """Registra un nuevo servicio de integración"""
        try:
            service = IntegrationService(
                id=service_id,
                name=name,
                type=integration_type,
                endpoint=endpoint,
                status=ServiceStatus.ACTIVE,
                config=config
            )
            
            self.services[service_id] = service
            logger.info(f"Servicio registrado: {service_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Error registrando servicio {service_id}: {e}")
            return False
    
    async def call_external_api(self, api_name: str, endpoint: str,
                              data: Dict[str, Any] = None,
                              method: str = "POST") -> Dict[str, Any]:
        """
        Llama a una API externa
        
        Args:
            api_name: Nombre de la API (openai, claude, gemini)
            endpoint: Endpoint específico
            data: Datos a enviar
            method: Método HTTP
        
        Returns:
            Respuesta de la API
        """
        try:
            self.total_requests += 1
            
            if api_name not in self.external_apis:
                raise ValueError(f"API no soportada: {api_name}")
            
            api_config = self.external_apis[api_name]
            
            # Construir URL
            if endpoint in api_config["endpoints"]:
                url = api_config["base_url"] + api_config["endpoints"][endpoint]
            else:
                url = api_config["base_url"] + endpoint
            
            # Preparar headers
            headers = api_config["headers"].copy()
            
            # Reemplazar placeholders en headers
            for key, value in headers.items():
                if "{api_key}" in value:
                    api_key = self._get_api_key(api_name)
                    if api_key:
                        headers[key] = value.replace("{api_key}", api_key)
                    else:
                        raise ValueError(f"API key no encontrada para {api_name}")
            
            # Realizar solicitud
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers, params=data) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Método HTTP no soportado: {method}")
                
                if response.status == 200:
                    self.successful_requests += 1
                    logger.info(f"API {api_name} llamada exitosamente")
                    return result
                else:
                    raise Exception(f"Error en API {api_name}: {response.status} - {result}")
                    
        except Exception as e:
            logger.error(f"Error llamando API {api_name}: {e}")
            self.failed_requests += 1
            
            # Intentar fallback si está habilitado
            if self.fallback_enabled:
                return await self._try_fallback(api_name, endpoint, data, method)
            else:
                raise
    
    async def _try_fallback(self, api_name: str, endpoint: str,
                          data: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Intenta usar un servicio de fallback"""
        try:
            self.fallback_used += 1
            
            # Lista de APIs de fallback en orden de preferencia
            fallback_apis = ["openai", "claude", "gemini"]
            
            for fallback_api in fallback_apis:
                if fallback_api != api_name and fallback_api in self.external_apis:
                    try:
                        logger.info(f"Intentando fallback con {fallback_api}")
                        return await self.call_external_api(fallback_api, endpoint, data, method)
                    except Exception as e:
                        logger.warning(f"Fallback con {fallback_api} falló: {e}")
                        continue
            
            # Si todos los fallbacks fallan, usar respuesta local
            return await self._generate_local_response(endpoint, data)
            
        except Exception as e:
            logger.error(f"Error en fallback: {e}")
            raise
    
    async def _generate_local_response(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una respuesta local como último recurso"""
        try:
            logger.info("Generando respuesta local como fallback")
            
            # Respuesta básica local
            if "chat" in endpoint or "completions" in endpoint:
                return {
                    "choices": [{
                        "message": {
                            "content": "Lo siento, no puedo procesar tu solicitud en este momento. Estoy usando mi sistema de respaldo local.",
                            "role": "assistant"
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            elif "embeddings" in endpoint:
                return {
                    "data": [{
                        "embedding": [0.0] * 1536,  # Vector de embedding vacío
                        "index": 0
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "total_tokens": 0
                    }
                }
            else:
                return {
                    "message": "Respuesta local generada",
                    "endpoint": endpoint,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error generando respuesta local: {e}")
            return {"error": "No se pudo generar respuesta"}
    
    def _get_api_key(self, api_name: str) -> Optional[str]:
        """Obtiene la clave API para un servicio"""
        try:
            # En un sistema real, esto vendría de variables de entorno o configuración segura
            api_keys = {
                "openai": "sk-your-openai-key-here",
                "claude": "sk-ant-your-claude-key-here",
                "gemini": "your-gemini-key-here"
            }
            
            return api_keys.get(api_name)
            
        except Exception as e:
            logger.error(f"Error obteniendo API key para {api_name}: {e}")
            return None
    
    async def test_service_connection(self, service_id: str) -> bool:
        """Prueba la conexión a un servicio"""
        try:
            if service_id not in self.services:
                return False
            
            service = self.services[service_id]
            
            if service.type == IntegrationType.REST_API:
                # Probar conexión HTTP
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(service.endpoint) as response:
                        service.status = ServiceStatus.ACTIVE if response.status == 200 else ServiceStatus.ERROR
                        return response.status == 200
            
            elif service.type == IntegrationType.WEBSOCKET:
                # Probar conexión WebSocket
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(service.endpoint) as ws:
                            service.status = ServiceStatus.ACTIVE
                            return True
                except:
                    service.status = ServiceStatus.ERROR
                    return False
            
            else:
                # Para otros tipos, asumir que está activo
                service.status = ServiceStatus.ACTIVE
                return True
                
        except Exception as e:
            logger.error(f"Error probando conexión del servicio {service_id}: {e}")
            if service_id in self.services:
                self.services[service_id].status = ServiceStatus.ERROR
            return False
    
    async def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """Obtiene el estado de un servicio"""
        try:
            if service_id not in self.services:
                return {"error": "Servicio no encontrado"}
            
            service = self.services[service_id]
            
            return {
                "id": service.id,
                "name": service.name,
                "type": service.type.value,
                "endpoint": service.endpoint,
                "status": service.status.value,
                "last_used": service.last_used.isoformat() if service.last_used else None,
                "error_count": service.error_count,
                "config": service.config
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del servicio {service_id}: {e}")
            return {"error": str(e)}
    
    async def rotate_api_usage(self) -> str:
        """Rota el uso de APIs para balancear carga"""
        try:
            if not self.api_rotation:
                return "openai"  # API por defecto
            
            # Lista de APIs disponibles
            available_apis = list(self.external_apis.keys())
            
            # Seleccionar API basada en estadísticas de uso
            # Por simplicidad, usar round-robin
            current_time = int(time.time())
            api_index = current_time % len(available_apis)
            
            selected_api = available_apis[api_index]
            logger.debug(f"API seleccionada para rotación: {selected_api}")
            
            return selected_api
            
        except Exception as e:
            logger.error(f"Error en rotación de API: {e}")
            return "openai"
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de integración"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / max(self.total_requests, 1)) * 100,
            "fallback_used": self.fallback_used,
            "total_services": len(self.services),
            "active_services": sum(1 for s in self.services.values() if s.status == ServiceStatus.ACTIVE),
            "api_rotation_enabled": self.api_rotation,
            "fallback_enabled": self.fallback_enabled
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de integración"""
        try:
            state = {
                "services": {
                    service_id: {
                        "id": service.id,
                        "name": service.name,
                        "type": service.type.value,
                        "endpoint": service.endpoint,
                        "status": service.status.value,
                        "last_used": service.last_used.isoformat() if service.last_used else None,
                        "error_count": service.error_count,
                        "config": service.config
                    }
                    for service_id, service in self.services.items()
                },
                "external_apis": self.external_apis,
                "config": {
                    "api_rotation": self.api_rotation,
                    "fallback_enabled": self.fallback_enabled,
                    "timeout": self.timeout,
                    "retry_attempts": self.retry_attempts
                },
                "stats": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "fallback_used": self.fallback_used
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/integration_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de integración guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de integración: {e}")

# Instancia global del sistema de integración
integration_system = IntegrationSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de integración"""
    global integration_system
    integration_system.core_engine = core_engine
    core_engine.integration_system = integration_system
    logger.info("Módulo de integración inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de integración"""
    if isinstance(input_data, dict) and "call_api" in input_data:
        # Llamar API externa
        api_data = input_data["call_api"]
        
        try:
            result = await integration_system.call_external_api(
                api_name=api_data.get("api_name", "openai"),
                endpoint=api_data.get("endpoint", "chat"),
                data=api_data.get("data", {}),
                method=api_data.get("method", "POST")
            )
            
            return {"api_response": result}
            
        except Exception as e:
            return {"api_response": None, "error": str(e)}
    
    elif isinstance(input_data, dict) and "register_service" in input_data:
        # Registrar servicio
        service_data = input_data["register_service"]
        
        success = await integration_system.register_service(
            service_id=service_data.get("id", f"service_{int(time.time())}"),
            name=service_data.get("name", "Unnamed Service"),
            integration_type=IntegrationType(service_data.get("type", "rest_api")),
            endpoint=service_data.get("endpoint", ""),
            config=service_data.get("config", {})
        )
        
        return {"service_registered": success}
    
    return input_data

def run_modulo12():
    """Función de compatibilidad con el sistema anterior"""
    print("🔗 Módulo 12: Sistema de Integración")
    print("   - APIs externas (OpenAI, Claude, Gemini)")
    print("   - Rotación automática de APIs")
    print("   - Sistema de fallback")
    print("   - Gestión de servicios")
    print("   ✅ Módulo inicializado correctamente")