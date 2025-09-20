"""
Módulo 4: Interfaz de Entrenamiento por otras IAs
Versión: 0.6.0
Funcionalidad: Interfaz estándar para entrenamiento por IAs externas, APIs y protocolos
"""

import asyncio
import json
import os
import logging
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import base64
import requests
from urllib.parse import urlparse

logger = logging.getLogger('LucIA_Training')

class TrainingProtocol(Enum):
    """Protocolos de entrenamiento soportados"""
    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    CUSTOM = "custom"

class TrainingStatus(Enum):
    """Estados de entrenamiento"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingSession:
    """Sesión de entrenamiento"""
    id: str
    trainer_ai_id: str
    protocol: TrainingProtocol
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    training_data: Dict[str, Any] = None
    model_config: Dict[str, Any] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    results: Dict[str, Any] = None

class AITrainingInterface:
    """
    Interfaz estándar para entrenamiento por IAs externas.
    Soporta múltiples protocolos y formatos de datos.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.training_sessions = {}
        self.active_trainers = {}
        self.training_queue = asyncio.Queue()
        self.supported_protocols = [TrainingProtocol.REST_API, TrainingProtocol.WEBSOCKET]
        self.api_endpoints = {}
        self.websocket_connections = {}
        
        # Configuración de seguridad
        self.api_key = self._generate_api_key()
        self.max_training_sessions = 10
        self.session_timeout = 3600  # 1 hora
        
        # Estadísticas
        self.total_sessions = 0
        self.successful_sessions = 0
        self.failed_sessions = 0
        
        # Inicializar directorios
        os.makedirs("data/training", exist_ok=True)
        os.makedirs("logs/training", exist_ok=True)
        
        logger.info("Interfaz de entrenamiento inicializada")
    
    def _generate_api_key(self) -> str:
        """Genera una clave API para autenticación"""
        return hashlib.sha256(f"lucia_training_{uuid.uuid4()}_{int(time.time())}".encode()).hexdigest()
    
    async def initialize_module(self, core_engine):
        """Inicializa el módulo de entrenamiento"""
        self.core_engine = core_engine
        core_engine.training_interface = self
        
        # Iniciar procesador de cola de entrenamiento
        asyncio.create_task(self._training_queue_processor())
        
        # Iniciar limpiador de sesiones
        asyncio.create_task(self._session_cleanup())
        
        logger.info("Módulo de entrenamiento inicializado")
    
    async def create_training_session(self, trainer_ai_id: str, protocol: TrainingProtocol,
                                    training_data: Dict[str, Any], 
                                    model_config: Dict[str, Any] = None) -> str:
        """
        Crea una nueva sesión de entrenamiento
        
        Args:
            trainer_ai_id: ID de la IA que va a entrenar
            protocol: Protocolo de comunicación
            training_data: Datos de entrenamiento
            model_config: Configuración del modelo
        
        Returns:
            ID de la sesión de entrenamiento
        """
        try:
            # Verificar límite de sesiones
            if len(self.training_sessions) >= self.max_training_sessions:
                raise Exception("Límite máximo de sesiones de entrenamiento alcanzado")
            
            # Crear nueva sesión
            session_id = str(uuid.uuid4())
            session = TrainingSession(
                id=session_id,
                trainer_ai_id=trainer_ai_id,
                protocol=protocol,
                status=TrainingStatus.PENDING,
                created_at=datetime.now(),
                training_data=training_data,
                model_config=model_config or {}
            )
            
            self.training_sessions[session_id] = session
            self.total_sessions += 1
            
            # Agregar a cola de procesamiento
            await self.training_queue.put(session_id)
            
            logger.info(f"Sesión de entrenamiento creada: {session_id} por {trainer_ai_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creando sesión de entrenamiento: {e}")
            raise
    
    async def start_training_session(self, session_id: str) -> bool:
        """
        Inicia una sesión de entrenamiento
        
        Args:
            session_id: ID de la sesión
        
        Returns:
            True si se inició exitosamente
        """
        try:
            if session_id not in self.training_sessions:
                logger.error(f"Sesión {session_id} no encontrada")
                return False
            
            session = self.training_sessions[session_id]
            
            if session.status != TrainingStatus.PENDING:
                logger.warning(f"Sesión {session_id} no está en estado PENDING")
                return False
            
            # Actualizar estado
            session.status = TrainingStatus.IN_PROGRESS
            session.started_at = datetime.now()
            
            # Iniciar entrenamiento según protocolo
            if session.protocol == TrainingProtocol.REST_API:
                success = await self._start_rest_api_training(session)
            elif session.protocol == TrainingProtocol.WEBSOCKET:
                success = await self._start_websocket_training(session)
            else:
                logger.error(f"Protocolo no soportado: {session.protocol}")
                return False
            
            if success:
                logger.info(f"Entrenamiento iniciado para sesión {session_id}")
                return True
            else:
                session.status = TrainingStatus.FAILED
                session.error_message = "Error iniciando entrenamiento"
                return False
                
        except Exception as e:
            logger.error(f"Error iniciando sesión {session_id}: {e}")
            if session_id in self.training_sessions:
                self.training_sessions[session_id].status = TrainingStatus.FAILED
                self.training_sessions[session_id].error_message = str(e)
            return False
    
    async def _start_rest_api_training(self, session: TrainingSession) -> bool:
        """Inicia entrenamiento vía REST API"""
        try:
            # Preparar datos de entrenamiento
            training_payload = {
                "session_id": session.id,
                "training_data": session.training_data,
                "model_config": session.model_config,
                "api_key": self.api_key
            }
            
            # Enviar a sistema de aprendizaje
            if self.core_engine and hasattr(self.core_engine, 'learning_engine'):
                # Crear modelo temporal
                model_name = f"external_{session.trainer_ai_id}_{session.id}"
                
                # Extraer características y objetivos
                features = session.training_data.get('features', [])
                targets = session.training_data.get('targets', [])
                
                if len(features) > 0 and len(targets) > 0:
                    # Entrenar modelo
                    metrics = await self.core_engine.learning_engine.train_model(
                        model_name, features, targets
                    )
                    
                    # Actualizar sesión con resultados
                    session.results = {
                        "model_name": model_name,
                        "metrics": metrics,
                        "training_samples": len(features)
                    }
                    
                    session.status = TrainingStatus.COMPLETED
                    session.completed_at = datetime.now()
                    session.progress = 100.0
                    
                    self.successful_sessions += 1
                    logger.info(f"Entrenamiento REST API completado para sesión {session.id}")
                    return True
                else:
                    session.error_message = "Datos de entrenamiento inválidos"
                    return False
            else:
                session.error_message = "Sistema de aprendizaje no disponible"
                return False
                
        except Exception as e:
            logger.error(f"Error en entrenamiento REST API: {e}")
            session.error_message = str(e)
            return False
    
    async def _start_websocket_training(self, session: TrainingSession) -> bool:
        """Inicia entrenamiento vía WebSocket"""
        try:
            # Implementar lógica de WebSocket
            # Por ahora, usar la misma lógica que REST API
            return await self._start_rest_api_training(session)
            
        except Exception as e:
            logger.error(f"Error en entrenamiento WebSocket: {e}")
            session.error_message = str(e)
            return False
    
    async def external_training(self, training_data: Any, ai_interface: str) -> bool:
        """
        Permite entrenamiento por IAs externas
        
        Args:
            training_data: Datos de entrenamiento
            ai_interface: Interfaz de la IA externa
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        try:
            logger.info(f"Entrenamiento externo solicitado por {ai_interface}")
            
            # Determinar protocolo basado en la interfaz
            protocol = TrainingProtocol.REST_API
            if "websocket" in ai_interface.lower():
                protocol = TrainingProtocol.WEBSOCKET
            
            # Crear sesión de entrenamiento
            session_id = await self.create_training_session(
                trainer_ai_id=ai_interface,
                protocol=protocol,
                training_data=training_data
            )
            
            # Iniciar entrenamiento
            success = await self.start_training_session(session_id)
            
            if success:
                logger.info(f"Entrenamiento externo completado exitosamente")
                return True
            else:
                logger.warning(f"Entrenamiento externo falló")
                return False
                
        except Exception as e:
            logger.error(f"Error en entrenamiento externo: {e}")
            return False
    
    async def get_training_session(self, session_id: str) -> Optional[TrainingSession]:
        """Obtiene información de una sesión de entrenamiento"""
        return self.training_sessions.get(session_id)
    
    async def cancel_training_session(self, session_id: str) -> bool:
        """Cancela una sesión de entrenamiento"""
        try:
            if session_id not in self.training_sessions:
                return False
            
            session = self.training_sessions[session_id]
            
            if session.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                return False
            
            session.status = TrainingStatus.CANCELLED
            session.completed_at = datetime.now()
            
            logger.info(f"Sesión de entrenamiento {session_id} cancelada")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando sesión {session_id}: {e}")
            return False
    
    async def get_training_progress(self, session_id: str) -> Dict[str, Any]:
        """Obtiene el progreso de una sesión de entrenamiento"""
        if session_id not in self.training_sessions:
            return {"error": "Sesión no encontrada"}
        
        session = self.training_sessions[session_id]
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "progress": session.progress,
            "created_at": session.created_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "error_message": session.error_message,
            "results": session.results
        }
    
    async def _training_queue_processor(self):
        """Procesa la cola de entrenamiento"""
        while True:
            try:
                session_id = await asyncio.wait_for(
                    self.training_queue.get(),
                    timeout=1.0
                )
                
                await self.start_training_session(session_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error procesando cola de entrenamiento: {e}")
    
    async def _session_cleanup(self):
        """Limpia sesiones antiguas y completadas"""
        while True:
            try:
                current_time = datetime.now()
                sessions_to_remove = []
                
                for session_id, session in self.training_sessions.items():
                    # Remover sesiones completadas hace más de 1 hora
                    if (session.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED] and
                        session.completed_at and
                        (current_time - session.completed_at).total_seconds() > self.session_timeout):
                        sessions_to_remove.append(session_id)
                    
                    # Remover sesiones pendientes hace más de 1 hora
                    elif (session.status == TrainingStatus.PENDING and
                          (current_time - session.created_at).total_seconds() > self.session_timeout):
                        sessions_to_remove.append(session_id)
                
                # Remover sesiones marcadas
                for session_id in sessions_to_remove:
                    del self.training_sessions[session_id]
                    logger.debug(f"Sesión {session_id} removida por limpieza")
                
                await asyncio.sleep(300)  # Verificar cada 5 minutos
                
            except Exception as e:
                logger.error(f"Error en limpieza de sesiones: {e}")
                await asyncio.sleep(60)
    
    async def setup_rest_api_endpoint(self, endpoint: str, port: int = 8080):
        """Configura endpoint REST API para entrenamiento"""
        try:
            self.api_endpoints[endpoint] = {
                "port": port,
                "active": True,
                "created_at": datetime.now()
            }
            
            logger.info(f"Endpoint REST API configurado: {endpoint}:{port}")
            
        except Exception as e:
            logger.error(f"Error configurando endpoint REST API: {e}")
    
    async def setup_websocket_endpoint(self, endpoint: str, port: int = 8081):
        """Configura endpoint WebSocket para entrenamiento"""
        try:
            self.websocket_connections[endpoint] = {
                "port": port,
                "active": True,
                "created_at": datetime.now()
            }
            
            logger.info(f"Endpoint WebSocket configurado: {endpoint}:{port}")
            
        except Exception as e:
            logger.error(f"Error configurando endpoint WebSocket: {e}")
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de entrenamiento"""
        active_sessions = sum(1 for s in self.training_sessions.values() 
                            if s.status == TrainingStatus.IN_PROGRESS)
        
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": active_sessions,
            "successful_sessions": self.successful_sessions,
            "failed_sessions": self.failed_sessions,
            "success_rate": (self.successful_sessions / max(self.total_sessions, 1)) * 100,
            "supported_protocols": [p.value for p in self.supported_protocols],
            "api_endpoints": len(self.api_endpoints),
            "websocket_connections": len(self.websocket_connections)
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de entrenamiento"""
        try:
            state = {
                "training_sessions": {
                    session_id: {
                        "id": session.id,
                        "trainer_ai_id": session.trainer_ai_id,
                        "protocol": session.protocol.value,
                        "status": session.status.value,
                        "created_at": session.created_at.isoformat(),
                        "started_at": session.started_at.isoformat() if session.started_at else None,
                        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                        "progress": session.progress,
                        "error_message": session.error_message,
                        "results": session.results
                    }
                    for session_id, session in self.training_sessions.items()
                },
                "stats": {
                    "total_sessions": self.total_sessions,
                    "successful_sessions": self.successful_sessions,
                    "failed_sessions": self.failed_sessions
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/training/training_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de entrenamiento guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de entrenamiento: {e}")

# Instancia global de la interfaz de entrenamiento
training_interface = AITrainingInterface()

async def initialize_module(core_engine):
    """Inicializa el módulo de entrenamiento"""
    global training_interface
    await training_interface.initialize_module(core_engine)
    logger.info("Módulo de entrenamiento inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de entrenamiento"""
    if isinstance(input_data, dict) and 'create_session' in input_data:
        # Crear sesión de entrenamiento
        session_data = input_data['create_session']
        
        try:
            session_id = await training_interface.create_training_session(
                trainer_ai_id=session_data.get('trainer_ai_id', 'unknown'),
                protocol=TrainingProtocol(session_data.get('protocol', 'rest_api')),
                training_data=session_data.get('training_data', {}),
                model_config=session_data.get('model_config', {})
            )
            
            return {'session_created': True, 'session_id': session_id}
            
        except Exception as e:
            return {'session_created': False, 'error': str(e)}
    
    elif isinstance(input_data, dict) and 'get_progress' in input_data:
        # Obtener progreso de sesión
        session_id = input_data['get_progress']
        progress = await training_interface.get_training_progress(session_id)
        return progress
    
    return input_data

def run_modulo4():
    """Función de compatibilidad con el sistema anterior"""
    print("🎓 Módulo 4: Interfaz de Entrenamiento por otras IAs")
    print("   - Protocolos de entrenamiento estándar")
    print("   - APIs REST y WebSocket")
    print("   - Gestión de sesiones de entrenamiento")
    print("   - Entrenamiento colaborativo")
    print("   ✅ Módulo inicializado correctamente")