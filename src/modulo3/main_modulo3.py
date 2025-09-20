"""
Módulo 3: Sistema de Comunicación entre IAs
Versión: 0.6.0
Funcionalidad: Comunicación, protocolos de intercambio y colaboración entre IAs
"""

import asyncio
import json
import socket
import threading
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import base64

logger = logging.getLogger('LucIA_Communication')

class MessageType(Enum):
    """Tipos de mensajes entre IAs"""
    GREETING = "greeting"
    DATA_EXCHANGE = "data_exchange"
    TRAINING_REQUEST = "training_request"
    TRAINING_RESPONSE = "training_response"
    COLLABORATION = "collaboration"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class AIMessage:
    """Estructura de mensaje entre IAs"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False
    correlation_id: Optional[str] = None
    signature: Optional[str] = None

class AICommunicationHub:
    """
    Hub de comunicación para intercambio entre IAs.
    Gestiona protocolos, seguridad y colaboración.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.ai_id = str(uuid.uuid4())
        self.connected_ais = {}
        self.message_queue = asyncio.Queue()
        self.response_handlers = {}
        self.is_running = False
        self.port = 8888
        self.host = "localhost"
        self.secret_key = self._generate_secret_key()
        
        # Configuración de comunicación
        self.max_message_size = 1024 * 1024  # 1MB
        self.heartbeat_interval = 30  # segundos
        self.message_timeout = 300  # 5 minutos
        
        # Estadísticas
        self.messages_sent = 0
        self.messages_received = 0
        self.connection_attempts = 0
        self.failed_connections = 0
        
        logger.info(f"Hub de comunicación inicializado con ID: {self.ai_id[:8]}...")
    
    def _generate_secret_key(self) -> str:
        """Genera una clave secreta para firmar mensajes"""
        return hashlib.sha256(f"lucia_ai_{self.ai_id}_{int(time.time())}".encode()).hexdigest()
    
    def _sign_message(self, message: AIMessage) -> str:
        """Firma un mensaje para verificación de integridad"""
        message_str = f"{message.id}{message.sender_id}{message.receiver_id}{message.message_type.value}{json.dumps(message.content)}"
        signature = hmac.new(
            self.secret_key.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _verify_message(self, message: AIMessage) -> bool:
        """Verifica la integridad de un mensaje"""
        if not message.signature:
            return False
        
        expected_signature = self._sign_message(message)
        return hmac.compare_digest(message.signature, expected_signature)
    
    async def start_communication_server(self):
        """Inicia el servidor de comunicación"""
        try:
            self.is_running = True
            
            # Iniciar servidor TCP
            server = await asyncio.start_server(
                self._handle_connection,
                self.host,
                self.port
            )
            
            logger.info(f"Servidor de comunicación iniciado en {self.host}:{self.port}")
            
            # Iniciar tareas de background
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._heartbeat_sender())
            asyncio.create_task(self._connection_monitor())
            
            # Mantener servidor activo
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            logger.error(f"Error iniciando servidor de comunicación: {e}")
            self.is_running = False
    
    async def _handle_connection(self, reader, writer):
        """Maneja nuevas conexiones de IAs"""
        try:
            client_address = writer.get_extra_info('peername')
            logger.info(f"Nueva conexión desde {client_address}")
            
            while self.is_running:
                # Leer tamaño del mensaje
                size_data = await reader.read(4)
                if not size_data:
                    break
                
                message_size = int.from_bytes(size_data, 'big')
                
                if message_size > self.max_message_size:
                    logger.warning(f"Mensaje demasiado grande: {message_size} bytes")
                    break
                
                # Leer mensaje completo
                message_data = await reader.read(message_size)
                if not message_data:
                    break
                
                # Deserializar mensaje
                message_dict = json.loads(message_data.decode('utf-8'))
                message = self._deserialize_message(message_dict)
                
                if message and self._verify_message(message):
                    await self.message_queue.put(message)
                    self.messages_received += 1
                    logger.debug(f"Mensaje recibido de {message.sender_id}")
                else:
                    logger.warning("Mensaje no válido o no verificado")
            
        except Exception as e:
            logger.error(f"Error manejando conexión: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _message_processor(self):
        """Procesa mensajes de la cola"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error procesando mensaje: {e}")
    
    async def _process_message(self, message: AIMessage):
        """Procesa un mensaje específico"""
        try:
            logger.debug(f"Procesando mensaje {message.id} de tipo {message.message_type.value}")
            
            # Actualizar información de AI conectada
            if message.sender_id not in self.connected_ais:
                self.connected_ais[message.sender_id] = {
                    'last_seen': datetime.now(),
                    'status': 'active',
                    'capabilities': []
                }
            else:
                self.connected_ais[message.sender_id]['last_seen'] = datetime.now()
            
            # Procesar según tipo de mensaje
            if message.message_type == MessageType.GREETING:
                await self._handle_greeting(message)
            elif message.message_type == MessageType.DATA_EXCHANGE:
                await self._handle_data_exchange(message)
            elif message.message_type == MessageType.TRAINING_REQUEST:
                await self._handle_training_request(message)
            elif message.message_type == MessageType.TRAINING_RESPONSE:
                await self._handle_training_response(message)
            elif message.message_type == MessageType.COLLABORATION:
                await self._handle_collaboration(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            else:
                logger.warning(f"Tipo de mensaje no reconocido: {message.message_type}")
            
            # Enviar respuesta si es requerida
            if message.requires_response:
                await self._send_response(message)
                
        except Exception as e:
            logger.error(f"Error procesando mensaje {message.id}: {e}")
    
    async def _handle_greeting(self, message: AIMessage):
        """Maneja mensajes de saludo"""
        logger.info(f"Saludo recibido de AI {message.sender_id}")
        
        # Enviar saludo de respuesta
        response = AIMessage(
            id=str(uuid.uuid4()),
            sender_id=self.ai_id,
            receiver_id=message.sender_id,
            message_type=MessageType.GREETING,
            content={
                "greeting": "Hola! Soy LucIA, lista para colaborar",
                "capabilities": ["learning", "memory", "reasoning", "training"],
                "version": "0.6.0"
            },
            timestamp=datetime.now(),
            requires_response=False
        )
        
        await self.send_message(response)
    
    async def _handle_data_exchange(self, message: AIMessage):
        """Maneja intercambio de datos"""
        logger.info(f"Intercambio de datos con AI {message.sender_id}")
        
        # Procesar datos recibidos
        data = message.content.get('data', {})
        data_type = message.content.get('type', 'unknown')
        
        # Almacenar en sistema de memoria si está disponible
        if self.core_engine and hasattr(self.core_engine, 'memory_system'):
            await self.core_engine.memory_system.store_memory(
                content=data,
                memory_type=f"ai_exchange_{data_type}",
                importance=0.7,
                metadata={
                    'source_ai': message.sender_id,
                    'exchange_timestamp': message.timestamp.isoformat()
                }
            )
    
    async def _handle_training_request(self, message: AIMessage):
        """Maneja solicitudes de entrenamiento"""
        logger.info(f"Solicitud de entrenamiento de AI {message.sender_id}")
        
        training_data = message.content.get('training_data', {})
        model_type = message.content.get('model_type', 'default')
        
        # Procesar entrenamiento si el sistema de aprendizaje está disponible
        if self.core_engine and hasattr(self.core_engine, 'learning_engine'):
            try:
                # Preparar datos de entrenamiento
                features = training_data.get('features', [])
                targets = training_data.get('targets', [])
                
                if len(features) > 0 and len(targets) > 0:
                    # Crear modelo temporal
                    model_name = f"collaborative_{message.sender_id}_{int(time.time())}"
                    
                    # Entrenar modelo
                    metrics = await self.core_engine.learning_engine.train_model(
                        model_name, 
                        features, 
                        targets
                    )
                    
                    # Enviar respuesta con resultados
                    response = AIMessage(
                        id=str(uuid.uuid4()),
                        sender_id=self.ai_id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.TRAINING_RESPONSE,
                        content={
                            "model_name": model_name,
                            "metrics": metrics,
                            "status": "success"
                        },
                        timestamp=datetime.now(),
                        requires_response=False,
                        correlation_id=message.id
                    )
                    
                    await self.send_message(response)
                else:
                    # Enviar error
                    await self._send_error_response(message, "Datos de entrenamiento inválidos")
            except Exception as e:
                await self._send_error_response(message, f"Error en entrenamiento: {str(e)}")
    
    async def _handle_training_response(self, message: AIMessage):
        """Maneja respuestas de entrenamiento"""
        logger.info(f"Respuesta de entrenamiento de AI {message.sender_id}")
        
        # Procesar respuesta según sea necesario
        status = message.content.get('status', 'unknown')
        if status == 'success':
            logger.info("Entrenamiento colaborativo completado exitosamente")
        else:
            logger.warning(f"Entrenamiento falló: {message.content.get('error', 'Error desconocido')}")
    
    async def _handle_collaboration(self, message: AIMessage):
        """Maneja solicitudes de colaboración"""
        logger.info(f"Solicitud de colaboración de AI {message.sender_id}")
        
        collaboration_type = message.content.get('type', 'general')
        task_data = message.content.get('task', {})
        
        # Procesar colaboración según el tipo
        if collaboration_type == 'joint_learning':
            await self._handle_joint_learning(message, task_data)
        elif collaboration_type == 'data_sharing':
            await self._handle_data_sharing(message, task_data)
        elif collaboration_type == 'problem_solving':
            await self._handle_problem_solving(message, task_data)
        else:
            logger.warning(f"Tipo de colaboración no soportado: {collaboration_type}")
    
    async def _handle_joint_learning(self, message: AIMessage, task_data: Dict):
        """Maneja aprendizaje conjunto"""
        logger.info("Iniciando aprendizaje conjunto")
        
        # Implementar lógica de aprendizaje conjunto
        # Por ahora, solo registrar la solicitud
        if self.core_engine and hasattr(self.core_engine, 'memory_system'):
            await self.core_engine.memory_system.store_memory(
                content=task_data,
                memory_type="joint_learning_task",
                importance=0.8,
                metadata={
                    'collaborator_ai': message.sender_id,
                    'task_timestamp': message.timestamp.isoformat()
                }
            )
    
    async def _handle_data_sharing(self, message: AIMessage, task_data: Dict):
        """Maneja compartir datos"""
        logger.info("Procesando solicitud de compartir datos")
        
        # Implementar lógica de compartir datos
        # Por ahora, solo registrar la solicitud
        pass
    
    async def _handle_problem_solving(self, message: AIMessage, task_data: Dict):
        """Maneja resolución conjunta de problemas"""
        logger.info("Iniciando resolución conjunta de problemas")
        
        # Implementar lógica de resolución de problemas
        # Por ahora, solo registrar la solicitud
        pass
    
    async def _handle_status_update(self, message: AIMessage):
        """Maneja actualizaciones de estado"""
        status = message.content.get('status', 'unknown')
        capabilities = message.content.get('capabilities', [])
        
        if message.sender_id in self.connected_ais:
            self.connected_ais[message.sender_id].update({
                'status': status,
                'capabilities': capabilities,
                'last_seen': datetime.now()
            })
        
        logger.debug(f"Estado actualizado para AI {message.sender_id}: {status}")
    
    async def _handle_heartbeat(self, message: AIMessage):
        """Maneja mensajes de heartbeat"""
        if message.sender_id in self.connected_ais:
            self.connected_ais[message.sender_id]['last_seen'] = datetime.now()
        
        logger.debug(f"Heartbeat recibido de AI {message.sender_id}")
    
    async def _send_response(self, original_message: AIMessage):
        """Envía respuesta a un mensaje"""
        # Implementar lógica de respuesta según el tipo de mensaje
        pass
    
    async def _send_error_response(self, original_message: AIMessage, error_msg: str):
        """Envía respuesta de error"""
        response = AIMessage(
            id=str(uuid.uuid4()),
            sender_id=self.ai_id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.ERROR,
            content={
                "error": error_msg,
                "original_message_id": original_message.id
            },
            timestamp=datetime.now(),
            requires_response=False,
            correlation_id=original_message.id
        )
        
        await self.send_message(response)
    
    async def send_message(self, message: AIMessage) -> bool:
        """Envía un mensaje a otra AI"""
        try:
            # Firmar mensaje
            message.signature = self._sign_message(message)
            
            # Serializar mensaje
            message_dict = self._serialize_message(message)
            message_data = json.dumps(message_dict).encode('utf-8')
            
            # Enviar a través de conexión existente o crear nueva
            # Por simplicidad, aquí solo registramos el envío
            logger.debug(f"Mensaje {message.id} enviado a {message.receiver_id}")
            self.messages_sent += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error enviando mensaje: {e}")
            return False
    
    def _serialize_message(self, message: AIMessage) -> Dict[str, Any]:
        """Serializa un mensaje para transmisión"""
        return {
            'id': message.id,
            'sender_id': message.sender_id,
            'receiver_id': message.receiver_id,
            'message_type': message.message_type.value,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'priority': message.priority,
            'requires_response': message.requires_response,
            'correlation_id': message.correlation_id,
            'signature': message.signature
        }
    
    def _deserialize_message(self, message_dict: Dict[str, Any]) -> Optional[AIMessage]:
        """Deserializa un mensaje desde diccionario"""
        try:
            return AIMessage(
                id=message_dict['id'],
                sender_id=message_dict['sender_id'],
                receiver_id=message_dict['receiver_id'],
                message_type=MessageType(message_dict['message_type']),
                content=message_dict['content'],
                timestamp=datetime.fromisoformat(message_dict['timestamp']),
                priority=message_dict.get('priority', 1),
                requires_response=message_dict.get('requires_response', False),
                correlation_id=message_dict.get('correlation_id'),
                signature=message_dict.get('signature')
            )
        except Exception as e:
            logger.error(f"Error deserializando mensaje: {e}")
            return None
    
    async def _heartbeat_sender(self):
        """Envía mensajes de heartbeat periódicos"""
        while self.is_running:
            try:
                for ai_id in self.connected_ais.keys():
                    heartbeat = AIMessage(
                        id=str(uuid.uuid4()),
                        sender_id=self.ai_id,
                        receiver_id=ai_id,
                        message_type=MessageType.HEARTBEAT,
                        content={"status": "alive"},
                        timestamp=datetime.now(),
                        requires_response=False
                    )
                    
                    await self.send_message(heartbeat)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error enviando heartbeat: {e}")
                await asyncio.sleep(60)
    
    async def _connection_monitor(self):
        """Monitorea conexiones y limpia IAs inactivas"""
        while self.is_running:
            try:
                current_time = datetime.now()
                inactive_ais = []
                
                for ai_id, info in self.connected_ais.items():
                    time_since_last_seen = current_time - info['last_seen']
                    if time_since_last_seen.total_seconds() > self.message_timeout:
                        inactive_ais.append(ai_id)
                
                # Remover IAs inactivas
                for ai_id in inactive_ais:
                    del self.connected_ais[ai_id]
                    logger.info(f"AI {ai_id} marcada como inactiva y removida")
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error monitoreando conexiones: {e}")
                await asyncio.sleep(60)
    
    async def get_communication_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de comunicación"""
        return {
            'ai_id': self.ai_id,
            'connected_ais': len(self.connected_ais),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'connection_attempts': self.connection_attempts,
            'failed_connections': self.failed_connections,
            'is_running': self.is_running,
            'active_connections': list(self.connected_ais.keys())
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de comunicación"""
        try:
            state = {
                'ai_id': self.ai_id,
                'connected_ais': self.connected_ais,
                'messages_sent': self.messages_sent,
                'messages_received': self.messages_received,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('data/communication_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info("Estado de comunicación guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de comunicación: {e}")

# Instancia global del hub de comunicación
communication_hub = AICommunicationHub()

async def initialize_module(core_engine):
    """Inicializa el módulo de comunicación"""
    global communication_hub
    communication_hub.core_engine = core_engine
    core_engine.communication_hub = communication_hub
    logger.info("Módulo de comunicación inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de comunicación"""
    if isinstance(input_data, dict) and 'send_message' in input_data:
        # Enviar mensaje a otra AI
        message_data = input_data['send_message']
        
        message = AIMessage(
            id=str(uuid.uuid4()),
            sender_id=communication_hub.ai_id,
            receiver_id=message_data.get('receiver_id', 'unknown'),
            message_type=MessageType(message_data.get('type', 'data_exchange')),
            content=message_data.get('content', {}),
            timestamp=datetime.now(),
            requires_response=message_data.get('requires_response', False)
        )
        
        success = await communication_hub.send_message(message)
        return {'message_sent': success, 'message_id': message.id}
    
    return input_data

def run_modulo3():
    """Función de compatibilidad con el sistema anterior"""
    print("🌐 Módulo 3: Sistema de Comunicación entre IAs")
    print("   - Protocolos de comunicación")
    print("   - Intercambio de datos")
    print("   - Colaboración entre IAs")
    print("   - Entrenamiento colaborativo")
    print("   ✅ Módulo inicializado correctamente")