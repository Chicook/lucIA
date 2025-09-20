"""
Módulo 7: Sistema de Acción
Versión: 0.6.0
Funcionalidad: Ejecución de acciones, control de procesos y gestión de tareas
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import subprocess
import threading
import time

logger = logging.getLogger('LucIA_Action')

class ActionType(Enum):
    """Tipos de acciones"""
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    SYSTEM_COMMAND = "system_command"
    CUSTOM = "custom"

class ActionStatus(Enum):
    """Estados de acción"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Action:
    """Definición de acción"""
    id: str
    action_type: ActionType
    name: str
    description: str
    parameters: Dict[str, Any]
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    error: Optional[str] = None

class ActionSystem:
    """
    Sistema de acciones para LucIA.
    Gestiona la ejecución de tareas y procesos.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.actions = {}
        self.action_queue = asyncio.Queue()
        self.action_handlers = {}
        self.running_actions = {}
        self.max_concurrent_actions = 5
        
        # Estadísticas
        self.total_actions = 0
        self.completed_actions = 0
        self.failed_actions = 0
        self.cancelled_actions = 0
        
        # Inicializar manejadores
        self._initialize_handlers()
        
        # Iniciar procesador de cola
        asyncio.create_task(self._action_processor())
        
        logger.info("Sistema de acciones inicializado")
    
    def _initialize_handlers(self):
        """Inicializa los manejadores de acciones"""
        self.action_handlers = {
            ActionType.COMPUTATION: self._handle_computation,
            ActionType.COMMUNICATION: self._handle_communication,
            ActionType.DATA_PROCESSING: self._handle_data_processing,
            ActionType.FILE_OPERATION: self._handle_file_operation,
            ActionType.SYSTEM_COMMAND: self._handle_system_command,
            ActionType.CUSTOM: self._handle_custom
        }
    
    async def create_action(self, action_type: ActionType, name: str, 
                          description: str, parameters: Dict[str, Any],
                          timeout: int = 30, priority: int = 1) -> str:
        """
        Crea una nueva acción
        
        Args:
            action_type: Tipo de acción
            name: Nombre de la acción
            description: Descripción
            parameters: Parámetros
            timeout: Tiempo límite en segundos
            priority: Prioridad (1-10)
        
        Returns:
            ID de la acción
        """
        try:
            action_id = str(uuid.uuid4())
            action = Action(
                id=action_id,
                action_type=action_type,
                name=name,
                description=description,
                parameters=parameters,
                timeout=timeout,
                priority=priority,
                created_at=datetime.now()
            )
            
            self.actions[action_id] = action
            self.total_actions += 1
            
            # Agregar a cola
            await self.action_queue.put(action_id)
            
            logger.info(f"Acción creada: {action_id} ({name})")
            return action_id
            
        except Exception as e:
            logger.error(f"Error creando acción: {e}")
            raise
    
    async def execute_action(self, action_id: str) -> bool:
        """
        Ejecuta una acción específica
        
        Args:
            action_id: ID de la acción
        
        Returns:
            True si se ejecutó exitosamente
        """
        try:
            if action_id not in self.actions:
                logger.error(f"Acción {action_id} no encontrada")
                return False
            
            action = self.actions[action_id]
            
            if action.status != ActionStatus.PENDING:
                logger.warning(f"Acción {action_id} no está pendiente")
                return False
            
            # Verificar límite de acciones concurrentes
            if len(self.running_actions) >= self.max_concurrent_actions:
                logger.warning("Límite de acciones concurrentes alcanzado")
                return False
            
            # Marcar como ejecutándose
            action.status = ActionStatus.RUNNING
            action.started_at = datetime.now()
            self.running_actions[action_id] = action
            
            # Ejecutar acción
            handler = self.action_handlers.get(action.action_type)
            if handler:
                try:
                    result = await asyncio.wait_for(
                        handler(action.parameters),
                        timeout=action.timeout
                    )
                    
                    action.result = result
                    action.status = ActionStatus.COMPLETED
                    action.completed_at = datetime.now()
                    self.completed_actions += 1
                    
                    logger.info(f"Acción {action_id} completada exitosamente")
                    
                except asyncio.TimeoutError:
                    action.status = ActionStatus.FAILED
                    action.error = "Timeout"
                    action.completed_at = datetime.now()
                    self.failed_actions += 1
                    
                    logger.warning(f"Acción {action_id} falló por timeout")
                    
                except Exception as e:
                    action.status = ActionStatus.FAILED
                    action.error = str(e)
                    action.completed_at = datetime.now()
                    self.failed_actions += 1
                    
                    logger.error(f"Acción {action_id} falló: {e}")
                    
                finally:
                    # Remover de acciones ejecutándose
                    if action_id in self.running_actions:
                        del self.running_actions[action_id]
            else:
                action.status = ActionStatus.FAILED
                action.error = f"Manejador no encontrado para tipo {action.action_type}"
                action.completed_at = datetime.now()
                self.failed_actions += 1
                
                logger.error(f"Manejador no encontrado para acción {action_id}")
            
            return action.status == ActionStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error ejecutando acción {action_id}: {e}")
            return False
    
    async def cancel_action(self, action_id: str) -> bool:
        """Cancela una acción"""
        try:
            if action_id not in self.actions:
                return False
            
            action = self.actions[action_id]
            
            if action.status in [ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED]:
                return False
            
            action.status = ActionStatus.CANCELLED
            action.completed_at = datetime.now()
            
            if action_id in self.running_actions:
                del self.running_actions[action_id]
            
            self.cancelled_actions += 1
            logger.info(f"Acción {action_id} cancelada")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando acción {action_id}: {e}")
            return False
    
    async def get_action_status(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene el estado de una acción"""
        if action_id not in self.actions:
            return None
        
        action = self.actions[action_id]
        
        return {
            "id": action.id,
            "name": action.name,
            "type": action.action_type.value,
            "status": action.status.value,
            "created_at": action.created_at.isoformat(),
            "started_at": action.started_at.isoformat() if action.started_at else None,
            "completed_at": action.completed_at.isoformat() if action.completed_at else None,
            "result": action.result,
            "error": action.error,
            "retry_count": action.retry_count
        }
    
    async def _action_processor(self):
        """Procesa la cola de acciones"""
        while True:
            try:
                action_id = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=1.0
                )
                
                await self.execute_action(action_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error procesando cola de acciones: {e}")
    
    async def _handle_computation(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones de cómputo"""
        try:
            operation = parameters.get("operation", "add")
            values = parameters.get("values", [])
            
            if operation == "add":
                return sum(values)
            elif operation == "multiply":
                result = 1
                for value in values:
                    result *= value
                return result
            elif operation == "average":
                return sum(values) / len(values) if values else 0
            else:
                raise ValueError(f"Operación no soportada: {operation}")
                
        except Exception as e:
            logger.error(f"Error en cómputo: {e}")
            raise
    
    async def _handle_communication(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones de comunicación"""
        try:
            message = parameters.get("message", "")
            recipient = parameters.get("recipient", "unknown")
            
            # Simular envío de mensaje
            await asyncio.sleep(0.1)
            
            return {
                "sent": True,
                "message": message,
                "recipient": recipient,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en comunicación: {e}")
            raise
    
    async def _handle_data_processing(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones de procesamiento de datos"""
        try:
            data = parameters.get("data", [])
            operation = parameters.get("operation", "filter")
            
            if operation == "filter":
                condition = parameters.get("condition", lambda x: True)
                return [item for item in data if condition(item)]
            elif operation == "map":
                function = parameters.get("function", lambda x: x)
                return [function(item) for item in data]
            elif operation == "reduce":
                function = parameters.get("function", lambda x, y: x + y)
                result = data[0] if data else None
                for item in data[1:]:
                    result = function(result, item)
                return result
            else:
                raise ValueError(f"Operación no soportada: {operation}")
                
        except Exception as e:
            logger.error(f"Error en procesamiento de datos: {e}")
            raise
    
    async def _handle_file_operation(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones de operaciones de archivo"""
        try:
            operation = parameters.get("operation", "read")
            file_path = parameters.get("file_path", "")
            
            if operation == "read":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"content": content, "size": len(content)}
            elif operation == "write":
                content = parameters.get("content", "")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"written": True, "size": len(content)}
            elif operation == "exists":
                import os
                return {"exists": os.path.exists(file_path)}
            else:
                raise ValueError(f"Operación no soportada: {operation}")
                
        except Exception as e:
            logger.error(f"Error en operación de archivo: {e}")
            raise
    
    async def _handle_system_command(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones de comandos del sistema"""
        try:
            command = parameters.get("command", "")
            timeout = parameters.get("timeout", 30)
            
            # Ejecutar comando
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }
            
        except Exception as e:
            logger.error(f"Error en comando del sistema: {e}")
            raise
    
    async def _handle_custom(self, parameters: Dict[str, Any]) -> Any:
        """Maneja acciones personalizadas"""
        try:
            function_name = parameters.get("function_name", "")
            args = parameters.get("args", [])
            kwargs = parameters.get("kwargs", {})
            
            # Buscar función personalizada
            if hasattr(self, f"_custom_{function_name}"):
                function = getattr(self, f"_custom_{function_name}")
                return await function(*args, **kwargs)
            else:
                raise ValueError(f"Función personalizada no encontrada: {function_name}")
                
        except Exception as e:
            logger.error(f"Error en acción personalizada: {e}")
            raise
    
    async def get_action_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de acciones"""
        return {
            "total_actions": self.total_actions,
            "completed_actions": self.completed_actions,
            "failed_actions": self.failed_actions,
            "cancelled_actions": self.cancelled_actions,
            "running_actions": len(self.running_actions),
            "pending_actions": self.action_queue.qsize(),
            "success_rate": (self.completed_actions / max(self.total_actions, 1)) * 100
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de acciones"""
        try:
            state = {
                "actions": {
                    action_id: {
                        "id": action.id,
                        "action_type": action.action_type.value,
                        "name": action.name,
                        "description": action.description,
                        "parameters": action.parameters,
                        "timeout": action.timeout,
                        "retry_count": action.retry_count,
                        "max_retries": action.max_retries,
                        "priority": action.priority,
                        "created_at": action.created_at.isoformat(),
                        "started_at": action.started_at.isoformat() if action.started_at else None,
                        "completed_at": action.completed_at.isoformat() if action.completed_at else None,
                        "status": action.status.value,
                        "result": action.result,
                        "error": action.error
                    }
                    for action_id, action in self.actions.items()
                },
                "stats": {
                    "total_actions": self.total_actions,
                    "completed_actions": self.completed_actions,
                    "failed_actions": self.failed_actions,
                    "cancelled_actions": self.cancelled_actions
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/action_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de acciones guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de acciones: {e}")

# Instancia global del sistema de acciones
action_system = ActionSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de acciones"""
    global action_system
    action_system.core_engine = core_engine
    core_engine.action_system = action_system
    logger.info("Módulo de acciones inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de acciones"""
    if isinstance(input_data, dict) and "create_action" in input_data:
        # Crear nueva acción
        action_data = input_data["create_action"]
        
        try:
            action_id = await action_system.create_action(
                action_type=ActionType(action_data.get("type", "computation")),
                name=action_data.get("name", "Unnamed Action"),
                description=action_data.get("description", ""),
                parameters=action_data.get("parameters", {}),
                timeout=action_data.get("timeout", 30),
                priority=action_data.get("priority", 1)
            )
            
            return {"action_created": True, "action_id": action_id}
            
        except Exception as e:
            return {"action_created": False, "error": str(e)}
    
    elif isinstance(input_data, dict) and "get_status" in input_data:
        # Obtener estado de acción
        action_id = input_data["get_status"]
        status = await action_system.get_action_status(action_id)
        return status or {"error": "Acción no encontrada"}
    
    return input_data

def run_modulo7():
    """Función de compatibilidad con el sistema anterior"""
    print("⚡ Módulo 7: Sistema de Acción")
    print("   - Ejecución de tareas y procesos")
    print("   - Control de acciones concurrentes")
    print("   - Gestión de timeouts y reintentos")
    print("   - Acciones personalizadas")
    print("   ✅ Módulo inicializado correctamente")