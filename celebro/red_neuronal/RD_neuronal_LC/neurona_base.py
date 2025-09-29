"""
Clase Base para Neuronas Independientes - Sistema Modular DNN
============================================================

Este módulo define la clase base para todas las neuronas independientes del sistema.
Cada neurona es un módulo Python completamente autónomo que puede procesar entradas,
aplicar activaciones y comunicarse con otras neuronas de manera distribuida.

Arquitectura Modular:
- Cada neurona es un proceso/módulo independiente
- Comunicación asíncrona entre neuronas
- Procesamiento distribuido y paralelo
- Escalabilidad horizontal

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import numpy as np
import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import uuid

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuronType(Enum):
    """Tipos de neuronas en la red"""
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"

class ActivationType(Enum):
    """Tipos de funciones de activación disponibles"""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"

@dataclass
class NeuronConfig:
    """Configuración de una neurona"""
    neuron_id: str
    neuron_type: NeuronType
    layer_index: int
    neuron_index: int
    activation: ActivationType = ActivationType.RELU
    bias: float = 0.0
    learning_rate: float = 0.001
    dropout_rate: float = 0.0
    batch_normalization: bool = False
    weight_decay: float = 0.0
    max_connections: int = 100
    processing_timeout: float = 1.0

@dataclass
class Connection:
    """Representa una conexión entre neuronas"""
    from_neuron: str
    to_neuron: str
    weight: float
    is_active: bool = True
    connection_id: str = None
    
    def __post_init__(self):
        if self.connection_id is None:
            self.connection_id = f"{self.from_neuron}_to_{self.to_neuron}"

@dataclass
class NeuronState:
    """Estado actual de una neurona"""
    neuron_id: str
    current_input: float = 0.0
    current_output: float = 0.0
    accumulated_input: float = 0.0
    activation_value: float = 0.0
    error_gradient: float = 0.0
    is_active: bool = True
    last_update: float = 0.0
    processing_count: int = 0

class MessageQueue:
    """Cola de mensajes para comunicación asíncrona entre neuronas"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.message_count = 0
        self.dropped_messages = 0
    
    def put(self, message: Dict[str, Any], timeout: float = 1.0) -> bool:
        """Agrega un mensaje a la cola"""
        try:
            self.queue.put(message, timeout=timeout)
            self.message_count += 1
            return True
        except queue.Full:
            self.dropped_messages += 1
            logger.warning(f"Cola llena, mensaje descartado. Total descartados: {self.dropped_messages}")
            return False
    
    def get(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Obtiene un mensaje de la cola"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Tamaño actual de la cola"""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """Verifica si la cola está vacía"""
        return self.queue.empty()

class BaseNeuron(ABC):
    """
    Clase base abstracta para todas las neuronas independientes.
    
    Cada neurona es un módulo completamente autónomo que puede:
    - Procesar entradas de manera independiente
    - Aplicar funciones de activación
    - Comunicarse con otras neuronas
    - Mantener su propio estado y configuración
    - Ejecutarse en paralelo con otras neuronas
    """
    
    def __init__(self, config: NeuronConfig):
        """
        Inicializa una neurona independiente.
        
        Args:
            config (NeuronConfig): Configuración de la neurona
        """
        self.config = config
        self.state = NeuronState(neuron_id=config.neuron_id)
        
        # Conexiones de entrada y salida
        self.input_connections: Dict[str, Connection] = {}
        self.output_connections: Dict[str, Connection] = {}
        
        # Cola de mensajes para comunicación
        self.message_queue = MessageQueue()
        
        # Estado de procesamiento
        self.is_processing = False
        self.is_initialized = False
        
        # Métricas de rendimiento
        self.metrics = {
            'total_processed': 0,
            'average_processing_time': 0.0,
            'error_count': 0,
            'last_error': None
        }
        
        # Inicializar pesos y bias
        self._initialize_weights()
        
        logger.info(f"Neurona {config.neuron_id} inicializada - Tipo: {config.neuron_type.value}")
    
    def _initialize_weights(self):
        """Inicializa los pesos de las conexiones"""
        # Usar inicialización He para ReLU, Xavier para otras activaciones
        if self.config.activation == ActivationType.RELU:
            self.weight_scale = np.sqrt(2.0)
        else:
            self.weight_scale = np.sqrt(1.0)
        
        # Bias inicial
        self.bias = self.config.bias
    
    async def process_input(self, input_data: Union[float, np.ndarray], 
                          source_neuron: str = None) -> float:
        """
        Procesa una entrada y devuelve la salida de la neurona.
        
        Args:
            input_data: Datos de entrada
            source_neuron: ID de la neurona que envió los datos
            
        Returns:
            float: Valor de salida de la neurona
        """
        try:
            start_time = time.time()
            self.is_processing = True
            
            # Actualizar estado
            self.state.current_input = float(input_data) if isinstance(input_data, (int, float)) else float(np.mean(input_data))
            self.state.accumulated_input += self.state.current_input
            
            # Aplicar función de activación
            output = await self._apply_activation(self.state.current_input + self.bias)
            
            # Actualizar estado
            self.state.current_output = output
            self.state.activation_value = output
            self.state.last_update = time.time()
            self.state.processing_count += 1
            
            # Actualizar métricas
            processing_time = time.time() - start_time
            self._update_metrics(processing_time)
            
            # Enviar salida a neuronas conectadas
            await self._propagate_output(output)
            
            self.is_processing = False
            return output
            
        except Exception as e:
            self.is_processing = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error procesando entrada en neurona {self.config.neuron_id}: {e}")
            return 0.0
    
    async def _apply_activation(self, input_value: float) -> float:
        """Aplica la función de activación correspondiente"""
        try:
            if self.config.activation == ActivationType.RELU:
                return max(0.0, input_value)
            elif self.config.activation == ActivationType.SIGMOID:
                return 1.0 / (1.0 + np.exp(-input_value))
            elif self.config.activation == ActivationType.TANH:
                return np.tanh(input_value)
            elif self.config.activation == ActivationType.SOFTMAX:
                # Softmax se aplica a nivel de capa, aquí devolvemos la entrada
                return input_value
            elif self.config.activation == ActivationType.LINEAR:
                return input_value
            elif self.config.activation == ActivationType.LEAKY_RELU:
                return max(0.01 * input_value, input_value)
            elif self.config.activation == ActivationType.ELU:
                return input_value if input_value >= 0 else 0.01 * (np.exp(input_value) - 1)
            elif self.config.activation == ActivationType.SWISH:
                return input_value * (1.0 / (1.0 + np.exp(-input_value)))
            elif self.config.activation == ActivationType.GELU:
                return 0.5 * input_value * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (input_value + 0.044715 * input_value**3)))
            else:
                return input_value
                
        except Exception as e:
            logger.error(f"Error aplicando activación {self.config.activation.value}: {e}")
            return 0.0
    
    async def _propagate_output(self, output_value: float):
        """Propaga la salida a todas las neuronas conectadas"""
        try:
            for connection_id, connection in self.output_connections.items():
                if connection.is_active:
                    # Crear mensaje para la neurona destino
                    message = {
                        'type': 'input',
                        'from_neuron': self.config.neuron_id,
                        'data': output_value * connection.weight,
                        'timestamp': time.time(),
                        'connection_id': connection_id
                    }
                    
                    # Enviar mensaje (esto se implementaría con el sistema de comunicación)
                    await self._send_message(connection.to_neuron, message)
                    
        except Exception as e:
            logger.error(f"Error propagando salida desde neurona {self.config.neuron_id}: {e}")
    
    async def _send_message(self, target_neuron: str, message: Dict[str, Any]):
        """Envía un mensaje a otra neurona"""
        # Esta función se implementaría con el sistema de comunicación global
        # Por ahora, solo registramos el mensaje
        logger.debug(f"Enviando mensaje de {self.config.neuron_id} a {target_neuron}: {message['type']}")
    
    def add_input_connection(self, connection: Connection):
        """Agrega una conexión de entrada"""
        self.input_connections[connection.connection_id] = connection
        logger.info(f"Conexión de entrada agregada: {connection.from_neuron} -> {self.config.neuron_id}")
    
    def add_output_connection(self, connection: Connection):
        """Agrega una conexión de salida"""
        self.output_connections[connection.connection_id] = connection
        logger.info(f"Conexión de salida agregada: {self.config.neuron_id} -> {connection.to_neuron}")
    
    def remove_connection(self, connection_id: str):
        """Remueve una conexión"""
        if connection_id in self.input_connections:
            del self.input_connections[connection_id]
        if connection_id in self.output_connections:
            del self.output_connections[connection_id]
        logger.info(f"Conexión {connection_id} removida")
    
    def get_state(self) -> Dict[str, Any]:
        """Obtiene el estado actual de la neurona"""
        return {
            'config': asdict(self.config),
            'state': asdict(self.state),
            'connections': {
                'input': len(self.input_connections),
                'output': len(self.output_connections)
            },
            'metrics': self.metrics,
            'is_processing': self.is_processing
        }
    
    def reset_state(self):
        """Resetea el estado de la neurona"""
        self.state = NeuronState(neuron_id=self.config.neuron_id)
        self.metrics['total_processed'] = 0
        logger.info(f"Estado de neurona {self.config.neuron_id} reseteado")
    
    def _update_metrics(self, processing_time: float):
        """Actualiza las métricas de rendimiento"""
        self.metrics['total_processed'] += 1
        
        # Actualizar tiempo promedio de procesamiento
        total = self.metrics['total_processed']
        current_avg = self.metrics['average_processing_time']
        self.metrics['average_processing_time'] = ((current_avg * (total - 1)) + processing_time) / total
    
    @abstractmethod
    async def initialize(self):
        """Inicializa la neurona (implementar en subclases)"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Cierra la neurona de manera segura (implementar en subclases)"""
        pass
    
    def __repr__(self):
        return f"Neuron(id={self.config.neuron_id}, type={self.config.neuron_type.value}, " \
               f"activation={self.config.activation.value}, connections={len(self.input_connections)}+{len(self.output_connections)})"


class NeuronFactory:
    """Factory para crear neuronas de diferentes tipos"""
    
    @staticmethod
    def create_neuron(config: NeuronConfig) -> BaseNeuron:
        """Crea una neurona del tipo especificado"""
        if config.neuron_type == NeuronType.INPUT:
            return InputNeuron(config)
        elif config.neuron_type == NeuronType.HIDDEN:
            return HiddenNeuron(config)
        elif config.neuron_type == NeuronType.OUTPUT:
            return OutputNeuron(config)
        else:
            raise ValueError(f"Tipo de neurona no soportado: {config.neuron_type}")


# Clases específicas para cada tipo de neurona

class InputNeuron(BaseNeuron):
    """Neurona de entrada - recibe datos externos"""
    
    async def initialize(self):
        """Inicializa la neurona de entrada"""
        self.is_initialized = True
        logger.info(f"Neurona de entrada {self.config.neuron_id} inicializada")
    
    async def shutdown(self):
        """Cierra la neurona de entrada"""
        self.is_initialized = False
        logger.info(f"Neurona de entrada {self.config.neuron_id} cerrada")


class HiddenNeuron(BaseNeuron):
    """Neurona oculta - procesa información intermedia"""
    
    async def initialize(self):
        """Inicializa la neurona oculta"""
        self.is_initialized = True
        logger.info(f"Neurona oculta {self.config.neuron_id} inicializada")
    
    async def shutdown(self):
        """Cierra la neurona oculta"""
        self.is_initialized = False
        logger.info(f"Neurona oculta {self.config.neuron_id} cerrada")


class OutputNeuron(BaseNeuron):
    """Neurona de salida - produce resultados finales"""
    
    async def initialize(self):
        """Inicializa la neurona de salida"""
        self.is_initialized = True
        logger.info(f"Neurona de salida {self.config.neuron_id} inicializada")
    
    async def shutdown(self):
        """Cierra la neurona de salida"""
        self.is_initialized = False
        logger.info(f"Neurona de salida {self.config.neuron_id} cerrada")


if __name__ == "__main__":
    """
    Script de prueba para la clase base de neuronas
    """
    print("=" * 60)
    print("PRUEBA DE CLASE BASE DE NEURONAS")
    print("=" * 60)
    
    # Crear configuración de prueba
    config = NeuronConfig(
        neuron_id="test_neuron_001",
        neuron_type=NeuronType.HIDDEN,
        layer_index=1,
        neuron_index=0,
        activation=ActivationType.RELU
    )
    
    # Crear neurona de prueba
    neuron = HiddenNeuron(config)
    
    # Probar procesamiento
    async def test_neuron():
        output = await neuron.process_input(1.5)
        print(f"Entrada: 1.5, Salida: {output:.4f}")
        
        # Probar diferentes funciones de activación
        activations = [ActivationType.RELU, ActivationType.SIGMOID, ActivationType.TANH]
        
        for activation in activations:
            neuron.config.activation = activation
            output = await neuron.process_input(1.0)
            print(f"Activación {activation.value}: {output:.4f}")
        
        # Mostrar estado
        state = neuron.get_state()
        print(f"\nEstado de la neurona: {json.dumps(state, indent=2, default=str)}")
    
    # Ejecutar prueba
    asyncio.run(test_neuron())
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 60)
