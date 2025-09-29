"""
Coordinador de Red Neuronal Modular - Sistema de Gestión Central
===============================================================

Este módulo coordina todos los módulos de neuronas independientes de la red neuronal.
Gestiona la comunicación, sincronización y procesamiento distribuido entre las 42 neuronas
que componen la arquitectura completa de la red neuronal profunda.

Arquitectura Coordinada:
- 8 neuronas de entrada (input_01 a input_08)
- 30 neuronas ocultas (10 + 10 + 10 en 3 capas)
- 4 neuronas de salida (output_01 a output_04)

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import multiprocessing
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkTopology:
    """Topología de la red neuronal modular"""
    input_neurons: List[str] = None
    hidden_layers: List[List[str]] = None
    output_neurons: List[str] = None
    
    def __post_init__(self):
        if self.input_neurons is None:
            self.input_neurons = [f"input_{i:02d}" for i in range(1, 9)]
        
        if self.hidden_layers is None:
            self.hidden_layers = [
                [f"hidden1_{i:02d}" for i in range(1, 11)],
                [f"hidden2_{i:02d}" for i in range(1, 11)],
                [f"hidden3_{i:02d}" for i in range(1, 11)]
            ]
        
        if self.output_neurons is None:
            self.output_neurons = [f"output_{i:02d}" for i in range(1, 5)]

@dataclass
class ProcessingCycle:
    """Información de un ciclo de procesamiento"""
    cycle_id: str
    start_time: float
    end_time: float = 0.0
    input_data: Dict[str, float] = None
    output_data: Dict[str, float] = None
    processing_status: str = "pending"
    error_count: int = 0
    processing_time: float = 0.0

class MessageBus:
    """Bus de mensajes para comunicación entre neuronas"""
    
    def __init__(self):
        self.message_queues: Dict[str, queue.Queue] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
    def create_queue(self, neuron_id: str):
        """Crea una cola de mensajes para una neurona"""
        self.message_queues[neuron_id] = queue.Queue(maxsize=100)
        logger.info(f"Cola de mensajes creada para neurona {neuron_id}")
    
    def send_message(self, from_neuron: str, to_neuron: str, message: Dict[str, Any]) -> bool:
        """Envía un mensaje entre neuronas"""
        try:
            if to_neuron not in self.message_queues:
                logger.warning(f"No hay cola de mensajes para neurona {to_neuron}")
                return False
            
            message_with_metadata = {
                'from': from_neuron,
                'to': to_neuron,
                'timestamp': time.time(),
                'data': message
            }
            
            self.message_queues[to_neuron].put(message_with_metadata, timeout=1.0)
            
            # Guardar en historial
            self.message_history.append(message_with_metadata)
            if len(self.message_history) > self.max_history_size:
                self.message_history.pop(0)
            
            return True
            
        except queue.Full:
            logger.warning(f"Cola llena para neurona {to_neuron}")
            return False
        except Exception as e:
            logger.error(f"Error enviando mensaje de {from_neuron} a {to_neuron}: {e}")
            return False
    
    def get_message(self, neuron_id: str, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Obtiene un mensaje de la cola de una neurona"""
        try:
            if neuron_id not in self.message_queues:
                return None
            
            return self.message_queues[neuron_id].get(timeout=timeout)
            
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error obteniendo mensaje para neurona {neuron_id}: {e}")
            return None

class NetworkCoordinator:
    """
    Coordinador principal de la red neuronal modular.
    
    Este coordinador gestiona:
    - Inicialización y configuración de todas las neuronas
    - Comunicación entre neuronas mediante bus de mensajes
    - Sincronización de ciclos de procesamiento
    - Monitoreo y métricas de rendimiento
    - Manejo de errores y recuperación
    """
    
    def __init__(self):
        """Inicializa el coordinador de red"""
        self.topology = NetworkTopology()
        self.message_bus = MessageBus()
        
        # Estado de la red
        self.is_initialized = False
        self.is_processing = False
        self.current_cycle = None
        self.cycle_count = 0
        
        # Neuronas activas
        self.active_neurons: Dict[str, Any] = {}
        self.neuron_status: Dict[str, str] = {}
        
        # Métricas de rendimiento
        self.network_metrics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'average_cycle_time': 0.0,
            'total_processing_time': 0.0,
            'messages_sent': 0,
            'messages_failed': 0,
            'last_cycle_time': 0.0
        }
        
        # Configuración de procesamiento
        self.processing_config = {
            'max_concurrent_cycles': 1,
            'cycle_timeout': 10.0,
            'message_timeout': 2.0,
            'retry_attempts': 3,
            'enable_parallel_processing': False
        }
        
        logger.info("Coordinador de red neuronal inicializado")
    
    async def initialize_network(self) -> bool:
        """
        Inicializa toda la red neuronal creando las colas de mensajes
        y configurando la topología.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            logger.info("Inicializando red neuronal modular...")
            
            # Crear colas de mensajes para todas las neuronas
            all_neurons = (
                self.topology.input_neurons + 
                [neuron for layer in self.topology.hidden_layers for neuron in layer] +
                self.topology.output_neurons
            )
            
            for neuron_id in all_neurons:
                self.message_bus.create_queue(neuron_id)
                self.neuron_status[neuron_id] = "initialized"
            
            self.is_initialized = True
            logger.info(f"Red neuronal inicializada con {len(all_neurons)} neuronas")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando red neuronal: {e}")
            return False
    
    async def process_input_data(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """
        Procesa datos de entrada a través de toda la red neuronal.
        
        Args:
            input_data: Diccionario con datos de entrada {neuron_id: value}
            
        Returns:
            Dict[str, float]: Diccionario con datos de salida {neuron_id: value}
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Red neuronal no inicializada")
            
            # Crear nuevo ciclo de procesamiento
            cycle_id = f"cycle_{self.cycle_count:06d}"
            self.current_cycle = ProcessingCycle(
                cycle_id=cycle_id,
                start_time=time.time(),
                input_data=input_data.copy()
            )
            
            self.is_processing = True
            logger.info(f"Iniciando ciclo de procesamiento {cycle_id}")
            
            # Fase 1: Procesar capa de entrada
            input_results = await self._process_input_layer(input_data)
            
            # Fase 2: Procesar capas ocultas
            hidden_results = await self._process_hidden_layers(input_results)
            
            # Fase 3: Procesar capa de salida
            output_results = await self._process_output_layer(hidden_results)
            
            # Finalizar ciclo
            await self._finalize_processing_cycle(output_results)
            
            self.is_processing = False
            return output_results
            
        except Exception as e:
            self.is_processing = False
            self.current_cycle.processing_status = "failed"
            self.current_cycle.error_count += 1
            logger.error(f"Error procesando datos de entrada: {e}")
            raise
    
    async def _process_input_layer(self, input_data: Dict[str, float]) -> Dict[str, float]:
        """Procesa la capa de entrada"""
        results = {}
        
        for neuron_id in self.topology.input_neurons:
            if neuron_id in input_data:
                # Simular procesamiento de neurona de entrada
                processed_value = input_data[neuron_id]  # Las neuronas de entrada son lineales
                results[neuron_id] = processed_value
                
                # Enviar a todas las neuronas de la primera capa oculta
                for hidden_neuron in self.topology.hidden_layers[0]:
                    message = {
                        'type': 'input',
                        'value': processed_value,
                        'cycle_id': self.current_cycle.cycle_id
                    }
                    self.message_bus.send_message(neuron_id, hidden_neuron, message)
        
        return results
    
    async def _process_hidden_layers(self, input_results: Dict[str, float]) -> Dict[str, float]:
        """Procesa las capas ocultas"""
        current_results = input_results.copy()
        
        for layer_index, layer_neurons in enumerate(self.topology.hidden_layers):
            layer_results = {}
            
            for neuron_id in layer_neurons:
                # Simular procesamiento de neurona oculta (ReLU)
                accumulated_input = 0.0
                input_count = 0
                
                # Recibir entradas de la capa anterior
                for _ in range(len(self.topology.input_neurons if layer_index == 0 else self.topology.hidden_layers[layer_index-1])):
                    message = self.message_bus.get_message(neuron_id, timeout=0.1)
                    if message and message['data']['type'] == 'input':
                        accumulated_input += message['data']['value']
                        input_count += 1
                
                # Aplicar activación ReLU
                if input_count > 0:
                    activation_value = max(0.0, accumulated_input + 0.1)  # Bias de 0.1
                    layer_results[neuron_id] = activation_value
                    
                    # Enviar a la siguiente capa
                    if layer_index < len(self.topology.hidden_layers) - 1:
                        # Enviar a la siguiente capa oculta
                        for next_neuron in self.topology.hidden_layers[layer_index + 1]:
                            message = {
                                'type': 'hidden_input',
                                'value': activation_value,
                                'cycle_id': self.current_cycle.cycle_id
                            }
                            self.message_bus.send_message(neuron_id, next_neuron, message)
                    else:
                        # Enviar a la capa de salida
                        for output_neuron in self.topology.output_neurons:
                            message = {
                                'type': 'hidden_input',
                                'value': activation_value,
                                'cycle_id': self.current_cycle.cycle_id
                            }
                            self.message_bus.send_message(neuron_id, output_neuron, message)
            
            current_results.update(layer_results)
        
        return current_results
    
    async def _process_output_layer(self, hidden_results: Dict[str, float]) -> Dict[str, float]:
        """Procesa la capa de salida"""
        output_results = {}
        
        for neuron_id in self.topology.output_neurons:
            # Simular procesamiento de neurona de salida
            accumulated_input = 0.0
            input_count = 0
            
            # Recibir entradas de la última capa oculta
            for _ in range(len(self.topology.hidden_layers[-1])):
                message = self.message_bus.get_message(neuron_id, timeout=0.1)
                if message and message['data']['type'] == 'hidden_input':
                    accumulated_input += message['data']['value']
                    input_count += 1
            
            # Aplicar activación softmax (simplificada)
            if input_count > 0:
                # Para simplificar, aplicamos sigmoid en lugar de softmax completo
                output_value = 1.0 / (1.0 + np.exp(-accumulated_input))
                output_results[neuron_id] = output_value
        
        return output_results
    
    async def _finalize_processing_cycle(self, output_results: Dict[str, float]):
        """Finaliza un ciclo de procesamiento y actualiza métricas"""
        self.current_cycle.end_time = time.time()
        self.current_cycle.output_data = output_results
        self.current_cycle.processing_time = self.current_cycle.end_time - self.current_cycle.start_time
        self.current_cycle.processing_status = "completed"
        
        # Actualizar métricas
        self.network_metrics['total_cycles'] += 1
        self.network_metrics['successful_cycles'] += 1
        self.network_metrics['total_processing_time'] += self.current_cycle.processing_time
        self.network_metrics['last_cycle_time'] = self.current_cycle.processing_time
        
        # Actualizar tiempo promedio
        total_cycles = self.network_metrics['total_cycles']
        current_avg = self.network_metrics['average_cycle_time']
        self.network_metrics['average_cycle_time'] = (
            (current_avg * (total_cycles - 1)) + self.current_cycle.processing_time
        ) / total_cycles
        
        self.cycle_count += 1
        
        logger.info(f"Ciclo {self.current_cycle.cycle_id} completado en {self.current_cycle.processing_time:.4f}s")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual de la red"""
        return {
            'is_initialized': self.is_initialized,
            'is_processing': self.is_processing,
            'current_cycle': asdict(self.current_cycle) if self.current_cycle else None,
            'network_metrics': self.network_metrics,
            'neuron_status': self.neuron_status,
            'topology': asdict(self.topology),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas detalladas de rendimiento"""
        if self.network_metrics['total_cycles'] == 0:
            return {'status': 'no_data'}
        
        success_rate = (self.network_metrics['successful_cycles'] / 
                       self.network_metrics['total_cycles'] * 100)
        
        return {
            'total_cycles': self.network_metrics['total_cycles'],
            'success_rate': success_rate,
            'average_cycle_time': self.network_metrics['average_cycle_time'],
            'last_cycle_time': self.network_metrics['last_cycle_time'],
            'total_processing_time': self.network_metrics['total_processing_time'],
            'messages_sent': self.network_metrics['messages_sent'],
            'messages_failed': self.network_metrics['messages_failed'],
            'throughput': self.network_metrics['total_cycles'] / max(1, self.network_metrics['total_processing_time']),
            'efficiency': success_rate / 100.0
        }
    
    async def shutdown(self):
        """Cierra la red neuronal de manera segura"""
        logger.info("Cerrando red neuronal...")
        self.is_processing = False
        self.is_initialized = False
        logger.info("Red neuronal cerrada")

# Función principal para demostrar el coordinador
async def main():
    """Función principal para demostrar el coordinador de red"""
    print("=" * 80)
    print("COORDINADOR DE RED NEURONAL MODULAR - DEMOSTRACIÓN")
    print("=" * 80)
    
    try:
        # Crear coordinador
        coordinator = NetworkCoordinator()
        
        # Inicializar red
        success = await coordinator.initialize_network()
        if not success:
            print("❌ Error inicializando red neuronal")
            return
        
        print("✅ Red neuronal inicializada exitosamente")
        print(f"   Neuronas de entrada: {len(coordinator.topology.input_neurons)}")
        print(f"   Capas ocultas: {len(coordinator.topology.hidden_layers)}")
        print(f"   Neuronas de salida: {len(coordinator.topology.output_neurons)}")
        
        # Procesar datos de prueba
        print("\n📊 Procesando datos de prueba...")
        
        test_inputs = [
            [0.1, 0.5, -0.2, 0.8, 1.2, -0.3, 0.7, 0.4],
            [0.3, -0.1, 0.9, -0.5, 0.6, 1.1, -0.8, 0.2],
            [-0.4, 0.7, 0.1, -0.9, 1.3, 0.5, -0.2, 0.8]
        ]
        
        for i, input_values in enumerate(test_inputs):
            print(f"\n--- Prueba {i + 1} ---")
            
            # Crear diccionario de entrada
            input_data = {f"input_{j:02d}": value for j, value in enumerate(input_values, 1)}
            
            # Procesar
            start_time = time.time()
            output_data = await coordinator.process_input_data(input_data)
            processing_time = time.time() - start_time
            
            print(f"Entrada: {input_values}")
            print(f"Salida: {list(output_data.values())}")
            print(f"Tiempo de procesamiento: {processing_time:.4f}s")
            
            await asyncio.sleep(0.5)  # Pausa entre pruebas
        
        # Mostrar métricas finales
        metrics = coordinator.get_performance_metrics()
        print(f"\n📈 Métricas de rendimiento:")
        print(f"   Total de ciclos: {metrics['total_cycles']}")
        print(f"   Tasa de éxito: {metrics['success_rate']:.1f}%")
        print(f"   Tiempo promedio por ciclo: {metrics['average_cycle_time']:.4f}s")
        print(f"   Throughput: {metrics['throughput']:.2f} ciclos/segundo")
        print(f"   Eficiencia: {metrics['efficiency']:.2f}")
        
        # Mostrar estado de la red
        status = coordinator.get_network_status()
        print(f"\n🔍 Estado de la red:")
        print(f"   Inicializada: {status['is_initialized']}")
        print(f"   Procesando: {status['is_processing']}")
        print(f"   Neuronas activas: {len(status['neuron_status'])}")
        
        # Cerrar coordinador
        await coordinator.shutdown()
        
        print("\n" + "=" * 80)
        print("✅ DEMOSTRACIÓN DEL COORDINADOR COMPLETADA EXITOSAMENTE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error en demostración del coordinador: {e}")
        logger.error(f"Error en demostración del coordinador: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar demostración
    asyncio.run(main())
