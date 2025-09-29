"""
Neurona Hidden Layer 2 - 05 - Módulo Independiente
=================================================

Quinta neurona de la segunda capa oculta de la red neuronal modular.
Esta neurona recibe datos de las 10 neuronas de hidden layer 1 y los procesa
de manera independiente, comunicándose con las neuronas de hidden layer 3.

Características:
- Tipo: HIDDEN
- Índice: 05
- Activación: ReLU
- Conexiones de entrada: 10 neuronas de hidden layer 1
- Conexiones de salida: 10 neuronas de hidden layer 3

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import sys
import os
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime

# Agregar el directorio padre al path para importar neurona_base
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neurona_base import (
    BaseNeuron, HiddenNeuron, NeuronConfig, NeuronType, 
    ActivationType, Connection, MessageQueue
)

# Configuración específica de esta neurona
class NeuronaHidden205Config(NeuronConfig):
    """Configuración específica para la neurona hidden layer 2 - 05"""
    
    def __init__(self):
        super().__init__(
            neuron_id="hidden2_05",
            neuron_type=NeuronType.HIDDEN,
            layer_index=2,
            neuron_index=5,
            activation=ActivationType.RELU,
            bias=0.1,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0001,
            max_connections=20,
            processing_timeout=1.0
        )
        
        self.input_neurons = [f"hidden1_{i:02d}" for i in range(1, 11)]
        self.output_neurons = [f"hidden3_{i:02d}" for i in range(1, 11)]
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0
        self.activation_threshold = 0.1
        self.learning_rate_decay = 0.99

class NeuronaHidden205(HiddenNeuron):
    """Neurona Hidden Layer 2 - 05 - Procesa información de la segunda capa oculta."""
    
    def __init__(self):
        config = NeuronaHidden205Config()
        super().__init__(config)
        
        self.hidden2_metrics = {
            'total_processing_cycles': 0, 'successful_activations': 0, 'zero_activations': 0,
            'threshold_activations': 0, 'average_activation': 0.0, 'max_activation': 0.0,
            'min_activation': float('inf'), 'activation_variance': 0.0, 'input_accumulation_time': 0.0,
            'processing_efficiency': 1.0
        }
        
        self.pending_inputs = {}
        self.activation_history = []
        self.is_accumulating = False
        self.last_activation_time = 0.0
        
        logger.info(f"Neurona hidden layer 2 - 05 inicializada - ID: {self.config.neuron_id}")
    
    async def receive_input_from_neuron(self, input_data: float, source_neuron: str) -> bool:
        """Recibe entrada de una neurona específica y la acumula."""
        try:
            if source_neuron not in self.config.input_neurons:
                logger.warning(f"Entrada de neurona no esperada: {source_neuron}")
                return False
            
            self.pending_inputs[source_neuron] = input_data
            self.config.accumulated_inputs += input_data
            self.config.input_count += 1
            
            if len(self.pending_inputs) >= self.config.expected_inputs:
                await self._process_accumulated_inputs()
            
            return True
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo entrada en neurona {self.config.neuron_id}: {e}")
            return False
    
    async def _process_accumulated_inputs(self):
        """Procesa todas las entradas acumuladas"""
        try:
            self.is_accumulating = True
            start_time = time.time()
            
            activation_input = self.config.accumulated_inputs + self.config.bias
            
            if activation_input > self.config.activation_threshold:
                activation_output = activation_input
            else:
                activation_output = 0.0
            
            self._update_hidden2_metrics(activation_output, activation_input)
            output = await self.process_input(activation_output, source_neuron="internal")
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.hidden2_metrics['input_accumulation_time'] = processing_time
            self.last_activation_time = time.time()
            
            logger.debug(f"Neurona {self.config.neuron_id} activada: {activation_input:.4f} -> {output:.4f}")
            
        except Exception as e:
            self.is_accumulating = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error procesando entradas acumuladas en neurona {self.config.neuron_id}: {e}")
    
    def _reset_accumulation_state(self):
        """Resetea el estado de acumulación para el siguiente ciclo"""
        self.pending_inputs.clear()
        self.config.accumulated_inputs = 0.0
        self.config.input_count = 0
        self.is_accumulating = False
    
    def _update_hidden2_metrics(self, activation_output: float, activation_input: float):
        """Actualiza las métricas específicas de segunda capa oculta"""
        self.hidden2_metrics['total_processing_cycles'] += 1
        self.hidden2_metrics['successful_activations'] += 1
        self.hidden2_metrics['max_activation'] = max(self.hidden2_metrics['max_activation'], activation_output)
        self.hidden2_metrics['min_activation'] = min(self.hidden2_metrics['min_activation'], activation_output)
        
        if activation_output == 0.0:
            self.hidden2_metrics['zero_activations'] += 1
        elif activation_output > self.config.activation_threshold:
            self.hidden2_metrics['threshold_activations'] += 1
        
        total_cycles = self.hidden2_metrics['total_processing_cycles']
        current_avg = self.hidden2_metrics['average_activation']
        self.hidden2_metrics['average_activation'] = ((current_avg * (total_cycles - 1)) + activation_output) / total_cycles
        
        self.activation_history.append(activation_output)
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
    
    def get_hidden2_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona de segunda capa oculta"""
        return {
            'neuron_id': self.config.neuron_id,
            'hidden2_metrics': self.hidden2_metrics,
            'activation_history': self.activation_history[-10:],
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_activation_time': self.last_activation_time,
            'activation_threshold': self.config.activation_threshold,
            'timestamp': datetime.now().isoformat()
        }

# Función principal
async def main():
    print("=" * 60)
    print("NEURONA HIDDEN LAYER 2 - 05 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        neurona = NeuronaHidden205()
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Umbral de activación: {neurona.config.activation_threshold}")
        
        print("\n📊 Simulando recepción de entradas...")
        
        for cycle in range(3):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            input_values = np.random.exponential(scale=0.5, size=10)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.02)
            
            await asyncio.sleep(0.1)
        
        stats = neurona.get_hidden2_statistics()
        print(f"\n📈 Ciclos de procesamiento: {stats['hidden2_metrics']['total_processing_cycles']}")
        print(f"   Promedio de activación: {stats['hidden2_metrics']['average_activation']:.4f}")
        
        await neurona.shutdown()
        print("\n✅ NEURONA HIDDEN LAYER 2 - 05 COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error en neurona hidden layer 2 - 05: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
