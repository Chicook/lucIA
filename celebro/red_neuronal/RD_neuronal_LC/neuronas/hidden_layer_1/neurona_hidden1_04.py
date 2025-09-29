"""
Neurona Hidden Layer 1 - 04 - Módulo Independiente
=================================================

Cuarta neurona de la primera capa oculta de la red neuronal modular.
Esta neurona recibe datos de las 8 neuronas de entrada y los procesa
de manera independiente, comunicándose con las neuronas de la segunda capa oculta.

Características:
- Tipo: HIDDEN
- Índice: 04
- Activación: ReLU
- Conexiones de entrada: 8 neuronas de input layer
- Conexiones de salida: 10 neuronas de hidden layer 2

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
class NeuronaHidden104Config(NeuronConfig):
    """Configuración específica para la neurona hidden layer 1 - 04"""
    
    def __init__(self):
        super().__init__(
            neuron_id="hidden1_04",
            neuron_type=NeuronType.HIDDEN,
            layer_index=1,
            neuron_index=4,
            activation=ActivationType.RELU,
            bias=0.1,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0001,
            max_connections=20,
            processing_timeout=1.0
        )
        
        self.input_neurons = ["input_01", "input_02", "input_03", "input_04", 
                             "input_05", "input_06", "input_07", "input_08"]
        self.output_neurons = ["hidden2_01", "hidden2_02", "hidden2_03", "hidden2_04", 
                              "hidden2_05", "hidden2_06", "hidden2_07", "hidden2_08", 
                              "hidden2_09", "hidden2_10"]
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0

class NeuronaHidden104(HiddenNeuron):
    """Neurona Hidden Layer 1 - 04 - Procesa información de la primera capa oculta."""
    
    def __init__(self):
        config = NeuronaHidden104Config()
        super().__init__(config)
        
        self.input_metrics = {
            'total_processing_cycles': 0, 'successful_activations': 0, 'zero_activations': 0,
            'average_activation': 0.0, 'max_activation': 0.0, 'min_activation': float('inf'),
            'input_accumulation_time': 0.0, 'processing_efficiency': 1.0
        }
        
        self.pending_inputs = {}
        self.activation_history = []
        self.is_accumulating = False
        self.last_activation_time = 0.0
        
        logger.info(f"Neurona hidden layer 1 - 04 inicializada - ID: {self.config.neuron_id}")
    
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
            activation_output = await self._apply_activation(activation_input)
            
            self._update_hidden_metrics(activation_output, activation_input)
            output = await self.process_input(activation_output, source_neuron="internal")
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.input_metrics['input_accumulation_time'] = processing_time
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
    
    def _update_hidden_metrics(self, activation_output: float, activation_input: float):
        """Actualiza las métricas específicas de entrada"""
        self.input_metrics['total_processing_cycles'] += 1
        self.input_metrics['successful_activations'] += 1
        self.input_metrics['min_activation'] = min(self.input_metrics['min_activation'], activation_output)
        self.input_metrics['max_activation'] = max(self.input_metrics['max_activation'], activation_output)
        
        if activation_output == 0.0:
            self.input_metrics['zero_activations'] += 1
        
        total = self.input_metrics['total_processing_cycles']
        current_avg = self.input_metrics['average_activation']
        self.input_metrics['average_activation'] = ((current_avg * (total - 1)) + activation_output) / total
        
        self.activation_history.append(activation_output)
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
    
    def get_input_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de entrada"""
        return {
            'neuron_id': self.config.neuron_id,
            'input_metrics': self.input_metrics,
            'activation_history': self.activation_history[-10:],
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_activation_time': self.last_activation_time,
            'timestamp': datetime.now().isoformat()
        }

# Función principal
async def main():
    print("=" * 60)
    print("NEURONA HIDDEN LAYER 1 - 04 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        neurona = NeuronaHidden104()
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Neuronas de entrada esperadas: {neurona.config.expected_inputs}")
        
        print("\n📊 Simulando recepción de entradas...")
        
        for cycle in range(3):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            input_values = np.random.normal(0.5, 0.3, 8)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.05)
            
            await asyncio.sleep(0.1)
        
        stats = neurona.get_input_statistics()
        print(f"\n📈 Ciclos de procesamiento: {stats['input_metrics']['total_processing_cycles']}")
        print(f"   Promedio de activación: {stats['input_metrics']['average_activation']:.4f}")
        
        await neurona.shutdown()
        print("\n✅ NEURONA HIDDEN LAYER 1 - 04 COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error en neurona hidden layer 1 - 04: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
