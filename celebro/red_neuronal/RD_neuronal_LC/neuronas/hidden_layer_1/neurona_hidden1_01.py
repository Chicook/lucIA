"""
Neurona Hidden Layer 1 - 01 - Módulo Independiente
=================================================

Primera neurona de la primera capa oculta de la red neuronal modular.
Esta neurona recibe datos de las 8 neuronas de entrada y los procesa
de manera independiente, comunicándose con las neuronas de la segunda capa oculta.

Características:
- Tipo: HIDDEN
- Índice: 01
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
class NeuronaHidden101Config(NeuronConfig):
    """Configuración específica para la neurona hidden layer 1 - 01"""
    
    def __init__(self):
        super().__init__(
            neuron_id="hidden1_01",
            neuron_type=NeuronType.HIDDEN,
            layer_index=1,
            neuron_index=1,
            activation=ActivationType.RELU,
            bias=0.1,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0001,
            max_connections=20,
            processing_timeout=1.0
        )
        
        # Configuraciones específicas para capa oculta
        self.input_neurons = ["input_01", "input_02", "input_03", "input_04", 
                             "input_05", "input_06", "input_07", "input_08"]
        self.output_neurons = ["hidden2_01", "hidden2_02", "hidden2_03", "hidden2_04", 
                              "hidden2_05", "hidden2_06", "hidden2_07", "hidden2_08", 
                              "hidden2_09", "hidden2_10"]
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0

class NeuronaHidden101(HiddenNeuron):
    """
    Neurona Hidden Layer 1 - 01 - Procesa información de la primera capa oculta.
    
    Esta neurona es responsable de:
    - Recibir y acumular entradas de las 8 neuronas de entrada
    - Aplicar función de activación ReLU
    - Transmitir la salida a las 10 neuronas de la segunda capa oculta
    - Mantener métricas de procesamiento y calidad
    """
    
    def __init__(self):
        """Inicializa la neurona hidden layer 1 - 01"""
        config = NeuronaHidden101Config()
        super().__init__(config)
        
        # Métricas específicas de capa oculta
        self.hidden_metrics = {
            'total_processing_cycles': 0,
            'successful_activations': 0,
            'zero_activations': 0,
            'average_activation': 0.0,
            'max_activation': 0.0,
            'min_activation': float('inf'),
            'input_accumulation_time': 0.0,
            'processing_efficiency': 1.0
        }
        
        # Estado específico
        self.pending_inputs = {}
        self.activation_history = []
        self.is_accumulating = False
        self.last_activation_time = 0.0
        
        logger.info(f"Neurona hidden layer 1 - 01 inicializada - ID: {self.config.neuron_id}")
    
    async def receive_input_from_neuron(self, input_data: float, source_neuron: str) -> bool:
        """
        Recibe entrada de una neurona específica y la acumula.
        
        Args:
            input_data (float): Dato de entrada
            source_neuron (str): ID de la neurona que envió el dato
            
        Returns:
            bool: True si se procesó exitosamente
        """
        try:
            if source_neuron not in self.config.input_neurons:
                logger.warning(f"Entrada de neurona no esperada: {source_neuron}")
                return False
            
            # Almacenar entrada
            self.pending_inputs[source_neuron] = input_data
            self.config.accumulated_inputs += input_data
            self.config.input_count += 1
            
            # Verificar si tenemos todas las entradas esperadas
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
            
            # Aplicar función de activación ReLU
            activation_input = self.config.accumulated_inputs + self.config.bias
            activation_output = await self._apply_activation(activation_input)
            
            # Actualizar métricas
            self._update_hidden_metrics(activation_output, activation_input)
            
            # Procesar con la neurona base
            output = await self.process_input(activation_output, source_neuron="internal")
            
            # Limpiar estado para el siguiente ciclo
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.hidden_metrics['input_accumulation_time'] = processing_time
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
        """Actualiza las métricas específicas de capa oculta"""
        self.hidden_metrics['total_processing_cycles'] += 1
        self.hidden_metrics['successful_activations'] += 1
        
        # Actualizar métricas de activación
        self.hidden_metrics['max_activation'] = max(self.hidden_metrics['max_activation'], activation_output)
        self.hidden_metrics['min_activation'] = min(self.hidden_metrics['min_activation'], activation_output)
        
        # Contar activaciones cero (ReLU)
        if activation_output == 0.0:
            self.hidden_metrics['zero_activations'] += 1
        
        # Actualizar promedio de activación
        total_cycles = self.hidden_metrics['total_processing_cycles']
        current_avg = self.hidden_metrics['average_activation']
        self.hidden_metrics['average_activation'] = ((current_avg * (total_cycles - 1)) + activation_output) / total_cycles
        
        # Guardar historial de activaciones (mantener solo las últimas 100)
        self.activation_history.append(activation_output)
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)
    
    def get_hidden_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona oculta"""
        return {
            'neuron_id': self.config.neuron_id,
            'hidden_metrics': self.hidden_metrics,
            'activation_history': self.activation_history[-10:],  # Últimas 10 activaciones
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_activation_time': self.last_activation_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_activation_analysis(self) -> Dict[str, Any]:
        """Analiza el patrón de activaciones de la neurona"""
        if not self.activation_history:
            return {'status': 'no_data'}
        
        activations = np.array(self.activation_history)
        
        return {
            'mean_activation': float(np.mean(activations)),
            'std_activation': float(np.std(activations)),
            'min_activation': float(np.min(activations)),
            'max_activation': float(np.max(activations)),
            'zero_percentage': float(np.sum(activations == 0.0) / len(activations) * 100),
            'active_percentage': float(np.sum(activations > 0.0) / len(activations) * 100),
            'data_points': len(activations)
        }

# Función principal para ejecutar la neurona como módulo independiente
async def main():
    """Función principal para ejecutar la neurona como proceso independiente"""
    print("=" * 60)
    print("NEURONA HIDDEN LAYER 1 - 01 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        # Crear instancia de la neurona
        neurona = NeuronaHidden101()
        
        # Inicializar
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Neuronas de entrada esperadas: {neurona.config.expected_inputs}")
        print(f"   Bias: {neurona.config.bias}")
        
        # Simular recepción de entradas de las neuronas de entrada
        print("\n📊 Simulando recepción de entradas...")
        
        # Simular múltiples ciclos de procesamiento
        for cycle in range(3):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            # Generar entradas simuladas de las 8 neuronas de entrada
            input_values = np.random.normal(0.5, 0.3, 8)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.05)  # Simular tiempo de procesamiento
            
            # Esperar un poco para ver el procesamiento
            await asyncio.sleep(0.1)
        
        # Mostrar estadísticas
        stats = neurona.get_hidden_statistics()
        print(f"\n📈 Estadísticas de la neurona oculta:")
        print(f"   Ciclos de procesamiento: {stats['hidden_metrics']['total_processing_cycles']}")
        print(f"   Activaciones exitosas: {stats['hidden_metrics']['successful_activations']}")
        print(f"   Activaciones cero: {stats['hidden_metrics']['zero_activations']}")
        print(f"   Promedio de activación: {stats['hidden_metrics']['average_activation']:.4f}")
        print(f"   Activación máxima: {stats['hidden_metrics']['max_activation']:.4f}")
        print(f"   Tiempo promedio de acumulación: {stats['hidden_metrics']['input_accumulation_time']:.4f}s")
        
        # Mostrar análisis de activaciones
        analysis = neurona.get_activation_analysis()
        if analysis['status'] != 'no_data':
            print(f"\n🔍 Análisis de activaciones:")
            print(f"   Media: {analysis['mean_activation']:.4f}")
            print(f"   Desviación estándar: {analysis['std_activation']:.4f}")
            print(f"   Porcentaje activo: {analysis['active_percentage']:.1f}%")
            print(f"   Porcentaje cero: {analysis['zero_percentage']:.1f}%")
        
        # Mostrar estado completo
        state = neurona.get_state()
        print(f"\n🔍 Estado completo de la neurona:")
        print(json.dumps(state, indent=2, default=str))
        
        # Cerrar neurona
        await neurona.shutdown()
        
        print("\n" + "=" * 60)
        print("✅ NEURONA HIDDEN LAYER 1 - 01 COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error en neurona hidden layer 1 - 01: {e}")
        logger.error(f"Error en neurona hidden layer 1 - 01: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar neurona
    asyncio.run(main())
