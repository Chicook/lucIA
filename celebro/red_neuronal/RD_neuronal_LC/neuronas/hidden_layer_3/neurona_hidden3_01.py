"""
Neurona Hidden Layer 3 - 01 - Módulo Independiente
=================================================

Primera neurona de la tercera capa oculta de la red neuronal modular.
Esta neurona recibe datos de las 10 neuronas de hidden layer 2 y los procesa
de manera independiente, comunicándose con las neuronas de salida.

Características:
- Tipo: HIDDEN
- Índice: 01
- Activación: ReLU
- Conexiones de entrada: 10 neuronas de hidden layer 2
- Conexiones de salida: 4 neuronas de salida

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
class NeuronaHidden301Config(NeuronConfig):
    """Configuración específica para la neurona hidden layer 3 - 01"""
    
    def __init__(self):
        super().__init__(
            neuron_id="hidden3_01",
            neuron_type=NeuronType.HIDDEN,
            layer_index=3,
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
        
        # Configuraciones específicas para tercera capa oculta
        self.input_neurons = [f"hidden2_{i:02d}" for i in range(1, 11)]
        self.output_neurons = [f"output_{i:02d}" for i in range(1, 5)]
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0
        
        # Parámetros específicos de la tercera capa
        self.activation_threshold = 0.05  # Umbral más bajo para tercera capa
        self.learning_rate_decay = 0.99
        self.dropout_probability = 0.1

class NeuronaHidden301(HiddenNeuron):
    """
    Neurona Hidden Layer 3 - 01 - Procesa información de la tercera capa oculta.
    
    Esta neurona es responsable de:
    - Recibir y acumular entradas de las 10 neuronas de hidden layer 2
    - Aplicar función de activación ReLU con umbral optimizado
    - Transmitir la salida a las 4 neuronas de salida
    - Mantener métricas avanzadas y patrones de activación
    """
    
    def __init__(self):
        """Inicializa la neurona hidden layer 3 - 01"""
        config = NeuronaHidden301Config()
        super().__init__(config)
        
        # Métricas específicas de tercera capa oculta
        self.hidden3_metrics = {
            'total_processing_cycles': 0,
            'successful_activations': 0,
            'zero_activations': 0,
            'threshold_activations': 0,
            'average_activation': 0.0,
            'max_activation': 0.0,
            'min_activation': float('inf'),
            'activation_variance': 0.0,
            'input_accumulation_time': 0.0,
            'processing_efficiency': 1.0,
            'dropout_applications': 0,
            'gradient_accumulation': 0.0
        }
        
        # Estado específico para tercera capa
        self.pending_inputs = {}
        self.activation_history = []
        self.gradient_history = []
        self.is_accumulating = False
        self.last_activation_time = 0.0
        self.activation_pattern = []
        self.feature_importance = {}
        
        logger.info(f"Neurona hidden layer 3 - 01 inicializada - ID: {self.config.neuron_id}")
    
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
            
            # Almacenar entrada con análisis de importancia
            self.pending_inputs[source_neuron] = input_data
            self.config.accumulated_inputs += input_data
            self.config.input_count += 1
            
            # Actualizar importancia de características
            self._update_feature_importance(source_neuron, input_data)
            
            # Verificar si tenemos todas las entradas esperadas
            if len(self.pending_inputs) >= self.config.expected_inputs:
                await self._process_accumulated_inputs()
            
            return True
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo entrada en neurona {self.config.neuron_id}: {e}")
            return False
    
    def _update_feature_importance(self, source_neuron: str, input_data: float):
        """Actualiza la importancia de las características de entrada"""
        try:
            if source_neuron not in self.feature_importance:
                self.feature_importance[source_neuron] = {
                    'total_contribution': 0.0,
                    'activation_count': 0,
                    'average_magnitude': 0.0
                }
            
            importance = self.feature_importance[source_neuron]
            importance['total_contribution'] += abs(input_data)
            importance['activation_count'] += 1
            importance['average_magnitude'] = importance['total_contribution'] / importance['activation_count']
            
        except Exception as e:
            logger.warning(f"Error actualizando importancia de características: {e}")
    
    async def _process_accumulated_inputs(self):
        """Procesa todas las entradas acumuladas con lógica avanzada para tercera capa"""
        try:
            self.is_accumulating = True
            start_time = time.time()
            
            # Aplicar función de activación ReLU con umbral optimizado
            activation_input = self.config.accumulated_inputs + self.config.bias
            
            # ReLU con umbral más bajo para tercera capa
            if activation_input > self.config.activation_threshold:
                activation_output = activation_input
            else:
                activation_output = 0.0
            
            # Aplicar dropout ocasional
            if np.random.random() < self.config.dropout_probability:
                activation_output = 0.0
                self.hidden3_metrics['dropout_applications'] += 1
            
            # Actualizar métricas avanzadas
            self._update_hidden3_metrics(activation_output, activation_input)
            
            # Procesar con la neurona base
            output = await self.process_input(activation_output, source_neuron="internal")
            
            # Actualizar patrón de activación
            self._update_activation_pattern(activation_output)
            
            # Limpiar estado para el siguiente ciclo
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.hidden3_metrics['input_accumulation_time'] = processing_time
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
    
    def _update_hidden3_metrics(self, activation_output: float, activation_input: float):
        """Actualiza las métricas específicas de tercera capa oculta"""
        self.hidden3_metrics['total_processing_cycles'] += 1
        self.hidden3_metrics['successful_activations'] += 1
        
        # Actualizar métricas de activación
        self.hidden3_metrics['max_activation'] = max(self.hidden3_metrics['max_activation'], activation_output)
        self.hidden3_metrics['min_activation'] = min(self.hidden3_metrics['min_activation'], activation_output)
        
        # Contar tipos de activaciones
        if activation_output == 0.0:
            self.hidden3_metrics['zero_activations'] += 1
        elif activation_output > self.config.activation_threshold:
            self.hidden3_metrics['threshold_activations'] += 1
        
        # Actualizar promedio y varianza
        total_cycles = self.hidden3_metrics['total_processing_cycles']
        current_avg = self.hidden3_metrics['average_activation']
        self.hidden3_metrics['average_activation'] = ((current_avg * (total_cycles - 1)) + activation_output) / total_cycles
        
        # Calcular varianza
        if total_cycles > 1:
            variance = ((activation_output - self.hidden3_metrics['average_activation']) ** 2) / total_cycles
            self.hidden3_metrics['activation_variance'] = (
                (self.hidden3_metrics['activation_variance'] * (total_cycles - 1)) + variance
            ) / total_cycles
        
        # Guardar historial de activaciones
        self.activation_history.append(activation_output)
        if len(self.activation_history) > 150:  # Más historial para tercera capa
            self.activation_history.pop(0)
    
    def _update_activation_pattern(self, activation_output: float):
        """Actualiza el patrón de activación de la neurona"""
        # Categorizar activación con más granularidad para tercera capa
        if activation_output == 0.0:
            pattern = "zero"
        elif activation_output < 0.2:
            pattern = "very_low"
        elif activation_output < 0.5:
            pattern = "low"
        elif activation_output < 1.0:
            pattern = "medium"
        elif activation_output < 2.0:
            pattern = "high"
        else:
            pattern = "very_high"
        
        self.activation_pattern.append(pattern)
        
        # Mantener solo los últimos 75 patrones
        if len(self.activation_pattern) > 75:
            self.activation_pattern.pop(0)
    
    def get_hidden3_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona de tercera capa oculta"""
        return {
            'neuron_id': self.config.neuron_id,
            'hidden3_metrics': self.hidden3_metrics,
            'activation_history': self.activation_history[-25:],  # Últimas 25 activaciones
            'activation_pattern': self.activation_pattern[-15:],  # Últimos 15 patrones
            'feature_importance': self.feature_importance,
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_activation_time': self.last_activation_time,
            'activation_threshold': self.config.activation_threshold,
            'dropout_probability': self.config.dropout_probability,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_activation_analysis(self) -> Dict[str, Any]:
        """Analiza el patrón de activaciones de la neurona de tercera capa"""
        if not self.activation_history:
            return {'status': 'no_data'}
        
        activations = np.array(self.activation_history)
        
        return {
            'mean_activation': float(np.mean(activations)),
            'std_activation': float(np.std(activations)),
            'min_activation': float(np.min(activations)),
            'max_activation': float(np.max(activations)),
            'zero_percentage': float(np.sum(activations == 0.0) / len(activations) * 100),
            'threshold_percentage': float(np.sum(activations > self.config.activation_threshold) / len(activations) * 100),
            'activation_stability': float(1.0 - np.std(activations) / max(0.001, np.mean(activations))),
            'data_points': len(activations),
            'pattern_diversity': len(set(self.activation_pattern[-30:])),
            'dropout_rate': float(self.hidden3_metrics['dropout_applications'] / max(1, self.hidden3_metrics['total_processing_cycles']) * 100)
        }
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Obtiene ranking de importancia de características"""
        ranking = []
        for neuron_id, importance in self.feature_importance.items():
            score = importance['average_magnitude'] * np.sqrt(importance['activation_count'])
            ranking.append((neuron_id, score))
        
        return sorted(ranking, key=lambda x: x[1], reverse=True)

# Función principal para ejecutar la neurona como módulo independiente
async def main():
    """Función principal para ejecutar la neurona como proceso independiente"""
    print("=" * 60)
    print("NEURONA HIDDEN LAYER 3 - 01 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        # Crear instancia de la neurona
        neurona = NeuronaHidden301()
        
        # Inicializar
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Umbral de activación: {neurona.config.activation_threshold}")
        print(f"   Probabilidad de dropout: {neurona.config.dropout_probability}")
        print(f"   Neuronas de entrada esperadas: {neurona.config.expected_inputs}")
        print(f"   Bias: {neurona.config.bias}")
        
        # Simular recepción de entradas de las neuronas de hidden layer 2
        print("\n📊 Simulando recepción de entradas de hidden layer 2...")
        
        # Simular múltiples ciclos de procesamiento
        for cycle in range(5):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            # Generar entradas simuladas de las 10 neuronas de hidden layer 2
            input_values = np.random.gamma(shape=2, scale=0.3, size=10)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.02)  # Simular tiempo de procesamiento
            
            # Esperar un poco para ver el procesamiento
            await asyncio.sleep(0.1)
        
        # Mostrar estadísticas avanzadas
        stats = neurona.get_hidden3_statistics()
        print(f"\n📈 Estadísticas de la neurona hidden layer 3:")
        print(f"   Ciclos de procesamiento: {stats['hidden3_metrics']['total_processing_cycles']}")
        print(f"   Activaciones exitosas: {stats['hidden3_metrics']['successful_activations']}")
        print(f"   Activaciones cero: {stats['hidden3_metrics']['zero_activations']}")
        print(f"   Activaciones sobre umbral: {stats['hidden3_metrics']['threshold_activations']}")
        print(f"   Aplicaciones de dropout: {stats['hidden3_metrics']['dropout_applications']}")
        print(f"   Promedio de activación: {stats['hidden3_metrics']['average_activation']:.4f}")
        print(f"   Varianza de activación: {stats['hidden3_metrics']['activation_variance']:.4f}")
        print(f"   Activación máxima: {stats['hidden3_metrics']['max_activation']:.4f}")
        print(f"   Tiempo promedio de acumulación: {stats['hidden3_metrics']['input_accumulation_time']:.4f}s")
        
        # Mostrar análisis de activaciones
        analysis = neurona.get_activation_analysis()
        if analysis['status'] != 'no_data':
            print(f"\n🔍 Análisis avanzado de activaciones:")
            print(f"   Media: {analysis['mean_activation']:.4f}")
            print(f"   Desviación estándar: {analysis['std_activation']:.4f}")
            print(f"   Porcentaje sobre umbral: {analysis['threshold_percentage']:.1f}%")
            print(f"   Estabilidad de activación: {analysis['activation_stability']:.4f}")
            print(f"   Diversidad de patrones: {analysis['pattern_diversity']}")
            print(f"   Tasa de dropout: {analysis['dropout_rate']:.1f}%")
        
        # Mostrar ranking de importancia de características
        ranking = neurona.get_feature_importance_ranking()
        if ranking:
            print(f"\n🏆 Ranking de importancia de características:")
            for i, (neuron_id, score) in enumerate(ranking[:5]):  # Top 5
                print(f"   {i+1}. {neuron_id}: {score:.4f}")
        
        # Mostrar estado completo
        state = neurona.get_state()
        print(f"\n🔍 Estado completo de la neurona:")
        print(json.dumps(state, indent=2, default=str))
        
        # Cerrar neurona
        await neurona.shutdown()
        
        print("\n" + "=" * 60)
        print("✅ NEURONA HIDDEN LAYER 3 - 01 COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error en neurona hidden layer 3 - 01: {e}")
        logger.error(f"Error en neurona hidden layer 3 - 01: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar neurona
    asyncio.run(main())
