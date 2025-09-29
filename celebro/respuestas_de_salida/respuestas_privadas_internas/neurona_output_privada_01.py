"""
Neurona Output Privada 01 - Módulo Independiente
===============================================

Primera neurona de salida para respuestas privadas internas de la red neuronal modular.
Esta neurona recibe datos de las 10 neuronas de hidden layer 3 y los procesa
para generar respuestas privadas internas del sistema.

Características:
- Tipo: OUTPUT
- Índice: 01
- Activación: Softmax
- Conexiones de entrada: 10 neuronas de hidden layer 3
- Propósito: Respuestas privadas internas del sistema
- Acceso: Solo componentes internos del sistema

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
    BaseNeuron, OutputNeuron, NeuronConfig, NeuronType, 
    ActivationType, Connection, MessageQueue
)

# Configuración específica de esta neurona
class NeuronaOutputPrivada01Config(NeuronConfig):
    """Configuración específica para la neurona output privada 01"""
    
    def __init__(self):
        super().__init__(
            neuron_id="output_privada_01",
            neuron_type=NeuronType.OUTPUT,
            layer_index=4,
            neuron_index=1,
            activation=ActivationType.SOFTMAX,
            bias=0.0,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0001,
            max_connections=20,
            processing_timeout=1.0
        )
        
        # Configuraciones específicas para neurona de salida privada
        self.input_neurons = [f"hidden3_{i:02d}" for i in range(1, 11)]
        self.output_type = "respuestas_privadas_internas"
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0
        
        # Parámetros específicos para salida privada
        self.privacy_level = "HIGH"
        self.encryption_enabled = True
        self.access_control = ["internal_system", "admin", "core_modules"]
        self.logging_level = "SECURE"
        self.confidence_threshold = 0.7

class NeuronaOutputPrivada01(OutputNeuron):
    """
    Neurona Output Privada 01 - Genera respuestas privadas internas del sistema.
    
    Esta neurona es responsable de:
    - Recibir y procesar entradas de las 10 neuronas de hidden layer 3
    - Aplicar función de activación Softmax para clasificación
    - Generar respuestas privadas internas del sistema
    - Mantener control de acceso y privacidad
    - Registrar actividad de forma segura
    """
    
    def __init__(self):
        """Inicializa la neurona output privada 01"""
        config = NeuronaOutputPrivada01Config()
        super().__init__(config)
        
        # Métricas específicas de salida privada
        self.output_private_metrics = {
            'total_processing_cycles': 0,
            'successful_classifications': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': float('inf'),
            'classification_time': 0.0,
            'privacy_violations': 0,
            'access_attempts': 0,
            'successful_accesses': 0
        }
        
        # Estado específico para salida privada
        self.pending_inputs = {}
        self.classification_history = []
        self.confidence_history = []
        self.is_accumulating = False
        self.last_classification_time = 0.0
        self.current_classification = None
        self.current_confidence = 0.0
        self.access_log = []
        
        logger.info(f"Neurona output privada 01 inicializada - ID: {self.config.neuron_id}")
    
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
            
            # Verificar acceso autorizado
            if not self._check_access_authorization(source_neuron):
                self.output_private_metrics['privacy_violations'] += 1
                logger.warning(f"Intento de acceso no autorizado desde {source_neuron}")
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
    
    def _check_access_authorization(self, source_neuron: str) -> bool:
        """Verifica si la neurona fuente tiene autorización para acceder"""
        try:
            # Simular verificación de autorización
            authorized_sources = self.config.input_neurons
            return source_neuron in authorized_sources
            
        except Exception as e:
            logger.warning(f"Error verificando autorización: {e}")
            return False
    
    async def _process_accumulated_inputs(self):
        """Procesa todas las entradas acumuladas con lógica de clasificación privada"""
        try:
            self.is_accumulating = True
            start_time = time.time()
            
            # Aplicar función de activación Softmax
            activation_input = self.config.accumulated_inputs + self.config.bias
            softmax_output = self._apply_softmax(activation_input)
            
            # Calcular confianza de la clasificación
            confidence = self._calculate_confidence(softmax_output)
            
            # Determinar clasificación basada en la salida Softmax
            classification = self._determine_classification(softmax_output, confidence)
            
            # Actualizar métricas
            self._update_output_private_metrics(softmax_output, confidence, classification)
            
            # Procesar con la neurona base
            output = await self.process_input(softmax_output, source_neuron="internal")
            
            # Actualizar estado de clasificación
            self.current_classification = classification
            self.current_confidence = confidence
            
            # Limpiar estado para el siguiente ciclo
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.output_private_metrics['classification_time'] = processing_time
            self.last_classification_time = time.time()
            
            logger.debug(f"Neurona {self.config.neuron_id} clasificada: {classification} (confianza: {confidence:.4f})")
            
        except Exception as e:
            self.is_accumulating = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error procesando entradas acumuladas en neurona {self.config.neuron_id}: {e}")
    
    def _apply_softmax(self, activation_input: float) -> float:
        """Aplica función de activación Softmax"""
        try:
            # Softmax simplificado para una sola neurona
            exp_input = np.exp(activation_input)
            return exp_input / (1 + exp_input)  # Normalización simplificada
            
        except Exception as e:
            logger.warning(f"Error aplicando Softmax: {e}")
            return 0.0
    
    def _calculate_confidence(self, softmax_output: float) -> float:
        """Calcula la confianza de la clasificación"""
        try:
            # Confianza basada en la magnitud de la salida Softmax
            confidence = min(1.0, max(0.0, abs(softmax_output) * 2))
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculando confianza: {e}")
            return 0.0
    
    def _determine_classification(self, softmax_output: float, confidence: float) -> str:
        """Determina la clasificación basada en la salida y confianza"""
        try:
            if confidence < self.config.confidence_threshold:
                return "LOW_CONFIDENCE"
            elif softmax_output > 0.7:
                return "PRIVATE_CLASS_A"
            elif softmax_output > 0.4:
                return "PRIVATE_CLASS_B"
            elif softmax_output > 0.1:
                return "PRIVATE_CLASS_C"
            else:
                return "PRIVATE_CLASS_D"
                
        except Exception as e:
            logger.warning(f"Error determinando clasificación: {e}")
            return "ERROR"
    
    def _reset_accumulation_state(self):
        """Resetea el estado de acumulación para el siguiente ciclo"""
        self.pending_inputs.clear()
        self.config.accumulated_inputs = 0.0
        self.config.input_count = 0
        self.is_accumulating = False
    
    def _update_output_private_metrics(self, softmax_output: float, confidence: float, classification: str):
        """Actualiza las métricas específicas de salida privada"""
        self.output_private_metrics['total_processing_cycles'] += 1
        self.output_private_metrics['successful_classifications'] += 1
        
        # Actualizar métricas de confianza
        self.output_private_metrics['max_confidence'] = max(self.output_private_metrics['max_confidence'], confidence)
        self.output_private_metrics['min_confidence'] = min(self.output_private_metrics['min_confidence'], confidence)
        
        # Contar tipos de predicciones
        if confidence >= self.config.confidence_threshold:
            self.output_private_metrics['high_confidence_predictions'] += 1
        else:
            self.output_private_metrics['low_confidence_predictions'] += 1
        
        # Actualizar promedio de confianza
        total_cycles = self.output_private_metrics['total_processing_cycles']
        current_avg = self.output_private_metrics['average_confidence']
        self.output_private_metrics['average_confidence'] = ((current_avg * (total_cycles - 1)) + confidence) / total_cycles
        
        # Guardar historial
        self.classification_history.append(classification)
        self.confidence_history.append(confidence)
        
        if len(self.classification_history) > 100:
            self.classification_history.pop(0)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
    
    def get_output_private_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona de salida privada"""
        return {
            'neuron_id': self.config.neuron_id,
            'output_type': self.config.output_type,
            'privacy_level': self.config.privacy_level,
            'output_private_metrics': self.output_private_metrics,
            'current_classification': self.current_classification,
            'current_confidence': self.current_confidence,
            'classification_history': self.classification_history[-10:],
            'confidence_history': self.confidence_history[-10:],
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_classification_time': self.last_classification_time,
            'confidence_threshold': self.config.confidence_threshold,
            'access_control': self.config.access_control,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_classification_analysis(self) -> Dict[str, Any]:
        """Analiza el patrón de clasificaciones de la neurona"""
        if not self.classification_history:
            return {'status': 'no_data'}
        
        classifications = np.array(self.classification_history)
        confidences = np.array(self.confidence_history)
        
        return {
            'total_classifications': len(classifications),
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'high_confidence_rate': float(np.sum(confidences >= self.config.confidence_threshold) / len(confidences) * 100),
            'most_common_classification': str(np.bincount(classifications.astype(str)).argmax()) if len(classifications) > 0 else "N/A",
            'classification_diversity': len(set(classifications)),
            'confidence_stability': float(1.0 - np.std(confidences) / max(0.001, np.mean(confidences)))
        }
    
    def log_access_attempt(self, source: str, success: bool):
        """Registra un intento de acceso"""
        access_record = {
            'source': source,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'privacy_level': self.config.privacy_level
        }
        
        self.access_log.append(access_record)
        self.output_private_metrics['access_attempts'] += 1
        
        if success:
            self.output_private_metrics['successful_accesses'] += 1
        
        # Mantener solo los últimos 50 registros
        if len(self.access_log) > 50:
            self.access_log.pop(0)

# Función principal para ejecutar la neurona como módulo independiente
async def main():
    """Función principal para ejecutar la neurona como proceso independiente"""
    print("=" * 60)
    print("NEURONA OUTPUT PRIVADA 01 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        # Crear instancia de la neurona
        neurona = NeuronaOutputPrivada01()
        
        # Inicializar
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Nivel de privacidad: {neurona.config.privacy_level}")
        print(f"   Umbral de confianza: {neurona.config.confidence_threshold}")
        print(f"   Control de acceso: {neurona.config.access_control}")
        print(f"   Neuronas de entrada esperadas: {neurona.config.expected_inputs}")
        
        # Simular recepción de entradas de las neuronas de hidden layer 3
        print("\n📊 Simulando recepción de entradas de hidden layer 3...")
        
        # Simular múltiples ciclos de procesamiento
        for cycle in range(5):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            # Generar entradas simuladas de las 10 neuronas de hidden layer 3
            input_values = np.random.gamma(shape=3, scale=0.2, size=10)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.02)  # Simular tiempo de procesamiento
            
            # Esperar un poco para ver el procesamiento
            await asyncio.sleep(0.1)
        
        # Mostrar estadísticas avanzadas
        stats = neurona.get_output_private_statistics()
        print(f"\n📈 Estadísticas de la neurona output privada:")
        print(f"   Ciclos de procesamiento: {stats['output_private_metrics']['total_processing_cycles']}")
        print(f"   Clasificaciones exitosas: {stats['output_private_metrics']['successful_classifications']}")
        print(f"   Predicciones alta confianza: {stats['output_private_metrics']['high_confidence_predictions']}")
        print(f"   Predicciones baja confianza: {stats['output_private_metrics']['low_confidence_predictions']}")
        print(f"   Promedio de confianza: {stats['output_private_metrics']['average_confidence']:.4f}")
        print(f"   Violaciones de privacidad: {stats['output_private_metrics']['privacy_violations']}")
        print(f"   Tiempo promedio de clasificación: {stats['output_private_metrics']['classification_time']:.4f}s")
        
        # Mostrar análisis de clasificaciones
        analysis = neurona.get_classification_analysis()
        if analysis['status'] != 'no_data':
            print(f"\n🔍 Análisis avanzado de clasificaciones:")
            print(f"   Total de clasificaciones: {analysis['total_classifications']}")
            print(f"   Confianza promedio: {analysis['average_confidence']:.4f}")
            print(f"   Tasa de alta confianza: {analysis['high_confidence_rate']:.1f}%")
            print(f"   Clasificación más común: {analysis['most_common_classification']}")
            print(f"   Diversidad de clasificaciones: {analysis['classification_diversity']}")
            print(f"   Estabilidad de confianza: {analysis['confidence_stability']:.4f}")
        
        # Mostrar estado completo
        state = neurona.get_state()
        print(f"\n🔍 Estado completo de la neurona:")
        print(json.dumps(state, indent=2, default=str))
        
        # Cerrar neurona
        await neurona.shutdown()
        
        print("\n" + "=" * 60)
        print("✅ NEURONA OUTPUT PRIVADA 01 COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error en neurona output privada 01: {e}")
        logger.error(f"Error en neurona output privada 01: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar neurona
    asyncio.run(main())
