"""
Neurona Output Pública 01 - Módulo Independiente
==============================================

Primera neurona de salida para respuestas públicas de la red neuronal modular.
Esta neurona recibe datos de las 10 neuronas de hidden layer 3 y los procesa
para generar respuestas públicas del sistema.

Características:
- Tipo: OUTPUT
- Índice: 01
- Activación: Softmax
- Conexiones de entrada: 10 neuronas de hidden layer 3
- Propósito: Respuestas públicas del sistema
- Acceso: Usuarios externos y componentes públicos

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
class NeuronaOutputPublica01Config(NeuronConfig):
    """Configuración específica para la neurona output pública 01"""
    
    def __init__(self):
        super().__init__(
            neuron_id="output_publica_01",
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
        
        # Configuraciones específicas para neurona de salida pública
        self.input_neurons = [f"hidden3_{i:02d}" for i in range(1, 11)]
        self.output_type = "respuestas_publicas"
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0
        
        # Parámetros específicos para salida pública
        self.privacy_level = "PUBLIC"
        self.encryption_enabled = False
        self.access_control = ["public", "users", "external_systems", "api_clients"]
        self.logging_level = "PUBLIC"
        self.confidence_threshold = 0.6
        self.response_format = "USER_FRIENDLY"

class NeuronaOutputPublica01(OutputNeuron):
    """
    Neurona Output Pública 01 - Genera respuestas públicas del sistema.
    
    Esta neurona es responsable de:
    - Recibir y procesar entradas de las 10 neuronas de hidden layer 3
    - Aplicar función de activación Softmax para clasificación
    - Generar respuestas públicas amigables para el usuario
    - Mantener logs públicos y accesibles
    - Proporcionar información útil sin comprometer privacidad
    """
    
    def __init__(self):
        """Inicializa la neurona output pública 01"""
        config = NeuronaOutputPublica01Config()
        super().__init__(config)
        
        # Métricas específicas de salida pública
        self.output_public_metrics = {
            'total_processing_cycles': 0,
            'successful_public_classifications': 0,
            'high_confidence_public_predictions': 0,
            'medium_confidence_public_predictions': 0,
            'low_confidence_public_predictions': 0,
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': float('inf'),
            'public_response_time': 0.0,
            'user_friendly_responses': 0,
            'external_access_attempts': 0,
            'successful_external_accesses': 0,
            'api_calls_served': 0
        }
        
        # Estado específico para salida pública
        self.pending_inputs = {}
        self.public_classification_history = []
        self.confidence_history = []
        self.user_response_history = []
        self.is_accumulating = False
        self.last_response_time = 0.0
        self.current_public_classification = None
        self.current_confidence = 0.0
        self.current_user_response = None
        self.access_log = []
        self.api_response_cache = {}
        
        logger.info(f"Neurona output pública 01 inicializada - ID: {self.config.neuron_id}")
    
    async def receive_input_from_neuron(self, input_data: float, source_neuron: str) -> bool:
        """Recibe entrada de una neurona específica y la acumula."""
        try:
            if source_neuron not in self.config.input_neurons:
                logger.warning(f"Entrada de neurona no esperada: {source_neuron}")
                return False
            
            # Verificar acceso público (más permisivo)
            if not self._check_public_access_authorization(source_neuron):
                self.output_public_metrics['external_access_attempts'] += 1
                logger.info(f"Intento de acceso público desde {source_neuron}")
                return False
            
            # Almacenar entrada
            self.pending_inputs[source_neuron] = input_data
            self.config.accumulated_inputs += input_data
            self.config.input_count += 1
            
            # Verificar si tenemos todas las entradas esperadas
            if len(self.pending_inputs) >= self.config.expected_inputs:
                await self._process_accumulated_inputs_for_public_response()
            
            return True
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo entrada en neurona {self.config.neuron_id}: {e}")
            return False
    
    def _check_public_access_authorization(self, source_neuron: str) -> bool:
        """Verifica acceso público (más permisivo)"""
        try:
            # Verificación pública más permisiva
            authorized_sources = self.config.input_neurons
            return source_neuron in authorized_sources
            
        except Exception as e:
            logger.warning(f"Error verificando autorización pública: {e}")
            return False
    
    async def _process_accumulated_inputs_for_public_response(self):
        """Procesa entradas para generar respuesta pública"""
        try:
            self.is_accumulating = True
            start_time = time.time()
            
            # Aplicar función de activación Softmax
            activation_input = self.config.accumulated_inputs + self.config.bias
            softmax_output = self._apply_public_softmax(activation_input)
            
            # Calcular confianza para respuesta pública
            confidence = self._calculate_public_confidence(softmax_output)
            
            # Determinar clasificación pública
            public_classification = self._determine_public_classification(softmax_output, confidence)
            
            # Generar respuesta amigable para el usuario
            user_friendly_response = self._generate_user_friendly_response(public_classification, confidence)
            
            # Actualizar métricas
            self._update_output_public_metrics(softmax_output, confidence, public_classification, user_friendly_response)
            
            # Procesar con la neurona base
            output = await self.process_input(softmax_output, source_neuron="internal")
            
            # Actualizar estado
            self.current_public_classification = public_classification
            self.current_confidence = confidence
            self.current_user_response = user_friendly_response
            
            # Limpiar estado para el siguiente ciclo
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.output_public_metrics['public_response_time'] = processing_time
            self.last_response_time = time.time()
            
            logger.debug(f"Neurona {self.config.neuron_id} respuesta pública: {public_classification} (confianza: {confidence:.4f})")
            
        except Exception as e:
            self.is_accumulating = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error procesando entradas acumuladas en neurona {self.config.neuron_id}: {e}")
    
    def _apply_public_softmax(self, activation_input: float) -> float:
        """Aplica función de activación Softmax para respuestas públicas"""
        try:
            # Softmax público con normalización amigable
            exp_input = np.exp(np.clip(activation_input, -100, 100))  # Rango más conservador
            return exp_input / (1 + exp_input)
            
        except Exception as e:
            logger.warning(f"Error aplicando Softmax público: {e}")
            return 0.0
    
    def _calculate_public_confidence(self, softmax_output: float) -> float:
        """Calcula confianza para respuesta pública"""
        try:
            # Confianza pública más conservadora
            base_confidence = min(1.0, max(0.0, abs(softmax_output) * 1.5))  # Factor más conservador
            
            # Factor de estabilidad basado en historial público
            if len(self.confidence_history) > 0:
                recent_confidence_std = np.std(self.confidence_history[-10:])
                stability_factor = max(0.7, 1.0 - recent_confidence_std)  # Más estable
                base_confidence *= stability_factor
            
            return base_confidence
            
        except Exception as e:
            logger.warning(f"Error calculando confianza pública: {e}")
            return 0.0
    
    def _determine_public_classification(self, softmax_output: float, confidence: float) -> str:
        """Determina clasificación pública basada en la salida y confianza"""
        try:
            if confidence < self.config.confidence_threshold:
                return "INSUFFICIENT_CONFIDENCE"
            elif softmax_output > 0.8:
                return "PUBLIC_CLASS_A_POSITIVE"
            elif softmax_output > 0.6:
                return "PUBLIC_CLASS_B_NEUTRAL"
            elif softmax_output > 0.4:
                return "PUBLIC_CLASS_C_CAUTIOUS"
            elif softmax_output > 0.2:
                return "PUBLIC_CLASS_D_NEGATIVE"
            else:
                return "PUBLIC_CLASS_E_REJECT"
                
        except Exception as e:
            logger.warning(f"Error determinando clasificación pública: {e}")
            return "ERROR_PUBLIC"
    
    def _generate_user_friendly_response(self, classification: str, confidence: float) -> str:
        """Genera respuesta amigable para el usuario"""
        try:
            response_templates = {
                "PUBLIC_CLASS_A_POSITIVE": f"Resultado muy positivo (confianza: {confidence:.1%})",
                "PUBLIC_CLASS_B_NEUTRAL": f"Resultado neutral (confianza: {confidence:.1%})",
                "PUBLIC_CLASS_C_CAUTIOUS": f"Resultado con precaución (confianza: {confidence:.1%})",
                "PUBLIC_CLASS_D_NEGATIVE": f"Resultado negativo (confianza: {confidence:.1%})",
                "PUBLIC_CLASS_E_REJECT": f"Resultado rechazado (confianza: {confidence:.1%})",
                "INSUFFICIENT_CONFIDENCE": "Confianza insuficiente para dar una respuesta precisa",
                "ERROR_PUBLIC": "Error en el procesamiento, por favor intente nuevamente"
            }
            
            return response_templates.get(classification, f"Respuesta: {classification} (confianza: {confidence:.1%})")
            
        except Exception as e:
            logger.warning(f"Error generando respuesta amigable: {e}")
            return "Error generando respuesta"
    
    def _reset_accumulation_state(self):
        """Resetea el estado de acumulación para el siguiente ciclo"""
        self.pending_inputs.clear()
        self.config.accumulated_inputs = 0.0
        self.config.input_count = 0
        self.is_accumulating = False
    
    def _update_output_public_metrics(self, softmax_output: float, confidence: float, classification: str, user_response: str):
        """Actualiza las métricas específicas de salida pública"""
        self.output_public_metrics['total_processing_cycles'] += 1
        self.output_public_metrics['successful_public_classifications'] += 1
        
        # Actualizar métricas de confianza
        self.output_public_metrics['max_confidence'] = max(self.output_public_metrics['max_confidence'], confidence)
        self.output_public_metrics['min_confidence'] = min(self.output_public_metrics['min_confidence'], confidence)
        
        # Contar tipos de predicciones públicas
        if confidence >= 0.8:
            self.output_public_metrics['high_confidence_public_predictions'] += 1
        elif confidence >= self.config.confidence_threshold:
            self.output_public_metrics['medium_confidence_public_predictions'] += 1
        else:
            self.output_public_metrics['low_confidence_public_predictions'] += 1
        
        # Actualizar promedio de confianza
        total_cycles = self.output_public_metrics['total_processing_cycles']
        current_avg = self.output_public_metrics['average_confidence']
        self.output_public_metrics['average_confidence'] = ((current_avg * (total_cycles - 1)) + confidence) / total_cycles
        
        # Contar respuestas amigables
        if "confianza" in user_response.lower():
            self.output_public_metrics['user_friendly_responses'] += 1
        
        # Guardar historial
        self.public_classification_history.append(classification)
        self.confidence_history.append(confidence)
        self.user_response_history.append(user_response)
        
        if len(self.public_classification_history) > 100:
            self.public_classification_history.pop(0)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        if len(self.user_response_history) > 100:
            self.user_response_history.pop(0)
    
    def get_output_public_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona de salida pública"""
        return {
            'neuron_id': self.config.neuron_id,
            'output_type': self.config.output_type,
            'privacy_level': self.config.privacy_level,
            'response_format': self.config.response_format,
            'output_public_metrics': self.output_public_metrics,
            'current_public_classification': self.current_public_classification,
            'current_confidence': self.current_confidence,
            'current_user_response': self.current_user_response,
            'public_classification_history': self.public_classification_history[-10:],
            'confidence_history': self.confidence_history[-10:],
            'user_response_history': self.user_response_history[-5:],
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_response_time': self.last_response_time,
            'confidence_threshold': self.config.confidence_threshold,
            'access_control': self.config.access_control,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_public_classification_analysis(self) -> Dict[str, Any]:
        """Analiza el patrón de clasificaciones públicas de la neurona"""
        if not self.public_classification_history:
            return {'status': 'no_data'}
        
        classifications = np.array(self.public_classification_history)
        confidences = np.array(self.confidence_history)
        
        return {
            'total_public_classifications': len(classifications),
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'high_confidence_rate': float(np.sum(confidences >= 0.8) / len(confidences) * 100),
            'medium_confidence_rate': float(np.sum(confidences >= self.config.confidence_threshold) / len(confidences) * 100),
            'most_common_public_classification': str(np.bincount(classifications.astype(str)).argmax()) if len(classifications) > 0 else "N/A",
            'public_classification_diversity': len(set(classifications)),
            'confidence_stability': float(1.0 - np.std(confidences) / max(0.001, np.mean(confidences))),
            'user_friendly_response_rate': float(self.output_public_metrics['user_friendly_responses'] / max(1, self.output_public_metrics['total_processing_cycles']) * 100),
            'external_access_success_rate': float(self.output_public_metrics['successful_external_accesses'] / max(1, self.output_public_metrics['external_access_attempts']) * 100)
        }

# Función principal
async def main():
    print("=" * 60)
    print("NEURONA OUTPUT PÚBLICA 01 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        neurona = NeuronaOutputPublica01()
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Nivel de privacidad: {neurona.config.privacy_level}")
        print(f"   Formato de respuesta: {neurona.config.response_format}")
        print(f"   Umbral de confianza: {neurona.config.confidence_threshold}")
        
        print("\n📊 Simulando recepción de entradas...")
        
        for cycle in range(3):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            input_values = np.random.gamma(shape=2, scale=0.4, size=10)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.02)
            
            await asyncio.sleep(0.1)
        
        stats = neurona.get_output_public_statistics()
        print(f"\n📈 Ciclos de procesamiento: {stats['output_public_metrics']['total_processing_cycles']}")
        print(f"   Promedio de confianza: {stats['output_public_metrics']['average_confidence']:.4f}")
        print(f"   Respuestas amigables: {stats['output_public_metrics']['user_friendly_responses']}")
        
        await neurona.shutdown()
        print("\n✅ NEURONA OUTPUT PÚBLICA 01 COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error en neurona output pública 01: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
