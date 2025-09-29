"""
Neurona Output Pública 02 - Módulo Independiente
==============================================

Segunda neurona de salida para respuestas públicas de la red neuronal modular.
Esta neurona recibe datos de las 10 neuronas de hidden layer 3 y los procesa
para generar respuestas públicas del sistema con enfoque en interactividad.

Características:
- Tipo: OUTPUT
- Índice: 02
- Activación: Softmax
- Conexiones de entrada: 10 neuronas de hidden layer 3
- Propósito: Respuestas públicas interactivas del sistema
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
class NeuronaOutputPublica02Config(NeuronConfig):
    """Configuración específica para la neurona output pública 02"""
    
    def __init__(self):
        super().__init__(
            neuron_id="output_publica_02",
            neuron_type=NeuronType.OUTPUT,
            layer_index=4,
            neuron_index=2,
            activation=ActivationType.SOFTMAX,
            bias=0.0,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0001,
            max_connections=20,
            processing_timeout=1.0
        )
        
        # Configuraciones específicas para neurona de salida pública interactiva
        self.input_neurons = [f"hidden3_{i:02d}" for i in range(1, 11)]
        self.output_type = "respuestas_publicas_interactivas"
        self.expected_inputs = len(self.input_neurons)
        self.accumulated_inputs = 0.0
        self.input_count = 0
        
        # Parámetros específicos para salida pública interactiva
        self.privacy_level = "PUBLIC"
        self.encryption_enabled = False
        self.access_control = ["public", "users", "external_systems", "api_clients", "interactive_services"]
        self.logging_level = "PUBLIC"
        self.confidence_threshold = 0.5  # Más permisivo para interactividad
        self.response_format = "INTERACTIVE_USER_FRIENDLY"
        self.interactivity_level = "HIGH"

class NeuronaOutputPublica02(OutputNeuron):
    """
    Neurona Output Pública 02 - Genera respuestas públicas interactivas del sistema.
    
    Esta neurona es responsable de:
    - Recibir y procesar entradas de las 10 neuronas de hidden layer 3
    - Aplicar función de activación Softmax para clasificación interactiva
    - Generar respuestas públicas interactivas y dinámicas
    - Mantener logs públicos y accesibles
    - Proporcionar información útil con alta interactividad
    """
    
    def __init__(self):
        """Inicializa la neurona output pública 02"""
        config = NeuronaOutputPublica02Config()
        super().__init__(config)
        
        # Métricas específicas de salida pública interactiva
        self.output_public_interactive_metrics = {
            'total_processing_cycles': 0,
            'successful_interactive_classifications': 0,
            'high_interactivity_responses': 0,
            'medium_interactivity_responses': 0,
            'low_interactivity_responses': 0,
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': float('inf'),
            'interactive_response_time': 0.0,
            'dynamic_responses_generated': 0,
            'external_interaction_attempts': 0,
            'successful_external_interactions': 0,
            'api_interactive_calls_served': 0,
            'user_engagement_score': 0.0
        }
        
        # Estado específico para salida pública interactiva
        self.pending_inputs = {}
        self.interactive_classification_history = []
        self.confidence_history = []
        self.interactive_response_history = []
        self.user_engagement_history = []
        self.is_accumulating = False
        self.last_interactive_response_time = 0.0
        self.current_interactive_classification = None
        self.current_confidence = 0.0
        self.current_interactive_response = None
        self.current_engagement_score = 0.0
        self.access_log = []
        self.interactive_response_cache = {}
        self.engagement_patterns = {}
        
        logger.info(f"Neurona output pública 02 inicializada - ID: {self.config.neuron_id}")
    
    async def receive_input_from_neuron(self, input_data: float, source_neuron: str) -> bool:
        """Recibe entrada de una neurona específica y la acumula."""
        try:
            if source_neuron not in self.config.input_neurons:
                logger.warning(f"Entrada de neurona no esperada: {source_neuron}")
                return False
            
            # Verificar acceso público interactivo (más permisivo)
            if not self._check_interactive_public_access_authorization(source_neuron):
                self.output_public_interactive_metrics['external_interaction_attempts'] += 1
                logger.info(f"Intento de acceso público interactivo desde {source_neuron}")
                return False
            
            # Almacenar entrada con análisis de interactividad
            self.pending_inputs[source_neuron] = input_data
            self.config.accumulated_inputs += input_data
            self.config.input_count += 1
            
            # Verificar si tenemos todas las entradas esperadas
            if len(self.pending_inputs) >= self.config.expected_inputs:
                await self._process_accumulated_inputs_for_interactive_response()
            
            return True
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo entrada en neurona {self.config.neuron_id}: {e}")
            return False
    
    def _check_interactive_public_access_authorization(self, source_neuron: str) -> bool:
        """Verifica acceso público interactivo (muy permisivo)"""
        try:
            # Verificación pública interactiva muy permisiva
            authorized_sources = self.config.input_neurons
            return source_neuron in authorized_sources
            
        except Exception as e:
            logger.warning(f"Error verificando autorización pública interactiva: {e}")
            return False
    
    async def _process_accumulated_inputs_for_interactive_response(self):
        """Procesa entradas para generar respuesta pública interactiva"""
        try:
            self.is_accumulating = True
            start_time = time.time()
            
            # Aplicar función de activación Softmax interactiva
            activation_input = self.config.accumulated_inputs + self.config.bias
            softmax_output = self._apply_interactive_softmax(activation_input)
            
            # Calcular confianza para respuesta interactiva
            confidence = self._calculate_interactive_confidence(softmax_output)
            
            # Determinar clasificación interactiva
            interactive_classification = self._determine_interactive_classification(softmax_output, confidence)
            
            # Generar respuesta interactiva
            interactive_response = self._generate_interactive_response(interactive_classification, confidence)
            
            # Calcular score de engagement
            engagement_score = self._calculate_engagement_score(interactive_response, confidence)
            
            # Actualizar métricas
            self._update_output_public_interactive_metrics(softmax_output, confidence, interactive_classification, interactive_response, engagement_score)
            
            # Procesar con la neurona base
            output = await self.process_input(softmax_output, source_neuron="internal")
            
            # Actualizar estado
            self.current_interactive_classification = interactive_classification
            self.current_confidence = confidence
            self.current_interactive_response = interactive_response
            self.current_engagement_score = engagement_score
            
            # Limpiar estado para el siguiente ciclo
            self._reset_accumulation_state()
            
            processing_time = time.time() - start_time
            self.output_public_interactive_metrics['interactive_response_time'] = processing_time
            self.last_interactive_response_time = time.time()
            
            logger.debug(f"Neurona {self.config.neuron_id} respuesta interactiva: {interactive_classification} (engagement: {engagement_score:.2f})")
            
        except Exception as e:
            self.is_accumulating = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error procesando entradas acumuladas en neurona {self.config.neuron_id}: {e}")
    
    def _apply_interactive_softmax(self, activation_input: float) -> float:
        """Aplica función de activación Softmax para respuestas interactivas"""
        try:
            # Softmax interactivo con mayor sensibilidad
            exp_input = np.exp(np.clip(activation_input, -50, 50))  # Rango más amplio
            return exp_input / (1 + exp_input)
            
        except Exception as e:
            logger.warning(f"Error aplicando Softmax interactivo: {e}")
            return 0.0
    
    def _calculate_interactive_confidence(self, softmax_output: float) -> float:
        """Calcula confianza para respuesta interactiva"""
        try:
            # Confianza interactiva más dinámica
            base_confidence = min(1.0, max(0.0, abs(softmax_output) * 2.5))  # Factor más dinámico
            
            # Factor de engagement basado en historial
            if len(self.confidence_history) > 0:
                recent_confidence_std = np.std(self.confidence_history[-10:])
                engagement_factor = max(0.5, 1.0 + recent_confidence_std * 0.1)  # Más engagement con variabilidad
                base_confidence *= engagement_factor
            
            return base_confidence
            
        except Exception as e:
            logger.warning(f"Error calculando confianza interactiva: {e}")
            return 0.0
    
    def _determine_interactive_classification(self, softmax_output: float, confidence: float) -> str:
        """Determina clasificación interactiva basada en la salida y confianza"""
        try:
            if confidence < self.config.confidence_threshold:
                return "INTERACTIVE_EXPLORATION"
            elif softmax_output > 0.8:
                return "INTERACTIVE_CLASS_A_ENTHUSIASTIC"
            elif softmax_output > 0.6:
                return "INTERACTIVE_CLASS_B_POSITIVE"
            elif softmax_output > 0.4:
                return "INTERACTIVE_CLASS_C_NEUTRAL"
            elif softmax_output > 0.2:
                return "INTERACTIVE_CLASS_D_CAUTIOUS"
            else:
                return "INTERACTIVE_CLASS_E_SKEPTICAL"
                
        except Exception as e:
            logger.warning(f"Error determinando clasificación interactiva: {e}")
            return "ERROR_INTERACTIVE"
    
    def _generate_interactive_response(self, classification: str, confidence: float) -> str:
        """Genera respuesta interactiva dinámica"""
        try:
            response_templates = {
                "INTERACTIVE_CLASS_A_ENTHUSIASTIC": f"¡Excelente! Resultado muy prometedor 🚀 (confianza: {confidence:.1%}) - ¿Te gustaría explorar más?",
                "INTERACTIVE_CLASS_B_POSITIVE": f"Muy bien! Resultado positivo 👍 (confianza: {confidence:.1%}) - ¿Qué opinas?",
                "INTERACTIVE_CLASS_C_NEUTRAL": f"Resultado neutral 🤔 (confianza: {confidence:.1%}) - ¿Te gustaría probar algo diferente?",
                "INTERACTIVE_CLASS_D_CAUTIOUS": f"Resultado con precaución ⚠️ (confianza: {confidence:.1%}) - ¿Quieres revisar los datos?",
                "INTERACTIVE_CLASS_E_SKEPTICAL": f"Resultado cuestionable ❓ (confianza: {confidence:.1%}) - ¿Podemos ajustar el enfoque?",
                "INTERACTIVE_EXPLORATION": "¡Exploremos juntos! 🔍 Confianza insuficiente - ¿Qué te gustaría probar?",
                "ERROR_INTERACTIVE": "Oops! Algo salió mal 😅 - ¿Intentamos de nuevo?"
            }
            
            return response_templates.get(classification, f"Respuesta interactiva: {classification} (confianza: {confidence:.1%})")
            
        except Exception as e:
            logger.warning(f"Error generando respuesta interactiva: {e}")
            return "Error generando respuesta interactiva"
    
    def _calculate_engagement_score(self, response: str, confidence: float) -> float:
        """Calcula score de engagement basado en la respuesta"""
        try:
            base_score = confidence
            
            # Factores de engagement
            if "🚀" in response or "¡Excelente!" in response:
                base_score += 0.3
            elif "👍" in response or "Muy bien!" in response:
                base_score += 0.2
            elif "🤔" in response or "¿Qué opinas?" in response:
                base_score += 0.1
            elif "⚠️" in response or "precaución" in response:
                base_score += 0.05
            elif "❓" in response or "cuestionable" in response:
                base_score -= 0.1
            elif "🔍" in response or "exploremos" in response:
                base_score += 0.15
            
            # Factor de interactividad
            if "¿" in response:  # Preguntas
                base_score += 0.1
            
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            logger.warning(f"Error calculando engagement score: {e}")
            return confidence
    
    def _reset_accumulation_state(self):
        """Resetea el estado de acumulación para el siguiente ciclo"""
        self.pending_inputs.clear()
        self.config.accumulated_inputs = 0.0
        self.config.input_count = 0
        self.is_accumulating = False
    
    def _update_output_public_interactive_metrics(self, softmax_output: float, confidence: float, classification: str, interactive_response: str, engagement_score: float):
        """Actualiza las métricas específicas de salida pública interactiva"""
        self.output_public_interactive_metrics['total_processing_cycles'] += 1
        self.output_public_interactive_metrics['successful_interactive_classifications'] += 1
        
        # Actualizar métricas de confianza
        self.output_public_interactive_metrics['max_confidence'] = max(self.output_public_interactive_metrics['max_confidence'], confidence)
        self.output_public_interactive_metrics['min_confidence'] = min(self.output_public_interactive_metrics['min_confidence'], confidence)
        
        # Contar tipos de respuestas interactivas
        if engagement_score >= 0.8:
            self.output_public_interactive_metrics['high_interactivity_responses'] += 1
        elif engagement_score >= 0.5:
            self.output_public_interactive_metrics['medium_interactivity_responses'] += 1
        else:
            self.output_public_interactive_metrics['low_interactivity_responses'] += 1
        
        # Actualizar promedio de confianza y engagement
        total_cycles = self.output_public_interactive_metrics['total_processing_cycles']
        current_avg_confidence = self.output_public_interactive_metrics['average_confidence']
        self.output_public_interactive_metrics['average_confidence'] = ((current_avg_confidence * (total_cycles - 1)) + confidence) / total_cycles
        
        current_engagement = self.output_public_interactive_metrics['user_engagement_score']
        self.output_public_interactive_metrics['user_engagement_score'] = ((current_engagement * (total_cycles - 1)) + engagement_score) / total_cycles
        
        # Contar respuestas dinámicas
        if "¿" in interactive_response or "🚀" in interactive_response or "👍" in interactive_response:
            self.output_public_interactive_metrics['dynamic_responses_generated'] += 1
        
        # Guardar historial
        self.interactive_classification_history.append(classification)
        self.confidence_history.append(confidence)
        self.interactive_response_history.append(interactive_response)
        self.user_engagement_history.append(engagement_score)
        
        if len(self.interactive_classification_history) > 100:
            self.interactive_classification_history.pop(0)
        if len(self.confidence_history) > 100:
            self.confidence_history.pop(0)
        if len(self.interactive_response_history) > 100:
            self.interactive_response_history.pop(0)
        if len(self.user_engagement_history) > 100:
            self.user_engagement_history.pop(0)
    
    def get_output_public_interactive_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de la neurona de salida pública interactiva"""
        return {
            'neuron_id': self.config.neuron_id,
            'output_type': self.config.output_type,
            'privacy_level': self.config.privacy_level,
            'response_format': self.config.response_format,
            'interactivity_level': self.config.interactivity_level,
            'output_public_interactive_metrics': self.output_public_interactive_metrics,
            'current_interactive_classification': self.current_interactive_classification,
            'current_confidence': self.current_confidence,
            'current_interactive_response': self.current_interactive_response,
            'current_engagement_score': self.current_engagement_score,
            'interactive_classification_history': self.interactive_classification_history[-10:],
            'confidence_history': self.confidence_history[-10:],
            'interactive_response_history': self.interactive_response_history[-5:],
            'user_engagement_history': self.user_engagement_history[-10:],
            'is_accumulating': self.is_accumulating,
            'pending_inputs_count': len(self.pending_inputs),
            'last_interactive_response_time': self.last_interactive_response_time,
            'confidence_threshold': self.config.confidence_threshold,
            'access_control': self.config.access_control,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_interactive_classification_analysis(self) -> Dict[str, Any]:
        """Analiza el patrón de clasificaciones interactivas de la neurona"""
        if not self.interactive_classification_history:
            return {'status': 'no_data'}
        
        classifications = np.array(self.interactive_classification_history)
        confidences = np.array(self.confidence_history)
        engagement_scores = np.array(self.user_engagement_history)
        
        return {
            'total_interactive_classifications': len(classifications),
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'average_engagement_score': float(np.mean(engagement_scores)),
            'engagement_std': float(np.std(engagement_scores)),
            'high_interactivity_rate': float(np.sum(engagement_scores >= 0.8) / len(engagement_scores) * 100),
            'medium_interactivity_rate': float(np.sum(engagement_scores >= 0.5) / len(engagement_scores) * 100),
            'most_common_interactive_classification': str(np.bincount(classifications.astype(str)).argmax()) if len(classifications) > 0 else "N/A",
            'interactive_classification_diversity': len(set(classifications)),
            'confidence_stability': float(1.0 - np.std(confidences) / max(0.001, np.mean(confidences))),
            'engagement_stability': float(1.0 - np.std(engagement_scores) / max(0.001, np.mean(engagement_scores))),
            'dynamic_response_rate': float(self.output_public_interactive_metrics['dynamic_responses_generated'] / max(1, self.output_public_interactive_metrics['total_processing_cycles']) * 100),
            'external_interaction_success_rate': float(self.output_public_interactive_metrics['successful_external_interactions'] / max(1, self.output_public_interactive_metrics['external_interaction_attempts']) * 100)
        }

# Función principal
async def main():
    print("=" * 60)
    print("NEURONA OUTPUT PÚBLICA 02 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        neurona = NeuronaOutputPublica02()
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Nivel de privacidad: {neurona.config.privacy_level}")
        print(f"   Nivel de interactividad: {neurona.config.interactivity_level}")
        print(f"   Formato de respuesta: {neurona.config.response_format}")
        
        print("\n📊 Simulando recepción de entradas...")
        
        for cycle in range(3):
            print(f"\n--- Ciclo {cycle + 1} ---")
            
            input_values = np.random.gamma(shape=2, scale=0.4, size=10)
            
            for i, (neuron_id, value) in enumerate(zip(neurona.config.input_neurons, input_values)):
                success = await neurona.receive_input_from_neuron(value, neuron_id)
                print(f"   Entrada {i+1} ({neuron_id}): {value:.4f} -> {'✅' if success else '❌'}")
                await asyncio.sleep(0.02)
            
            await asyncio.sleep(0.1)
        
        stats = neurona.get_output_public_interactive_statistics()
        print(f"\n📈 Ciclos de procesamiento: {stats['output_public_interactive_metrics']['total_processing_cycles']}")
        print(f"   Promedio de confianza: {stats['output_public_interactive_metrics']['average_confidence']:.4f}")
        print(f"   Score de engagement: {stats['output_public_interactive_metrics']['user_engagement_score']:.4f}")
        print(f"   Respuestas dinámicas: {stats['output_public_interactive_metrics']['dynamic_responses_generated']}")
        
        await neurona.shutdown()
        print("\n✅ NEURONA OUTPUT PÚBLICA 02 COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error en neurona output pública 02: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
