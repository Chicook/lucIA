"""
Mejorador de Predicciones
Mejora la calidad y precisión de las predicciones de los modelos de generación.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import re
import json

logger = logging.getLogger('PredictionEnhancer')

class PredictionEnhancer:
    """
    Mejorador de predicciones que optimiza la calidad de las respuestas generadas.
    """
    
    def __init__(self):
        self.enhancement_metrics = {
            'predictions_enhanced': 0,
            'quality_improvements': 0,
            'accuracy_improvements': 0,
            'errors_corrected': 0,
            'context_improvements': 0
        }
        
        self.quality_thresholds = {
            'min_length': 10,
            'max_length': 1000,
            'min_confidence': 0.3,
            'max_confidence': 1.0
        }
        
        self.enhancement_strategies = {
            'grammar_correction': True,
            'context_enhancement': True,
            'confidence_boosting': True,
            'length_optimization': True,
            'coherence_improvement': True,
            'factual_validation': True
        }
    
    def enhance_prediction(self, prediction: Union[str, Dict], 
                          context: Optional[Dict] = None,
                          model_type: str = "text_generation") -> Union[str, Dict]:
        """
        Mejora una predicción basada en el contexto y tipo de modelo.
        
        Args:
            prediction: Predicción original
            context: Contexto adicional
            model_type: Tipo de modelo (text_generation, classification, etc.)
            
        Returns:
            Predicción mejorada
        """
        try:
            if isinstance(prediction, dict):
                return self._enhance_dict_prediction(prediction, context, model_type)
            elif isinstance(prediction, str):
                return self._enhance_text_prediction(prediction, context, model_type)
            else:
                logger.warning(f"Tipo de predicción no soportado: {type(prediction)}")
                return prediction
                
        except Exception as e:
            logger.error(f"Error mejorando predicción: {e}")
            return prediction
    
    def _enhance_text_prediction(self, text: str, context: Optional[Dict], 
                                model_type: str) -> str:
        """Mejora una predicción de texto"""
        try:
            enhanced_text = text
            
            # Aplicar mejoras de calidad
            if self.enhancement_strategies['grammar_correction']:
                enhanced_text = self._correct_grammar(enhanced_text)
            
            if self.enhancement_strategies['length_optimization']:
                enhanced_text = self._optimize_length(enhanced_text)
            
            if self.enhancement_strategies['coherence_improvement']:
                enhanced_text = self._improve_coherence(enhanced_text)
            
            if self.enhancement_strategies['context_enhancement'] and context:
                enhanced_text = self._enhance_with_context(enhanced_text, context)
            
            # Validar calidad final
            quality_score = self._calculate_quality_score(enhanced_text)
            if quality_score > self.quality_thresholds['min_confidence']:
                self.enhancement_metrics['predictions_enhanced'] += 1
                self.enhancement_metrics['quality_improvements'] += 1
                logger.info(f"Predicción de texto mejorada (calidad: {quality_score:.2f})")
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error mejorando predicción de texto: {e}")
            return text
    
    def _enhance_dict_prediction(self, prediction: Dict, context: Optional[Dict], 
                                model_type: str) -> Dict:
        """Mejora una predicción en formato diccionario"""
        try:
            enhanced_prediction = prediction.copy()
            
            # Mejorar texto si existe
            if 'text' in enhanced_prediction:
                enhanced_prediction['text'] = self._enhance_text_prediction(
                    enhanced_prediction['text'], context, model_type
                )
            
            if 'generated_response' in enhanced_prediction:
                enhanced_prediction['generated_response'] = self._enhance_text_prediction(
                    enhanced_prediction['generated_response'], context, model_type
                )
            
            # Mejorar confianza si existe
            if 'confidence' in enhanced_prediction:
                enhanced_prediction['confidence'] = self._boost_confidence(
                    enhanced_prediction['confidence']
                )
            
            if 'probabilities' in enhanced_prediction:
                enhanced_prediction['probabilities'] = self._normalize_probabilities(
                    enhanced_prediction['probabilities']
                )
            
            # Agregar metadatos de mejora
            enhanced_prediction['enhanced'] = True
            enhanced_prediction['enhancement_timestamp'] = datetime.now().isoformat()
            
            self.enhancement_metrics['predictions_enhanced'] += 1
            logger.info("Predicción de diccionario mejorada")
            
            return enhanced_prediction
            
        except Exception as e:
            logger.error(f"Error mejorando predicción de diccionario: {e}")
            return prediction
    
    def _correct_grammar(self, text: str) -> str:
        """Corrige errores gramaticales básicos"""
        try:
            # Correcciones básicas de gramática
            corrections = {
                r'\b(?:a|an)\s+([aeiou])': r'an \1',  # a/an
                r'\b(?:a|an)\s+([^aeiou])': r'a \1',
                r'\s+': ' ',  # Múltiples espacios
                r'\.\s*\.': '.',  # Múltiples puntos
                r'!\s*!': '!',  # Múltiples exclamaciones
                r'\?\s*\?': '?',  # Múltiples interrogaciones
            }
            
            corrected_text = text
            for pattern, replacement in corrections.items():
                corrected_text = re.sub(pattern, replacement, corrected_text)
            
            # Capitalizar primera letra
            if corrected_text and not corrected_text[0].isupper():
                corrected_text = corrected_text[0].upper() + corrected_text[1:]
            
            return corrected_text.strip()
            
        except Exception as e:
            logger.error(f"Error corrigiendo gramática: {e}")
            return text
    
    def _optimize_length(self, text: str) -> str:
        """Optimiza la longitud del texto"""
        try:
            min_length = self.quality_thresholds['min_length']
            max_length = self.quality_thresholds['max_length']
            
            if len(text) < min_length:
                # Expandir texto si es muy corto
                return self._expand_text(text)
            elif len(text) > max_length:
                # Truncar texto si es muy largo
                return self._truncate_text(text, max_length)
            else:
                return text
                
        except Exception as e:
            logger.error(f"Error optimizando longitud: {e}")
            return text
    
    def _expand_text(self, text: str) -> str:
        """Expande texto que es demasiado corto"""
        try:
            # Agregar contexto adicional para textos cortos
            if len(text.split()) < 3:
                return f"Basándome en la información disponible: {text}. ¿Te gustaría que profundice en algún aspecto específico?"
            else:
                return text
                
        except Exception as e:
            logger.error(f"Error expandiendo texto: {e}")
            return text
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Trunca texto que es demasiado largo"""
        try:
            if len(text) <= max_length:
                return text
            
            # Truncar en el último punto o espacio
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            last_space = truncated.rfind(' ')
            
            if last_period > max_length * 0.8:
                return truncated[:last_period + 1]
            elif last_space > max_length * 0.8:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
                
        except Exception as e:
            logger.error(f"Error truncando texto: {e}")
            return text[:max_length] + "..."
    
    def _improve_coherence(self, text: str) -> str:
        """Mejora la coherencia del texto"""
        try:
            # Agregar conectores para mejorar flujo
            connectors = {
                r'\.\s*([A-Z])': r'. Además, \1',
                r'!\s*([A-Z])': r'! Por otro lado, \1',
                r'\?\s*([A-Z])': r'? En este sentido, \1'
            }
            
            improved_text = text
            for pattern, replacement in connectors.items():
                improved_text = re.sub(pattern, replacement, improved_text, count=1)
            
            return improved_text
            
        except Exception as e:
            logger.error(f"Error mejorando coherencia: {e}")
            return text
    
    def _enhance_with_context(self, text: str, context: Dict) -> str:
        """Mejora el texto usando contexto adicional"""
        try:
            # Agregar información de contexto relevante
            if 'topic' in context:
                topic = context['topic']
                if topic.lower() not in text.lower():
                    text = f"En el contexto de {topic}: {text}"
            
            if 'user_level' in context:
                level = context['user_level']
                if level == 'beginner':
                    text = f"Para explicar de manera sencilla: {text}"
                elif level == 'expert':
                    text = f"Desde una perspectiva técnica: {text}"
            
            if 'session_context' in context:
                session = context['session_context']
                if session == 'security_training':
                    text = f"En el contexto de ciberseguridad: {text}"
            
            self.enhancement_metrics['context_improvements'] += 1
            return text
            
        except Exception as e:
            logger.error(f"Error mejorando con contexto: {e}")
            return text
    
    def _boost_confidence(self, confidence: float) -> float:
        """Mejora la confianza de una predicción"""
        try:
            if confidence < self.quality_thresholds['min_confidence']:
                # Aumentar confianza si es muy baja
                boosted_confidence = min(
                    confidence * 1.2, 
                    self.quality_thresholds['max_confidence']
                )
                return round(boosted_confidence, 3)
            else:
                return confidence
                
        except Exception as e:
            logger.error(f"Error mejorando confianza: {e}")
            return confidence
    
    def _normalize_probabilities(self, probabilities: List[float]) -> List[float]:
        """Normaliza probabilidades para que sumen 1.0"""
        try:
            if not probabilities:
                return probabilities
            
            total = sum(probabilities)
            if total == 0:
                return [1.0 / len(probabilities)] * len(probabilities)
            
            normalized = [p / total for p in probabilities]
            return [round(p, 4) for p in normalized]
            
        except Exception as e:
            logger.error(f"Error normalizando probabilidades: {e}")
            return probabilities
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calcula un puntaje de calidad para el texto"""
        try:
            if not text or len(text.strip()) == 0:
                return 0.0
            
            score = 0.0
            
            # Puntaje por longitud
            length_score = min(len(text) / 100, 1.0) * 0.3
            score += length_score
            
            # Puntaje por estructura (puntuación)
            punctuation_count = len(re.findall(r'[.!?]', text))
            structure_score = min(punctuation_count / 3, 1.0) * 0.2
            score += structure_score
            
            # Puntaje por vocabulario (palabras únicas)
            words = text.lower().split()
            unique_words = len(set(words))
            vocabulary_score = min(unique_words / 20, 1.0) * 0.2
            score += vocabulary_score
            
            # Puntaje por coherencia (palabras de conexión)
            connectors = ['además', 'por otro lado', 'sin embargo', 'por lo tanto', 'en consecuencia']
            connector_count = sum(1 for connector in connectors if connector in text.lower())
            coherence_score = min(connector_count / 2, 1.0) * 0.3
            score += coherence_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculando puntaje de calidad: {e}")
            return 0.5
    
    def enhance_security_predictions(self, prediction: Union[str, Dict], 
                                   security_context: Dict) -> Union[str, Dict]:
        """
        Mejora predicciones específicas de ciberseguridad.
        
        Args:
            prediction: Predicción original
            security_context: Contexto de seguridad
            
        Returns:
            Predicción mejorada para ciberseguridad
        """
        try:
            # Agregar contexto de seguridad
            if isinstance(prediction, str):
                enhanced = self._add_security_context(prediction, security_context)
            else:
                enhanced = prediction.copy()
                if 'text' in enhanced:
                    enhanced['text'] = self._add_security_context(enhanced['text'], security_context)
            
            # Validar contenido de seguridad
            enhanced = self._validate_security_content(enhanced)
            
            self.enhancement_metrics['predictions_enhanced'] += 1
            logger.info("Predicción de seguridad mejorada")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error mejorando predicción de seguridad: {e}")
            return prediction
    
    def _add_security_context(self, text: str, security_context: Dict) -> str:
        """Agrega contexto de seguridad al texto"""
        try:
            # Agregar información de seguridad relevante
            if 'threat_level' in security_context:
                threat_level = security_context['threat_level']
                if threat_level == 'high':
                    text = f"⚠️ ALERTA DE SEGURIDAD ALTA: {text}"
                elif threat_level == 'medium':
                    text = f"🔶 Advertencia de seguridad: {text}"
            
            if 'security_category' in security_context:
                category = security_context['security_category']
                text = f"[{category.upper()}] {text}"
            
            return text
            
        except Exception as e:
            logger.error(f"Error agregando contexto de seguridad: {e}")
            return text
    
    def _validate_security_content(self, content: Union[str, Dict]) -> Union[str, Dict]:
        """Valida que el contenido sea apropiado para ciberseguridad"""
        try:
            if isinstance(content, str):
                # Verificar que no contenga información sensible
                sensitive_patterns = [
                    r'password\s*[:=]\s*\w+',
                    r'api[_-]?key\s*[:=]\s*\w+',
                    r'token\s*[:=]\s*\w+',
                    r'secret\s*[:=]\s*\w+'
                ]
                
                for pattern in sensitive_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        content = re.sub(pattern, '[INFORMACIÓN SENSIBLE OCULTA]', content, flags=re.IGNORECASE)
                
                return content
            else:
                return content
                
        except Exception as e:
            logger.error(f"Error validando contenido de seguridad: {e}")
            return content
    
    def get_enhancement_report(self) -> Dict[str, Any]:
        """Genera reporte de mejoras aplicadas"""
        return {
            'enhancement_metrics': self.enhancement_metrics,
            'quality_thresholds': self.quality_thresholds,
            'enhancement_strategies': self.enhancement_strategies,
            'recommendations': self._generate_enhancement_recommendations()
        }
    
    def _generate_enhancement_recommendations(self) -> List[str]:
        """Genera recomendaciones de mejora"""
        recommendations = []
        
        if self.enhancement_metrics['predictions_enhanced'] > 0:
            recommendations.append("Sistema de mejora de predicciones funcionando correctamente")
        
        if self.enhancement_metrics['quality_improvements'] > 0:
            recommendations.append("Mejoras de calidad aplicadas exitosamente")
        
        if self.enhancement_metrics['context_improvements'] > 0:
            recommendations.append("Mejoras de contexto implementadas")
        
        if self.enhancement_metrics['errors_corrected'] > 0:
            recommendations.append("Errores corregidos en predicciones")
        
        return recommendations
