"""
@celebro Core - Sistema de Interpretación de Respuestas de IAs
Versión: 0.6.0
Cerebro central que coordina todos los módulos de @celebro
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import os

from .response_analyzer import ResponseAnalyzer, ResponseAnalysis
from .context_processor import ContextProcessor, ContextualizedResponse
from .response_generator import ResponseGenerator, AlternativeResponse
from .knowledge_synthesizer import KnowledgeSynthesizer, SynthesizedKnowledge
from .tensorflow_integration import TensorFlowCelebroIntegration, ModelType, TrainingStatus

logger = logging.getLogger('Celebro_Core')

@dataclass
class CelebroSession:
    """Sesión de @celebro"""
    session_id: str
    start_time: datetime
    responses_processed: int
    knowledge_synthesized: int
    context_updates: int
    active: bool

class CelebroCore:
    """
    Núcleo central de @celebro.
    Coordina el análisis, procesamiento de contexto, generación de respuestas
    y síntesis de conocimiento de múltiples IAs.
    """
    
    def __init__(self):
        self.response_analyzer = ResponseAnalyzer()
        self.context_processor = ContextProcessor()
        self.response_generator = ResponseGenerator()
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.tensorflow_integration = TensorFlowCelebroIntegration()
        
        self.sessions = {}
        self.response_database = {}
        self.knowledge_database = {}
        self.context_database = {}
        
        # Configuración
        self.max_responses_per_session = 100
        self.knowledge_synthesis_threshold = 5
        self.context_update_interval = 30  # segundos
        
        logger.info("@celebro Core inicializado")
    
    async def initialize(self):
        """Inicializa @celebro"""
        try:
            # Crear directorios necesarios
            os.makedirs('celebro/data', exist_ok=True)
            os.makedirs('celebro/cache', exist_ok=True)
            os.makedirs('celebro/logs', exist_ok=True)
            
            logger.info("@celebro inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando @celebro: {e}")
            return False
    
    async def process_ai_response(self, response: str, source_ai: str, 
                                user_context: Dict[str, Any] = None,
                                session_id: str = None) -> Dict[str, Any]:
        """
        Procesa una respuesta de IA externa
        
        Args:
            response: Respuesta de la IA externa
            source_ai: Nombre de la IA que generó la respuesta
            user_context: Contexto del usuario
            session_id: ID de sesión (opcional)
        
        Returns:
            Resultado del procesamiento completo
        """
        try:
            # Crear o obtener sesión
            if not session_id:
                session_id = f"session_{int(datetime.now().timestamp())}"
            
            session = await self._get_or_create_session(session_id)
            
            # 1. Analizar respuesta
            logger.info(f"Analizando respuesta de {source_ai}")
            analysis = await self.response_analyzer.analyze_response(
                response, source_ai, user_context
            )
            
            # 2. Procesar contexto
            logger.info("Procesando contexto")
            contextualized = await self.context_processor.process_context(
                response, analysis, user_context
            )
            
            # 3. Generar respuestas alternativas
            logger.info("Generando respuestas alternativas")
            alternatives = await self.response_generator.generate_alternatives(
                response, analysis, contextualized, num_alternatives=3
            )
            
            # 4. Actualizar base de conocimiento
            await self._update_knowledge_base(analysis, contextualized, alternatives)
            
            # 5. Verificar si es necesario sintetizar conocimiento
            if len(self.response_database) >= self.knowledge_synthesis_threshold:
                logger.info("Sintetizando conocimiento acumulado")
                synthesized = await self._synthesize_accumulated_knowledge()
            else:
                synthesized = None
            
            # 6. Actualizar sesión
            await self._update_session(session, analysis, contextualized, alternatives)
            
            # 7. Preparar resultado
            result = {
                "session_id": session_id,
                "source_ai": source_ai,
                "analysis": {
                    "response_type": analysis.response_type.value,
                    "confidence": analysis.confidence,
                    "key_concepts": analysis.key_concepts,
                    "sentiment": analysis.sentiment,
                    "complexity": analysis.complexity,
                    "technical_level": analysis.technical_level,
                    "emotional_tone": analysis.emotional_tone,
                    "main_intent": analysis.main_intent
                },
                "contextualized_response": {
                    "original": contextualized.original_response,
                    "transformed": contextualized.transformed_response,
                    "confidence": contextualized.confidence,
                    "transformations_applied": contextualized.transformation_rules
                },
                "alternatives": [
                    {
                        "text": alt.alternative_text,
                        "style": alt.style.value,
                        "length": alt.length.value,
                        "confidence": alt.confidence,
                        "semantic_similarity": alt.semantic_similarity
                    }
                    for alt in alternatives
                ],
                "knowledge_synthesis": {
                    "topic": synthesized.topic if synthesized else None,
                    "method": synthesized.synthesis_method.value if synthesized else None,
                    "confidence": synthesized.overall_confidence if synthesized else None,
                    "coherence": synthesized.coherence_score if synthesized else None
                } if synthesized else None,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Respuesta procesada exitosamente: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando respuesta de IA: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def query_celebro(self, query: str, context: Dict[str, Any] = None,
                          session_id: str = None) -> Dict[str, Any]:
        """
        Consulta @celebro para obtener respuestas contextualizadas
        
        Args:
            query: Consulta del usuario
            context: Contexto adicional
            session_id: ID de sesión
        
        Returns:
            Respuesta contextualizada de @celebro
        """
        try:
            # Crear o obtener sesión
            if not session_id:
                session_id = f"query_{int(datetime.now().timestamp())}"
            
            session = await self._get_or_create_session(session_id)
            
            # Buscar conocimiento relevante
            relevant_knowledge = await self._find_relevant_knowledge(query, context)
            
            # Generar respuesta contextualizada
            response = await self._generate_contextualized_response(
                query, relevant_knowledge, context
            )
            
            # Actualizar sesión
            session.responses_processed += 1
            
            result = {
                "session_id": session_id,
                "query": query,
                "response": response,
                "knowledge_sources": len(relevant_knowledge),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Consulta procesada: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_or_create_session(self, session_id: str) -> CelebroSession:
        """Obtiene o crea una sesión"""
        if session_id not in self.sessions:
            self.sessions[session_id] = CelebroSession(
                session_id=session_id,
                start_time=datetime.now(),
                responses_processed=0,
                knowledge_synthesized=0,
                context_updates=0,
                active=True
            )
        
        return self.sessions[session_id]
    
    async def _update_knowledge_base(self, analysis: ResponseAnalysis, 
                                   contextualized: ContextualizedResponse,
                                   alternatives: List[AlternativeResponse]):
        """Actualiza la base de conocimiento"""
        try:
            # Guardar análisis
            self.response_database[analysis.response_id] = {
                "analysis": analysis,
                "contextualized": contextualized,
                "alternatives": alternatives,
                "timestamp": datetime.now()
            }
            
            # Guardar contexto
            self.context_database[contextualized.timestamp.isoformat()] = contextualized
            
            logger.debug(f"Base de conocimiento actualizada: {analysis.response_id}")
            
        except Exception as e:
            logger.error(f"Error actualizando base de conocimiento: {e}")
    
    async def _synthesize_accumulated_knowledge(self) -> Optional[SynthesizedKnowledge]:
        """Sintetiza conocimiento acumulado"""
        try:
            # Obtener respuestas recientes
            recent_responses = list(self.response_database.values())[-self.knowledge_synthesis_threshold:]
            
            # Extraer análisis
            analyses = [item["analysis"] for item in recent_responses]
            
            # Sintetizar conocimiento
            synthesized = await self.knowledge_synthesizer.synthesize_knowledge(analyses)
            
            # Guardar en base de conocimiento
            self.knowledge_database[synthesized.id] = synthesized
            
            logger.info(f"Conocimiento sintetizado: {synthesized.id}")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error sintetizando conocimiento: {e}")
            return None
    
    async def _update_session(self, session: CelebroSession, analysis: ResponseAnalysis,
                            contextualized: ContextualizedResponse,
                            alternatives: List[AlternativeResponse]):
        """Actualiza la sesión"""
        try:
            session.responses_processed += 1
            session.context_updates += 1
            
            # Verificar límites de sesión
            if session.responses_processed >= self.max_responses_per_session:
                session.active = False
                logger.info(f"Sesión {session.session_id} completada")
            
        except Exception as e:
            logger.error(f"Error actualizando sesión: {e}")
    
    async def _find_relevant_knowledge(self, query: str, context: Dict[str, Any]) -> List[Any]:
        """Encuentra conocimiento relevante para una consulta"""
        try:
            relevant_knowledge = []
            
            # Buscar en base de conocimiento sintetizado
            for knowledge_id, knowledge in self.knowledge_database.items():
                if self._is_knowledge_relevant(query, knowledge):
                    relevant_knowledge.append(knowledge)
            
            # Buscar en respuestas individuales
            for response_id, response_data in self.response_database.items():
                analysis = response_data["analysis"]
                if self._is_analysis_relevant(query, analysis):
                    relevant_knowledge.append(analysis)
            
            return relevant_knowledge
            
        except Exception as e:
            logger.error(f"Error encontrando conocimiento relevante: {e}")
            return []
    
    def _is_knowledge_relevant(self, query: str, knowledge: SynthesizedKnowledge) -> bool:
        """Verifica si el conocimiento es relevante para la consulta"""
        try:
            query_lower = query.lower()
            topic_lower = knowledge.topic.lower()
            
            # Relevancia basada en tema
            if topic_lower in query_lower or query_lower in topic_lower:
                return True
            
            # Relevancia basada en conceptos
            for chunk in knowledge.knowledge_chunks:
                if any(concept.lower() in query_lower for concept in chunk.content.split()):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verificando relevancia del conocimiento: {e}")
            return False
    
    def _is_analysis_relevant(self, query: str, analysis: ResponseAnalysis) -> bool:
        """Verifica si el análisis es relevante para la consulta"""
        try:
            query_lower = query.lower()
            
            # Relevancia basada en conceptos clave
            for concept in analysis.key_concepts:
                if concept.lower() in query_lower:
                    return True
            
            # Relevancia basada en contenido
            if any(word in query_lower for word in analysis.original_response.lower().split()):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verificando relevancia del análisis: {e}")
            return False
    
    async def _generate_contextualized_response(self, query: str, 
                                              relevant_knowledge: List[Any],
                                              context: Dict[str, Any]) -> str:
        """Genera una respuesta contextualizada"""
        try:
            if not relevant_knowledge:
                return "No tengo información relevante para responder tu consulta en este momento."
            
            # Seleccionar la mejor fuente de conocimiento
            best_knowledge = max(relevant_knowledge, key=lambda k: self._calculate_knowledge_score(k))
            
            # Generar respuesta basada en el conocimiento
            if hasattr(best_knowledge, 'knowledge_chunks'):  # Conocimiento sintetizado
                response = self._generate_from_synthesized_knowledge(best_knowledge, query)
            else:  # Análisis individual
                response = self._generate_from_analysis(best_knowledge, query)
            
            # Aplicar contextualización adicional
            if context:
                response = self._apply_contextualization(response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generando respuesta contextualizada: {e}")
            return "Lo siento, ocurrió un error al procesar tu consulta."
    
    def _calculate_knowledge_score(self, knowledge: Any) -> float:
        """Calcula la puntuación de un fragmento de conocimiento"""
        try:
            if hasattr(knowledge, 'overall_confidence'):  # Conocimiento sintetizado
                return knowledge.overall_confidence
            elif hasattr(knowledge, 'confidence'):  # Análisis individual
                return knowledge.confidence
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculando puntuación de conocimiento: {e}")
            return 0.5
    
    def _generate_from_synthesized_knowledge(self, knowledge: SynthesizedKnowledge, query: str) -> str:
        """Genera respuesta desde conocimiento sintetizado"""
        try:
            # Usar el primer fragmento de conocimiento como base
            if knowledge.knowledge_chunks:
                base_content = knowledge.knowledge_chunks[0].content
                
                # Personalizar según la consulta
                if "cómo" in query.lower():
                    return f"Para responder tu pregunta sobre cómo hacer algo: {base_content}"
                elif "qué" in query.lower():
                    return f"En respuesta a qué es: {base_content}"
                elif "por qué" in query.lower():
                    return f"La razón es: {base_content}"
                else:
                    return f"Basándome en el conocimiento disponible: {base_content}"
            else:
                return "No tengo información específica sobre este tema."
                
        except Exception as e:
            logger.error(f"Error generando desde conocimiento sintetizado: {e}")
            return "Error procesando conocimiento sintetizado."
    
    def _generate_from_analysis(self, analysis: ResponseAnalysis, query: str) -> str:
        """Genera respuesta desde análisis individual"""
        try:
            # Usar la respuesta original como base
            base_content = analysis.original_response
            
            # Personalizar según el tipo de respuesta
            if analysis.response_type.value == "technical":
                return f"Desde una perspectiva técnica: {base_content}"
            elif analysis.response_type.value == "creative":
                return f"De manera creativa: {base_content}"
            elif analysis.response_type.value == "analytical":
                return f"Analíticamente: {base_content}"
            else:
                return base_content
                
        except Exception as e:
            logger.error(f"Error generando desde análisis: {e}")
            return "Error procesando análisis individual."
    
    def _apply_contextualization(self, response: str, context: Dict[str, Any]) -> str:
        """Aplica contextualización adicional"""
        try:
            # Contextualización basada en el contexto del usuario
            if "time_of_day" in context:
                time = context["time_of_day"]
                if time in ["morning", "mañana"]:
                    response = f"Buenos días. {response}"
                elif time in ["evening", "tarde"]:
                    response = f"Buenas tardes. {response}"
                elif time in ["night", "noche"]:
                    response = f"Buenas noches. {response}"
            
            if "user_expertise" in context:
                expertise = context["user_expertise"]
                if expertise == "beginner":
                    response = f"Para explicarlo de manera simple: {response}"
                elif expertise == "expert":
                    response = f"Como experto, te explico: {response}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error aplicando contextualización: {e}")
            return response
    
    async def get_celebro_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de @celebro"""
        try:
            # Estadísticas de sesiones
            active_sessions = sum(1 for s in self.sessions.values() if s.active)
            total_responses = sum(s.responses_processed for s in self.sessions.values())
            
            # Estadísticas de módulos
            analyzer_stats = await self.response_analyzer.get_analysis_stats()
            context_stats = await self.context_processor.get_context_stats()
            generator_stats = await self.response_generator.get_generation_stats()
            synthesizer_stats = await self.knowledge_synthesizer.get_synthesis_stats()
            
            return {
                "sessions": {
                    "total": len(self.sessions),
                    "active": active_sessions,
                    "total_responses_processed": total_responses
                },
                "knowledge_base": {
                    "total_responses": len(self.response_database),
                    "synthesized_knowledge": len(self.knowledge_database),
                    "context_entries": len(self.context_database)
                },
                "modules": {
                    "analyzer": analyzer_stats,
                    "context_processor": context_stats,
                    "generator": generator_stats,
                    "synthesizer": synthesizer_stats
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    async def export_knowledge(self, format: str = "json") -> Dict[str, Any]:
        """Exporta el conocimiento de @celebro"""
        try:
            if format == "json":
                return {
                    "sessions": {sid: {
                        "session_id": s.session_id,
                        "start_time": s.start_time.isoformat(),
                        "responses_processed": s.responses_processed,
                        "knowledge_synthesized": s.knowledge_synthesized,
                        "context_updates": s.context_updates,
                        "active": s.active
                    } for sid, s in self.sessions.items()},
                    "knowledge_database": {kid: {
                        "id": k.id,
                        "topic": k.topic,
                        "synthesis_method": k.synthesis_method.value,
                        "overall_confidence": k.overall_confidence,
                        "coherence_score": k.coherence_score,
                        "completeness_score": k.completeness_score,
                        "timestamp": k.timestamp.isoformat()
                    } for kid, k in self.knowledge_database.items()},
                    "export_timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Formato no soportado: {format}"}
                
        except Exception as e:
            logger.error(f"Error exportando conocimiento: {e}")
            return {"error": str(e)}
    
    # ===== MÉTODOS DE TENSORFLOW =====
    
    def create_tensorflow_model(self, model_name: str, model_type: str, 
                               num_classes: int = None, 
                               security_categories: List[str] = None) -> str:
        """
        Crea un modelo de TensorFlow
        
        Args:
            model_name: Nombre del modelo
            model_type: Tipo de modelo (text_classification, sentiment_analysis, etc.)
            num_classes: Número de clases (para clasificación)
            security_categories: Categorías de seguridad (para análisis de seguridad)
            
        Returns:
            ID del modelo creado
        """
        try:
            model_type_enum = ModelType(model_type)
            
            if model_type_enum == ModelType.TEXT_CLASSIFICATION:
                if not num_classes:
                    raise ValueError("num_classes es requerido para clasificación de texto")
                return self.tensorflow_integration.create_text_classification_model(
                    model_name, num_classes
                )
            
            elif model_type_enum == ModelType.SENTIMENT_ANALYSIS:
                return self.tensorflow_integration.create_sentiment_analysis_model(model_name)
            
            elif model_type_enum == ModelType.RESPONSE_GENERATION:
                return self.tensorflow_integration.create_response_generation_model(model_name)
            
            elif model_type_enum == ModelType.SECURITY_ANALYSIS:
                if not security_categories:
                    security_categories = [
                        'authentication', 'encryption', 'malware', 'phishing',
                        'firewall', 'vulnerability', 'compliance', 'incident_response'
                    ]
                return self.tensorflow_integration.create_security_analysis_model(
                    model_name, security_categories
                )
            
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creando modelo TensorFlow: {e}")
            raise
    
    def train_tensorflow_model(self, model_id: str, training_data: List[str], 
                             labels: List[Union[int, str]],
                             validation_data: Tuple[List[str], List[Union[int, str]]] = None) -> Dict[str, Any]:
        """
        Entrena un modelo de TensorFlow
        
        Args:
            model_id: ID del modelo
            training_data: Datos de entrenamiento
            labels: Etiquetas de entrenamiento
            validation_data: Datos de validación (opcional)
            
        Returns:
            Métricas de entrenamiento
        """
        try:
            metrics = self.tensorflow_integration.train_model(
                model_id, training_data, labels, validation_data
            )
            
            return {
                'model_id': model_id,
                'accuracy': metrics.accuracy,
                'loss': metrics.loss,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'training_time': metrics.training_time,
                'validation_accuracy': metrics.validation_accuracy,
                'validation_loss': metrics.validation_loss,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error entrenando modelo TensorFlow: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def predict_with_tensorflow(self, model_id: str, text: str) -> Dict[str, Any]:
        """
        Realiza predicción con un modelo de TensorFlow
        
        Args:
            model_id: ID del modelo
            text: Texto a analizar
            
        Returns:
            Resultado de la predicción
        """
        try:
            result = self.tensorflow_integration.predict(model_id, text)
            result['model_id'] = model_id
            result['input_text'] = text
            return result
            
        except Exception as e:
            logger.error(f"Error en predicción TensorFlow: {e}")
            return {'error': str(e)}
    
    def get_tensorflow_models(self) -> List[Dict[str, Any]]:
        """Obtiene lista de modelos TensorFlow disponibles"""
        try:
            return self.tensorflow_integration.list_models()
        except Exception as e:
            logger.error(f"Error obteniendo modelos TensorFlow: {e}")
            return []
    
    def get_tensorflow_model_info(self, model_id: str) -> Dict[str, Any]:
        """Obtiene información de un modelo TensorFlow"""
        try:
            return self.tensorflow_integration.get_model_info(model_id)
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
            return {'error': str(e)}
    
    def delete_tensorflow_model(self, model_id: str) -> bool:
        """Elimina un modelo TensorFlow"""
        try:
            return self.tensorflow_integration.delete_model(model_id)
        except Exception as e:
            logger.error(f"Error eliminando modelo TensorFlow: {e}")
            return False
    
    def analyze_response_with_ai(self, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analiza una respuesta usando modelos de IA (TensorFlow + @celebro)
        
        Args:
            response: Respuesta a analizar
            context: Contexto adicional
            
        Returns:
            Análisis completo de la respuesta
        """
        try:
            # Análisis tradicional con @celebro
            traditional_analysis = self.analyze_response(response, context)
            
            # Análisis con TensorFlow si hay modelos disponibles
            ai_analysis = {}
            models = self.get_tensorflow_models()
            
            for model_info in models:
                model_id = model_info['model_id']
                model_type = model_info.get('model_type', '')
                
                try:
                    prediction = self.predict_with_tensorflow(model_id, response)
                    ai_analysis[model_type] = prediction
                except Exception as e:
                    logger.warning(f"Error analizando con modelo {model_id}: {e}")
            
            # Combinar análisis
            combined_analysis = {
                'traditional_analysis': traditional_analysis,
                'ai_analysis': ai_analysis,
                'analysis_timestamp': datetime.now().isoformat(),
                'models_used': len(ai_analysis)
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error en análisis combinado: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema @celebro"""
        try:
            return {
                'active_sessions': len(self.sessions),
                'total_responses': len(self.response_database),
                'total_knowledge': len(self.knowledge_database),
                'total_contexts': len(self.context_database),
                'tensorflow_models': len(self.tensorflow_integration.models),
                'tensorflow_status': self.tensorflow_integration.training_status.value,
                'components_status': {
                    'response_analyzer': getattr(self.response_analyzer, 'is_initialized', True),
                    'context_processor': getattr(self.context_processor, 'is_initialized', True),
                    'response_generator': getattr(self.response_generator, 'is_initialized', True),
                    'knowledge_synthesizer': getattr(self.knowledge_synthesizer, 'is_initialized', True),
                    'tensorflow_integration': True
                },
                'tensorflow_models_info': self.get_tensorflow_models()
            }
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {'error': str(e)}
