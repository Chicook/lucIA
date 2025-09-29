"""
Procesador de Contexto
Versión: 0.6.0
Procesa y gestiona el contexto para la interpretación de respuestas
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Celebro_Context')

class ContextType(Enum):
    """Tipos de contexto"""
    CONVERSATION = "conversation"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    CULTURAL = "cultural"
    DOMAIN_SPECIFIC = "domain_specific"

class ContextPriority(Enum):
    """Prioridades de contexto"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ContextElement:
    """Elemento de contexto"""
    id: str
    context_type: ContextType
    content: Any
    priority: ContextPriority
    relevance_score: float
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]

@dataclass
class ContextualizedResponse:
    """Respuesta contextualizada"""
    original_response: str
    context_elements: List[ContextElement]
    transformed_response: str
    transformation_rules: List[str]
    confidence: float
    timestamp: datetime

class ContextProcessor:
    """
    Procesador de contexto para @celebro.
    Gestiona y procesa el contexto para la interpretación de respuestas.
    """
    
    def __init__(self):
        self.context_database = {}
        self.context_rules = self._initialize_context_rules()
        self.transformation_templates = self._initialize_transformation_templates()
        self.context_history = []
        
        logger.info("Procesador de contexto inicializado")
    
    def _initialize_context_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Inicializa reglas de contexto"""
        return {
            "technical": [
                {
                    "condition": "technical_level >= 4",
                    "transformation": "add_technical_context",
                    "priority": ContextPriority.HIGH
                },
                {
                    "condition": "programming_terms in response",
                    "transformation": "simplify_technical_language",
                    "priority": ContextPriority.MEDIUM
                }
            ],
            "emotional": [
                {
                    "condition": "sentiment < -0.3",
                    "transformation": "add_emotional_support",
                    "priority": ContextPriority.HIGH
                },
                {
                    "condition": "emotional_tone == 'enthusiastic'",
                    "transformation": "maintain_enthusiasm",
                    "priority": ContextPriority.MEDIUM
                }
            ],
            "cultural": [
                {
                    "condition": "language == 'spanish'",
                    "transformation": "adapt_to_spanish_culture",
                    "priority": ContextPriority.MEDIUM
                },
                {
                    "condition": "cultural_references in response",
                    "transformation": "explain_cultural_context",
                    "priority": ContextPriority.LOW
                }
            ],
            "temporal": [
                {
                    "condition": "current_time.hour < 6 or current_time.hour > 22",
                    "transformation": "adapt_to_time_context",
                    "priority": ContextPriority.LOW
                }
            ]
        }
    
    def _initialize_transformation_templates(self) -> Dict[str, str]:
        """Inicializa plantillas de transformación"""
        return {
            "add_technical_context": "Desde una perspectiva técnica, {response}",
            "simplify_technical_language": "En términos más simples, {response}",
            "add_emotional_support": "Entiendo que esto puede ser desafiante. {response}",
            "maintain_enthusiasm": "¡Excelente! {response}",
            "adapt_to_spanish_culture": "En el contexto cultural hispanohablante, {response}",
            "explain_cultural_context": "Para proporcionar contexto cultural, {response}",
            "adapt_to_time_context": "Considerando la hora actual, {response}",
            "add_practical_example": "{response} Por ejemplo, {example}",
            "add_visual_context": "Para visualizar mejor esto, {response}",
            "add_step_by_step": "Paso a paso: {response}"
        }
    
    async def process_context(self, response: str, analysis: Any, 
                            user_context: Dict[str, Any] = None) -> ContextualizedResponse:
        """
        Procesa el contexto para una respuesta
        
        Args:
            response: Respuesta original
            analysis: Análisis de la respuesta
            user_context: Contexto del usuario
        
        Returns:
            Respuesta contextualizada
        """
        try:
            # Recopilar elementos de contexto
            context_elements = await self._collect_context_elements(response, analysis, user_context)
            
            # Aplicar reglas de contexto
            applicable_rules = self._find_applicable_rules(response, analysis, context_elements)
            
            # Transformar respuesta
            transformed_response, transformation_rules = await self._transform_response(
                response, analysis, context_elements, applicable_rules
            )
            
            # Calcular confianza
            confidence = self._calculate_transformation_confidence(
                context_elements, applicable_rules
            )
            
            # Crear respuesta contextualizada
            contextualized = ContextualizedResponse(
                original_response=response,
                context_elements=context_elements,
                transformed_response=transformed_response,
                transformation_rules=transformation_rules,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Guardar en historial
            self.context_history.append(contextualized)
            
            logger.info(f"Respuesta contextualizada: {len(transformation_rules)} transformaciones aplicadas")
            return contextualized
            
        except Exception as e:
            logger.error(f"Error procesando contexto: {e}")
            return self._create_fallback_response(response)
    
    async def _collect_context_elements(self, response: str, analysis: Any, 
                                      user_context: Dict[str, Any]) -> List[ContextElement]:
        """Recopila elementos de contexto relevantes"""
        try:
            context_elements = []
            
            # Contexto técnico
            if analysis.technical_level >= 3:
                tech_context = ContextElement(
                    id=f"tech_{int(datetime.now().timestamp())}",
                    context_type=ContextType.TECHNICAL,
                    content={
                        "technical_level": analysis.technical_level,
                        "concepts": analysis.key_concepts,
                        "complexity": analysis.complexity
                    },
                    priority=ContextPriority.HIGH,
                    relevance_score=0.8,
                    timestamp=datetime.now(),
                    source="analysis",
                    metadata={"response_id": analysis.response_id}
                )
                context_elements.append(tech_context)
            
            # Contexto emocional
            if abs(analysis.sentiment) > 0.3:
                emotional_context = ContextElement(
                    id=f"emo_{int(datetime.now().timestamp())}",
                    context_type=ContextType.EMOTIONAL,
                    content={
                        "sentiment": analysis.sentiment,
                        "emotional_tone": analysis.emotional_tone,
                        "intent": analysis.main_intent
                    },
                    priority=ContextPriority.MEDIUM,
                    relevance_score=0.6,
                    timestamp=datetime.now(),
                    source="analysis",
                    metadata={"response_id": analysis.response_id}
                )
                context_elements.append(emotional_context)
            
            # Contexto temporal
            current_time = datetime.now()
            temporal_context = ContextElement(
                id=f"temp_{int(datetime.now().timestamp())}",
                context_type=ContextType.TEMPORAL,
                content={
                    "hour": current_time.hour,
                    "day_of_week": current_time.weekday(),
                    "is_weekend": current_time.weekday() >= 5
                },
                priority=ContextPriority.LOW,
                relevance_score=0.3,
                timestamp=current_time,
                source="system",
                metadata={}
            )
            context_elements.append(temporal_context)
            
            # Contexto cultural/idioma
            if analysis.language != "unknown":
                cultural_context = ContextElement(
                    id=f"cult_{int(datetime.now().timestamp())}",
                    context_type=ContextType.CULTURAL,
                    content={
                        "language": analysis.language,
                        "cultural_indicators": self._detect_cultural_indicators(response)
                    },
                    priority=ContextPriority.MEDIUM,
                    relevance_score=0.5,
                    timestamp=datetime.now(),
                    source="analysis",
                    metadata={"response_id": analysis.response_id}
                )
                context_elements.append(cultural_context)
            
            # Contexto del usuario (si está disponible)
            if user_context:
                user_context_element = ContextElement(
                    id=f"user_{int(datetime.now().timestamp())}",
                    context_type=ContextType.CONVERSATION,
                    content=user_context,
                    priority=ContextPriority.HIGH,
                    relevance_score=0.9,
                    timestamp=datetime.now(),
                    source="user",
                    metadata={}
                )
                context_elements.append(user_context_element)
            
            return context_elements
            
        except Exception as e:
            logger.error(f"Error recopilando contexto: {e}")
            return []
    
    def _detect_cultural_indicators(self, response: str) -> List[str]:
        """Detecta indicadores culturales en la respuesta"""
        try:
            cultural_indicators = []
            response_lower = response.lower()
            
            # Indicadores culturales hispanos
            if any(word in response_lower for word in ["gracias", "por favor", "disculpe"]):
                cultural_indicators.append("formal_politeness")
            
            if any(word in response_lower for word in ["familia", "comunidad", "colectivo"]):
                cultural_indicators.append("collectivist_values")
            
            if any(word in response_lower for word in ["tradición", "historia", "cultura"]):
                cultural_indicators.append("traditional_values")
            
            return cultural_indicators
            
        except Exception as e:
            logger.error(f"Error detectando indicadores culturales: {e}")
            return []
    
    def _find_applicable_rules(self, response: str, analysis: Any, 
                             context_elements: List[ContextElement]) -> List[Dict[str, Any]]:
        """Encuentra reglas aplicables basadas en el contexto"""
        try:
            applicable_rules = []
            
            for context_type, rules in self.context_rules.items():
                for rule in rules:
                    if self._evaluate_rule_condition(rule["condition"], response, analysis, context_elements):
                        applicable_rules.append(rule)
            
            # Ordenar por prioridad
            applicable_rules.sort(key=lambda x: x["priority"].value, reverse=True)
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error encontrando reglas aplicables: {e}")
            return []
    
    def _evaluate_rule_condition(self, condition: str, response: str, analysis: Any, 
                               context_elements: List[ContextElement]) -> bool:
        """Evalúa si una condición de regla se cumple"""
        try:
            # Evaluar condiciones simples
            if "technical_level >= 4" in condition:
                return analysis.technical_level >= 4
            
            if "sentiment < -0.3" in condition:
                return analysis.sentiment < -0.3
            
            if "language == 'spanish'" in condition:
                return analysis.language == "spanish"
            
            if "emotional_tone == 'enthusiastic'" in condition:
                return analysis.emotional_tone == "enthusiastic"
            
            if "current_time.hour < 6 or current_time.hour > 22" in condition:
                current_hour = datetime.now().hour
                return current_hour < 6 or current_hour > 22
            
            # Condiciones más complejas
            if "programming_terms in response" in condition:
                programming_terms = ["código", "programación", "algoritmo", "función", "variable"]
                return any(term in response.lower() for term in programming_terms)
            
            if "cultural_references in response" in condition:
                cultural_refs = ["tradición", "cultura", "historia", "sociedad"]
                return any(ref in response.lower() for ref in cultural_refs)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluando condición: {e}")
            return False
    
    async def _transform_response(self, response: str, analysis: Any, 
                                context_elements: List[ContextElement], 
                                applicable_rules: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Transforma la respuesta basada en las reglas aplicables"""
        try:
            transformed_response = response
            transformation_rules = []
            
            for rule in applicable_rules[:3]:  # Aplicar máximo 3 transformaciones
                transformation_type = rule["transformation"]
                
                if transformation_type in self.transformation_templates:
                    template = self.transformation_templates[transformation_type]
                    
                    # Aplicar transformación
                    if "{response}" in template:
                        transformed_response = template.format(response=transformed_response)
                    else:
                        transformed_response = f"{template} {transformed_response}"
                    
                    transformation_rules.append(transformation_type)
            
            # Aplicar transformaciones adicionales basadas en el contexto
            transformed_response = await self._apply_contextual_transformations(
                transformed_response, context_elements
            )
            
            return transformed_response, transformation_rules
            
        except Exception as e:
            logger.error(f"Error transformando respuesta: {e}")
            return response, []
    
    async def _apply_contextual_transformations(self, response: str, 
                                              context_elements: List[ContextElement]) -> str:
        """Aplica transformaciones contextuales adicionales"""
        try:
            transformed = response
            
            # Transformación basada en complejidad
            complexity_context = next(
                (ce for ce in context_elements if ce.context_type == ContextType.TECHNICAL), 
                None
            )
            
            if complexity_context and complexity_context.content.get("complexity", 0) > 0.7:
                # Simplificar respuesta compleja
                transformed = f"Para explicar esto de manera más clara: {transformed}"
            
            # Transformación basada en sentimiento
            emotional_context = next(
                (ce for ce in context_elements if ce.context_type == ContextType.EMOTIONAL), 
                None
            )
            
            if emotional_context and emotional_context.content.get("sentiment", 0) < -0.2:
                # Agregar apoyo emocional
                transformed = f"Entiendo que esto puede ser desafiante. {transformed}"
            
            # Transformación basada en tiempo
            temporal_context = next(
                (ce for ce in context_elements if ce.context_type == ContextType.TEMPORAL), 
                None
            )
            
            if temporal_context:
                hour = temporal_context.content.get("hour", 12)
                if hour < 6 or hour > 22:
                    transformed = f"Considerando la hora, {transformed.lower()}"
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error aplicando transformaciones contextuales: {e}")
            return response
    
    def _calculate_transformation_confidence(self, context_elements: List[ContextElement], 
                                           applicable_rules: List[Dict[str, Any]]) -> float:
        """Calcula la confianza en las transformaciones aplicadas"""
        try:
            if not context_elements or not applicable_rules:
                return 0.5
            
            # Confianza basada en relevancia del contexto
            avg_relevance = sum(ce.relevance_score for ce in context_elements) / len(context_elements)
            
            # Confianza basada en prioridad de reglas
            avg_priority = sum(rule["priority"].value for rule in applicable_rules) / len(applicable_rules)
            priority_factor = avg_priority / 4.0  # Normalizar a 0-1
            
            # Confianza combinada
            confidence = (avg_relevance * 0.6 + priority_factor * 0.4)
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def _create_fallback_response(self, response: str) -> ContextualizedResponse:
        """Crea una respuesta de respaldo en caso de error"""
        return ContextualizedResponse(
            original_response=response,
            context_elements=[],
            transformed_response=response,
            transformation_rules=[],
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    async def get_context_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del procesador de contexto"""
        return {
            "total_contextualizations": len(self.context_history),
            "context_types_used": {
                ct.value: sum(1 for ch in self.context_history 
                             for ce in ch.context_elements if ce.context_type == ct)
                for ct in ContextType
            },
            "avg_confidence": sum(ch.confidence for ch in self.context_history) / max(len(self.context_history), 1),
            "most_used_transformations": self._get_most_used_transformations()
        }
    
    def _get_most_used_transformations(self) -> List[Tuple[str, int]]:
        """Obtiene las transformaciones más utilizadas"""
        try:
            transformation_counts = {}
            
            for contextualized in self.context_history:
                for rule in contextualized.transformation_rules:
                    transformation_counts[rule] = transformation_counts.get(rule, 0) + 1
            
            return sorted(transformation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
        except Exception as e:
            logger.error(f"Error obteniendo transformaciones más usadas: {e}")
            return []
