"""
Analizador de Respuestas de IAs
Versión: 0.6.0
Analiza respuestas de IAs externas y extrae información semántica
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Celebro_Analyzer')

class ResponseType(Enum):
    """Tipos de respuestas identificadas"""
    FACTUAL = "factual"
    INSTRUCTIVE = "instructive"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"
    TECHNICAL = "technical"
    PHILOSOPHICAL = "philosophical"

class ConfidenceLevel(Enum):
    """Niveles de confianza en el análisis"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class ResponseAnalysis:
    """Análisis de una respuesta de IA"""
    response_id: str
    original_response: str
    response_type: ResponseType
    confidence: float
    key_concepts: List[str]
    sentiment: float
    complexity: float
    language: str
    technical_level: int
    emotional_tone: str
    main_intent: str
    extracted_facts: List[str]
    suggestions: List[str]
    timestamp: datetime

class ResponseAnalyzer:
    """
    Analizador de respuestas de IAs externas.
    Extrae información semántica y contextual de las respuestas.
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.patterns = self._initialize_patterns()
        self.concept_database = self._initialize_concept_database()
        
        logger.info("Analizador de respuestas inicializado")
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Inicializa patrones de análisis"""
        return {
            "factual_indicators": [
                r"según.*datos", r"estadísticamente", r"investigaciones muestran",
                r"estudios indican", r"la evidencia", r"científicamente"
            ],
            "instructive_indicators": [
                r"para.*hacer", r"pasos.*siguientes", r"debes", r"necesitas",
                r"instrucciones", r"procedimiento", r"cómo.*hacer"
            ],
            "creative_indicators": [
                r"imagina", r"crea", r"diseña", r"inventa", r"innovador",
                r"original", r"único", r"artístico"
            ],
            "analytical_indicators": [
                r"analicemos", r"examinemos", r"evaluemos", r"comparemos",
                r"pros.*contras", r"ventajas.*desventajas", r"análisis"
            ],
            "emotional_indicators": [
                r"me siento", r"emocionalmente", r"sentimientos", r"corazón",
                r"alma", r"pasional", r"inspirador"
            ],
            "technical_indicators": [
                r"algoritmo", r"código", r"programación", r"técnicamente",
                r"implementación", r"arquitectura", r"protocolo"
            ],
            "philosophical_indicators": [
                r"filosofía", r"ética", r"moral", r"existencia", r"significado",
                r"propósito", r"verdad", r"realidad"
            ]
        }
    
    def _initialize_concept_database(self) -> Dict[str, List[str]]:
        """Inicializa base de datos de conceptos"""
        return {
            "tecnologia": ["programación", "algoritmos", "IA", "machine learning", "datos"],
            "ciencia": ["investigación", "experimento", "hipótesis", "teoría", "método científico"],
            "arte": ["creatividad", "expresión", "estética", "diseño", "composición"],
            "filosofia": ["ética", "moral", "existencia", "conocimiento", "verdad"],
            "educacion": ["aprendizaje", "enseñanza", "conocimiento", "habilidades", "desarrollo"],
            "negocios": ["estrategia", "mercado", "competencia", "innovación", "liderazgo"]
        }
    
    async def analyze_response(self, response: str, source_ai: str = "unknown", 
                             context: Dict[str, Any] = None) -> ResponseAnalysis:
        """
        Analiza una respuesta de IA externa
        
        Args:
            response: Respuesta a analizar
            source_ai: IA que generó la respuesta
            context: Contexto adicional
        
        Returns:
            Análisis completo de la respuesta
        """
        try:
            # Generar ID único
            response_id = self._generate_response_id(response, source_ai)
            
            # Verificar caché
            if response_id in self.analysis_cache:
                logger.debug(f"Análisis encontrado en caché: {response_id}")
                return self.analysis_cache[response_id]
            
            # Análisis básico
            response_type = self._classify_response_type(response)
            confidence = self._calculate_confidence(response, response_type)
            key_concepts = self._extract_key_concepts(response)
            sentiment = self._analyze_sentiment(response)
            complexity = self._calculate_complexity(response)
            language = self._detect_language(response)
            technical_level = self._assess_technical_level(response)
            emotional_tone = self._analyze_emotional_tone(response)
            main_intent = self._identify_main_intent(response)
            extracted_facts = self._extract_facts(response)
            suggestions = self._generate_suggestions(response, response_type)
            
            # Crear análisis
            analysis = ResponseAnalysis(
                response_id=response_id,
                original_response=response,
                response_type=response_type,
                confidence=confidence,
                key_concepts=key_concepts,
                sentiment=sentiment,
                complexity=complexity,
                language=language,
                technical_level=technical_level,
                emotional_tone=emotional_tone,
                main_intent=main_intent,
                extracted_facts=extracted_facts,
                suggestions=suggestions,
                timestamp=datetime.now()
            )
            
            # Guardar en caché
            self.analysis_cache[response_id] = analysis
            
            logger.info(f"Respuesta analizada: {response_id} - Tipo: {response_type.value}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando respuesta: {e}")
            # Retornar análisis básico en caso de error
            return self._create_basic_analysis(response, source_ai)
    
    def _generate_response_id(self, response: str, source_ai: str) -> str:
        """Genera ID único para la respuesta"""
        content_hash = hashlib.md5(response.encode()).hexdigest()[:8]
        timestamp = int(datetime.now().timestamp())
        return f"resp_{source_ai}_{content_hash}_{timestamp}"
    
    def _classify_response_type(self, response: str) -> ResponseType:
        """Clasifica el tipo de respuesta"""
        response_lower = response.lower()
        
        # Contar indicadores por tipo
        type_scores = {}
        for response_type, indicators in self.patterns.items():
            score = 0
            for pattern in indicators:
                matches = len(re.findall(pattern, response_lower))
                score += matches
            type_scores[response_type] = score
        
        # Determinar tipo principal
        if type_scores.get("factual_indicators", 0) > 0:
            return ResponseType.FACTUAL
        elif type_scores.get("instructive_indicators", 0) > 0:
            return ResponseType.INSTRUCTIVE
        elif type_scores.get("creative_indicators", 0) > 0:
            return ResponseType.CREATIVE
        elif type_scores.get("analytical_indicators", 0) > 0:
            return ResponseType.ANALYTICAL
        elif type_scores.get("emotional_indicators", 0) > 0:
            return ResponseType.EMOTIONAL
        elif type_scores.get("technical_indicators", 0) > 0:
            return ResponseType.TECHNICAL
        elif type_scores.get("philosophical_indicators", 0) > 0:
            return ResponseType.PHILOSOPHICAL
        else:
            return ResponseType.FACTUAL  # Por defecto
    
    def _calculate_confidence(self, response: str, response_type: ResponseType) -> float:
        """Calcula la confianza en el análisis"""
        try:
            # Factores de confianza
            length_factor = min(1.0, len(response) / 500)  # Respuestas más largas = más confianza
            structure_factor = self._assess_structure_quality(response)
            clarity_factor = self._assess_clarity(response)
            
            # Ponderación
            confidence = (length_factor * 0.3 + structure_factor * 0.4 + clarity_factor * 0.3)
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def _extract_key_concepts(self, response: str) -> List[str]:
        """Extrae conceptos clave de la respuesta"""
        try:
            concepts = []
            response_lower = response.lower()
            
            # Buscar conceptos en la base de datos
            for category, concept_list in self.concept_database.items():
                for concept in concept_list:
                    if concept.lower() in response_lower:
                        concepts.append(concept)
            
            # Extraer palabras importantes (nombres propios, términos técnicos)
            important_words = re.findall(r'\b[A-Z][a-z]+\b', response)
            concepts.extend(important_words)
            
            # Extraer términos técnicos
            technical_terms = re.findall(r'\b\w*[a-z]{3,}\w*\b', response)
            concepts.extend([term for term in technical_terms if len(term) > 4])
            
            # Eliminar duplicados y limitar
            unique_concepts = list(set(concepts))[:10]
            
            return unique_concepts
            
        except Exception as e:
            logger.error(f"Error extrayendo conceptos: {e}")
            return []
    
    def _analyze_sentiment(self, response: str) -> float:
        """Analiza el sentimiento de la respuesta"""
        try:
            positive_words = [
                "excelente", "bueno", "genial", "fantástico", "perfecto", "increíble",
                "maravilloso", "brillante", "exitoso", "positivo", "optimista"
            ]
            negative_words = [
                "malo", "terrible", "horrible", "pésimo", "negativo", "pesimista",
                "problemático", "difícil", "complicado", "frustrante"
            ]
            
            response_lower = response.lower()
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}")
            return 0.0
    
    def _calculate_complexity(self, response: str) -> float:
        """Calcula la complejidad de la respuesta"""
        try:
            words = response.split()
            sentences = re.split(r'[.!?]+', response)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Factores de complejidad
            avg_words_per_sentence = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Normalizar
            complexity = (avg_words_per_sentence * 0.1 + avg_word_length * 0.1) / 2
            return min(1.0, max(0.0, complexity))
            
        except Exception as e:
            logger.error(f"Error calculando complejidad: {e}")
            return 0.5
    
    def _detect_language(self, response: str) -> str:
        """Detecta el idioma de la respuesta"""
        try:
            spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"]
            english_words = ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"]
            
            response_lower = response.lower()
            spanish_count = sum(1 for word in spanish_words if word in response_lower)
            english_count = sum(1 for word in english_words if word in response_lower)
            
            if spanish_count > english_count:
                return "spanish"
            elif english_count > spanish_count:
                return "english"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error detectando idioma: {e}")
            return "unknown"
    
    def _assess_technical_level(self, response: str) -> int:
        """Evalúa el nivel técnico de la respuesta (1-5)"""
        try:
            technical_terms = [
                "algoritmo", "programación", "código", "implementación", "arquitectura",
                "protocolo", "API", "base de datos", "framework", "biblioteca"
            ]
            
            response_lower = response.lower()
            technical_count = sum(1 for term in technical_terms if term in response_lower)
            
            if technical_count >= 5:
                return 5
            elif technical_count >= 3:
                return 4
            elif technical_count >= 2:
                return 3
            elif technical_count >= 1:
                return 2
            else:
                return 1
                
        except Exception as e:
            logger.error(f"Error evaluando nivel técnico: {e}")
            return 1
    
    def _analyze_emotional_tone(self, response: str) -> str:
        """Analiza el tono emocional de la respuesta"""
        try:
            response_lower = response.lower()
            
            if any(word in response_lower for word in ["!", "¡", "increíble", "fantástico"]):
                return "enthusiastic"
            elif any(word in response_lower for word in ["serio", "importante", "crítico"]):
                return "serious"
            elif any(word in response_lower for word in ["amigable", "cordial", "cálido"]):
                return "friendly"
            elif any(word in response_lower for word in ["técnico", "específico", "detallado"]):
                return "technical"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error analizando tono emocional: {e}")
            return "neutral"
    
    def _identify_main_intent(self, response: str) -> str:
        """Identifica la intención principal de la respuesta"""
        try:
            response_lower = response.lower()
            
            if any(word in response_lower for word in ["cómo", "pasos", "procedimiento"]):
                return "instruct"
            elif any(word in response_lower for word in ["qué", "cuál", "definir"]):
                return "explain"
            elif any(word in response_lower for word in ["por qué", "razón", "causa"]):
                return "analyze"
            elif any(word in response_lower for word in ["crea", "diseña", "inventa"]):
                return "create"
            else:
                return "inform"
                
        except Exception as e:
            logger.error(f"Error identificando intención: {e}")
            return "inform"
    
    def _extract_facts(self, response: str) -> List[str]:
        """Extrae hechos de la respuesta"""
        try:
            facts = []
            
            # Buscar patrones de hechos
            fact_patterns = [
                r"(\d+%|\d+\.\d+%)\s+de",
                r"según.*estudios?",
                r"la investigación.*muestra",
                r"estadísticamente.*hablando"
            ]
            
            for pattern in fact_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                facts.extend(matches)
            
            return facts[:5]  # Limitar a 5 hechos
            
        except Exception as e:
            logger.error(f"Error extrayendo hechos: {e}")
            return []
    
    def _generate_suggestions(self, response: str, response_type: ResponseType) -> List[str]:
        """Genera sugerencias para mejorar la respuesta"""
        try:
            suggestions = []
            
            if response_type == ResponseType.TECHNICAL:
                suggestions.append("Agregar ejemplos prácticos")
                suggestions.append("Simplificar terminología técnica")
            elif response_type == ResponseType.CREATIVE:
                suggestions.append("Incluir más detalles específicos")
                suggestions.append("Agregar contexto práctico")
            elif response_type == ResponseType.ANALYTICAL:
                suggestions.append("Incluir conclusiones claras")
                suggestions.append("Agregar recomendaciones")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generando sugerencias: {e}")
            return []
    
    def _assess_structure_quality(self, response: str) -> float:
        """Evalúa la calidad estructural de la respuesta"""
        try:
            # Factores estructurales
            has_paragraphs = len(response.split('\n\n')) > 1
            has_punctuation = any(p in response for p in ['.', '!', '?'])
            has_capitalization = any(c.isupper() for c in response)
            
            score = 0
            if has_paragraphs:
                score += 0.3
            if has_punctuation:
                score += 0.4
            if has_capitalization:
                score += 0.3
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluando estructura: {e}")
            return 0.5
    
    def _assess_clarity(self, response: str) -> float:
        """Evalúa la claridad de la respuesta"""
        try:
            # Factores de claridad
            sentence_length = len(response.split('.'))
            word_length = len(response.split())
            
            if sentence_length == 0 or word_length == 0:
                return 0.5
            
            avg_words_per_sentence = word_length / sentence_length
            
            # Claridad basada en longitud promedio de oraciones
            if 10 <= avg_words_per_sentence <= 20:
                return 1.0
            elif 5 <= avg_words_per_sentence <= 30:
                return 0.7
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Error evaluando claridad: {e}")
            return 0.5
    
    def _create_basic_analysis(self, response: str, source_ai: str) -> ResponseAnalysis:
        """Crea un análisis básico en caso de error"""
        return ResponseAnalysis(
            response_id=f"basic_{int(datetime.now().timestamp())}",
            original_response=response,
            response_type=ResponseType.FACTUAL,
            confidence=0.3,
            key_concepts=[],
            sentiment=0.0,
            complexity=0.5,
            language="unknown",
            technical_level=1,
            emotional_tone="neutral",
            main_intent="inform",
            extracted_facts=[],
            suggestions=[],
            timestamp=datetime.now()
        )
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del analizador"""
        return {
            "total_analyses": len(self.analysis_cache),
            "response_types": {
                rt.value: sum(1 for a in self.analysis_cache.values() if a.response_type == rt)
                for rt in ResponseType
            },
            "avg_confidence": sum(a.confidence for a in self.analysis_cache.values()) / max(len(self.analysis_cache), 1),
            "languages_detected": list(set(a.language for a in self.analysis_cache.values()))
        }
