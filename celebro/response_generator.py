"""
Generador de Respuestas Alternativas
Versión: 0.6.0
Genera respuestas alternativas con el mismo significado
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger('Celebro_Generator')

class ResponseStyle(Enum):
    """Estilos de respuesta"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"
    CREATIVE = "creative"
    CONVERSATIONAL = "conversational"

class ResponseLength(Enum):
    """Longitudes de respuesta"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

@dataclass
class AlternativeResponse:
    """Respuesta alternativa generada"""
    id: str
    original_response: str
    alternative_text: str
    style: ResponseStyle
    length: ResponseLength
    confidence: float
    transformation_applied: str
    semantic_similarity: float
    timestamp: datetime

class ResponseGenerator:
    """
    Generador de respuestas alternativas para @celebro.
    Crea variaciones de respuestas manteniendo el significado semántico.
    """
    
    def __init__(self):
        self.generation_templates = self._initialize_templates()
        self.synonym_database = self._initialize_synonyms()
        self.style_patterns = self._initialize_style_patterns()
        self.generated_responses = []
        
        logger.info("Generador de respuestas inicializado")
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Inicializa plantillas de generación"""
        return {
            "formal": [
                "Desde una perspectiva académica, {content}",
                "Según los estándares establecidos, {content}",
                "Basándome en la evidencia disponible, {content}",
                "De acuerdo con la literatura especializada, {content}"
            ],
            "casual": [
                "Bueno, {content}",
                "Mira, {content}",
                "Te explico, {content}",
                "Pues resulta que {content}"
            ],
            "technical": [
                "Técnicamente hablando, {content}",
                "Desde el punto de vista técnico, {content}",
                "En términos de implementación, {content}",
                "A nivel de arquitectura, {content}"
            ],
            "simplified": [
                "En palabras simples, {content}",
                "Para que sea más claro, {content}",
                "Básicamente, {content}",
                "Lo que quiero decir es que {content}"
            ],
            "creative": [
                "Imagina que {content}",
                "Como si fuera {content}",
                "Visualiza esto: {content}",
                "Déjame pintarte un cuadro: {content}"
            ],
            "conversational": [
                "Te cuento que {content}",
                "Sabes qué, {content}",
                "Mira, te explico: {content}",
                "Oye, {content}"
            ]
        }
    
    def _initialize_synonyms(self) -> Dict[str, List[str]]:
        """Inicializa base de datos de sinónimos"""
        return {
            "importante": ["relevante", "significativo", "crucial", "esencial", "fundamental"],
            "bueno": ["excelente", "genial", "fantástico", "maravilloso", "brillante"],
            "malo": ["terrible", "horrible", "pésimo", "deficiente", "inadecuado"],
            "grande": ["enorme", "inmenso", "colosal", "gigante", "masivo"],
            "pequeño": ["diminuto", "minúsculo", "microscópico", "reducido", "compacto"],
            "rápido": ["veloz", "ágil", "expeditivo", "pronto", "inmediato"],
            "lento": ["pausado", "tranquilo", "gradual", "progresivo", "mesurado"],
            "fácil": ["sencillo", "simple", "básico", "elemental", "directo"],
            "difícil": ["complejo", "intrincado", "complicado", "desafiante", "arduo"],
            "nuevo": ["innovador", "moderno", "actual", "contemporáneo", "reciente"],
            "viejo": ["antiguo", "tradicional", "clásico", "veterano", "experimentado"],
            "mejor": ["superior", "óptimo", "ideal", "perfecto", "excelente"],
            "peor": ["inferior", "deficiente", "inadecuado", "insuficiente", "limitado"]
        }
    
    def _initialize_style_patterns(self) -> Dict[ResponseStyle, Dict[str, Any]]:
        """Inicializa patrones de estilo"""
        return {
            ResponseStyle.FORMAL: {
                "sentence_starters": ["Además", "Por consiguiente", "En consecuencia", "Por tanto"],
                "connectors": ["además", "asimismo", "por otro lado", "en contraste"],
                "endings": ["conclusivamente", "en resumen", "para finalizar", "en definitiva"]
            },
            ResponseStyle.CASUAL: {
                "sentence_starters": ["Y bueno", "Pues", "Entonces", "Así que"],
                "connectors": ["y", "pero", "o sea", "como que"],
                "endings": ["y ya", "eso es todo", "nada más", "y listo"]
            },
            ResponseStyle.TECHNICAL: {
                "sentence_starters": ["Técnicamente", "Desde el punto de vista técnico", "En términos de", "A nivel de"],
                "connectors": ["por consiguiente", "por lo tanto", "en consecuencia", "así mismo"],
                "endings": ["en conclusión", "en resumen técnico", "para finalizar", "en definitiva"]
            },
            ResponseStyle.SIMPLIFIED: {
                "sentence_starters": ["Básicamente", "En resumen", "Lo que pasa es que", "Te explico"],
                "connectors": ["y", "pero", "entonces", "por eso"],
                "endings": ["y eso es todo", "en pocas palabras", "así de simple", "y ya está"]
            },
            ResponseStyle.CREATIVE: {
                "sentence_starters": ["Imagina", "Visualiza", "Como si fuera", "Déjame que te cuente"],
                "connectors": ["y entonces", "de repente", "como por arte de magia", "y voilà"],
                "endings": ["y así es como", "y ahí tienes", "y listo", "y ya está"]
            },
            ResponseStyle.CONVERSATIONAL: {
                "sentence_starters": ["Te cuento que", "Sabes qué", "Mira", "Oye"],
                "connectors": ["y", "pero", "o sea", "como que"],
                "endings": ["y ya", "eso es todo", "nada más", "y listo"]
            }
        }
    
    async def generate_alternatives(self, response: str, analysis: Any, 
                                  context: Any, num_alternatives: int = 3) -> List[AlternativeResponse]:
        """
        Genera respuestas alternativas
        
        Args:
            response: Respuesta original
            analysis: Análisis de la respuesta
            context: Contexto de la respuesta
            num_alternatives: Número de alternativas a generar
        
        Returns:
            Lista de respuestas alternativas
        """
        try:
            alternatives = []
            
            # Determinar estilos apropiados basados en el análisis
            appropriate_styles = self._select_appropriate_styles(analysis, context)
            
            # Generar alternativas
            for i in range(min(num_alternatives, len(appropriate_styles))):
                style = appropriate_styles[i]
                
                # Generar respuesta alternativa
                alternative_text = await self._generate_alternative_text(
                    response, style, analysis, context
                )
                
                # Calcular métricas
                confidence = self._calculate_generation_confidence(alternative_text, response, style)
                semantic_similarity = self._calculate_semantic_similarity(alternative_text, response)
                
                # Crear respuesta alternativa
                alternative = AlternativeResponse(
                    id=f"alt_{int(datetime.now().timestamp())}_{i}",
                    original_response=response,
                    alternative_text=alternative_text,
                    style=style,
                    length=self._determine_length(alternative_text),
                    confidence=confidence,
                    transformation_applied=f"style_{style.value}",
                    semantic_similarity=semantic_similarity,
                    timestamp=datetime.now()
                )
                
                alternatives.append(alternative)
            
            # Guardar en historial
            self.generated_responses.extend(alternatives)
            
            logger.info(f"Generadas {len(alternatives)} respuestas alternativas")
            return alternatives
            
        except Exception as e:
            logger.error(f"Error generando alternativas: {e}")
            return []
    
    def _select_appropriate_styles(self, analysis: Any, context: Any) -> List[ResponseStyle]:
        """Selecciona estilos apropiados basados en el análisis"""
        try:
            styles = []
            
            # Estilo basado en tipo de respuesta
            if analysis.response_type.value == "technical":
                styles.extend([ResponseStyle.TECHNICAL, ResponseStyle.SIMPLIFIED])
            elif analysis.response_type.value == "creative":
                styles.extend([ResponseStyle.CREATIVE, ResponseStyle.CASUAL])
            elif analysis.response_type.value == "factual":
                styles.extend([ResponseStyle.FORMAL, ResponseStyle.CONVERSATIONAL])
            else:
                styles.extend([ResponseStyle.CONVERSATIONAL, ResponseStyle.CASUAL])
            
            # Ajustar basado en complejidad
            if analysis.complexity > 0.7:
                styles.append(ResponseStyle.SIMPLIFIED)
            
            # Ajustar basado en nivel técnico
            if analysis.technical_level >= 4:
                styles.append(ResponseStyle.TECHNICAL)
            
            # Ajustar basado en contexto emocional
            if abs(analysis.sentiment) > 0.5:
                styles.append(ResponseStyle.CONVERSATIONAL)
            
            # Eliminar duplicados y limitar
            unique_styles = list(dict.fromkeys(styles))[:4]
            
            return unique_styles
            
        except Exception as e:
            logger.error(f"Error seleccionando estilos: {e}")
            return [ResponseStyle.CONVERSATIONAL, ResponseStyle.CASUAL]
    
    async def _generate_alternative_text(self, response: str, style: ResponseStyle, 
                                       analysis: Any, context: Any) -> str:
        """Genera texto alternativo para un estilo específico"""
        try:
            # Aplicar transformaciones de sinónimos
            synonymized_text = self._apply_synonym_transformations(response)
            
            # Aplicar transformaciones de estilo
            styled_text = self._apply_style_transformations(synonymized_text, style)
            
            # Aplicar transformaciones de plantilla
            templated_text = self._apply_template_transformations(styled_text, style)
            
            # Ajustar longitud si es necesario
            final_text = self._adjust_length(templated_text, style, analysis)
            
            return final_text
            
        except Exception as e:
            logger.error(f"Error generando texto alternativo: {e}")
            return response
    
    def _apply_synonym_transformations(self, text: str) -> str:
        """Aplica transformaciones de sinónimos"""
        try:
            transformed_text = text
            
            for original, synonyms in self.synonym_database.items():
                if original in transformed_text.lower():
                    # Seleccionar sinónimo aleatorio
                    synonym = random.choice(synonyms)
                    
                    # Reemplazar (manteniendo capitalización)
                    pattern = re.compile(re.escape(original), re.IGNORECASE)
                    transformed_text = pattern.sub(synonym, transformed_text)
            
            return transformed_text
            
        except Exception as e:
            logger.error(f"Error aplicando sinónimos: {e}")
            return text
    
    def _apply_style_transformations(self, text: str, style: ResponseStyle) -> str:
        """Aplica transformaciones de estilo"""
        try:
            if style not in self.style_patterns:
                return text
            
            patterns = self.style_patterns[style]
            transformed_text = text
            
            # Agregar conector inicial ocasionalmente
            if random.random() < 0.3 and patterns["sentence_starters"]:
                starter = random.choice(patterns["sentence_starters"])
                transformed_text = f"{starter}, {transformed_text.lower()}"
            
            # Reemplazar conectores
            for connector in patterns["connectors"]:
                # Patrones comunes de conectores
                common_connectors = ["y", "pero", "entonces", "por eso"]
                for common in common_connectors:
                    if common in transformed_text.lower() and connector != common:
                        transformed_text = transformed_text.replace(common, connector)
            
            return transformed_text
            
        except Exception as e:
            logger.error(f"Error aplicando transformaciones de estilo: {e}")
            return text
    
    def _apply_template_transformations(self, text: str, style: ResponseStyle) -> str:
        """Aplica transformaciones de plantilla"""
        try:
            if style.value not in self.generation_templates:
                return text
            
            templates = self.generation_templates[style.value]
            
            # Aplicar plantilla ocasionalmente
            if random.random() < 0.4:
                template = random.choice(templates)
                if "{content}" in template:
                    return template.format(content=text)
                else:
                    return f"{template} {text}"
            
            return text
            
        except Exception as e:
            logger.error(f"Error aplicando plantillas: {e}")
            return text
    
    def _adjust_length(self, text: str, style: ResponseStyle, analysis: Any) -> str:
        """Ajusta la longitud del texto según el estilo"""
        try:
            # Longitud objetivo basada en el estilo
            length_targets = {
                ResponseStyle.FORMAL: 150,
                ResponseStyle.CASUAL: 100,
                ResponseStyle.TECHNICAL: 200,
                ResponseStyle.SIMPLIFIED: 80,
                ResponseStyle.CREATIVE: 120,
                ResponseStyle.CONVERSATIONAL: 90
            }
            
            target_length = length_targets.get(style, 100)
            current_length = len(text)
            
            if current_length < target_length * 0.7:
                # Expandir texto
                return self._expand_text(text, style)
            elif current_length > target_length * 1.5:
                # Comprimir texto
                return self._compress_text(text)
            else:
                return text
                
        except Exception as e:
            logger.error(f"Error ajustando longitud: {e}")
            return text
    
    def _expand_text(self, text: str, style: ResponseStyle) -> str:
        """Expande el texto para alcanzar la longitud objetivo"""
        try:
            # Agregar explicaciones adicionales
            expansions = [
                "Para ser más específico, ",
                "En detalle, ",
                "Más concretamente, ",
                "Para aclarar, "
            ]
            
            if random.random() < 0.5:
                expansion = random.choice(expansions)
                return f"{expansion}{text.lower()}"
            
            return text
            
        except Exception as e:
            logger.error(f"Error expandiendo texto: {e}")
            return text
    
    def _compress_text(self, text: str) -> str:
        """Comprime el texto para reducir la longitud"""
        try:
            # Eliminar palabras redundantes
            redundant_words = ["realmente", "verdaderamente", "absolutamente", "completamente"]
            
            compressed = text
            for word in redundant_words:
                compressed = compressed.replace(f" {word} ", " ")
            
            # Simplificar conectores largos
            connector_replacements = {
                "por consiguiente": "entonces",
                "en consecuencia": "por eso",
                "asimismo": "también",
                "por otro lado": "pero"
            }
            
            for long_connector, short_connector in connector_replacements.items():
                compressed = compressed.replace(long_connector, short_connector)
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error comprimiendo texto: {e}")
            return text
    
    def _determine_length(self, text: str) -> ResponseLength:
        """Determina la longitud de la respuesta"""
        word_count = len(text.split())
        
        if word_count < 50:
            return ResponseLength.SHORT
        elif word_count < 150:
            return ResponseLength.MEDIUM
        else:
            return ResponseLength.LONG
    
    def _calculate_generation_confidence(self, alternative_text: str, original_text: str, 
                                       style: ResponseStyle) -> float:
        """Calcula la confianza en la generación"""
        try:
            # Factores de confianza
            length_factor = min(1.0, len(alternative_text) / 200)
            style_consistency = self._check_style_consistency(alternative_text, style)
            semantic_preservation = self._check_semantic_preservation(alternative_text, original_text)
            
            confidence = (length_factor * 0.3 + style_consistency * 0.4 + semantic_preservation * 0.3)
            
            return min(1.0, max(0.1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5
    
    def _check_style_consistency(self, text: str, style: ResponseStyle) -> float:
        """Verifica la consistencia del estilo"""
        try:
            if style not in self.style_patterns:
                return 0.5
            
            patterns = self.style_patterns[style]
            consistency_score = 0.0
            
            # Verificar presencia de elementos de estilo
            for starter in patterns["sentence_starters"]:
                if starter.lower() in text.lower():
                    consistency_score += 0.3
            
            for connector in patterns["connectors"]:
                if connector in text.lower():
                    consistency_score += 0.2
            
            return min(1.0, consistency_score)
            
        except Exception as e:
            logger.error(f"Error verificando consistencia de estilo: {e}")
            return 0.5
    
    def _check_semantic_preservation(self, alternative_text: str, original_text: str) -> float:
        """Verifica la preservación semántica"""
        try:
            # Comparación simple de palabras clave
            original_words = set(original_text.lower().split())
            alternative_words = set(alternative_text.lower().split())
            
            # Calcular intersección
            common_words = original_words.intersection(alternative_words)
            
            if len(original_words) == 0:
                return 0.5
            
            similarity = len(common_words) / len(original_words)
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Error verificando preservación semántica: {e}")
            return 0.5
    
    def _calculate_semantic_similarity(self, alternative_text: str, original_text: str) -> float:
        """Calcula la similitud semántica entre textos"""
        try:
            # Método simple basado en palabras comunes
            original_words = set(original_text.lower().split())
            alternative_words = set(alternative_text.lower().split())
            
            if len(original_words) == 0 and len(alternative_words) == 0:
                return 1.0
            
            if len(original_words) == 0 or len(alternative_words) == 0:
                return 0.0
            
            # Jaccard similarity
            intersection = len(original_words.intersection(alternative_words))
            union = len(original_words.union(alternative_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando similitud semántica: {e}")
            return 0.5
    
    async def get_generation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del generador"""
        return {
            "total_generated": len(self.generated_responses),
            "styles_used": {
                style.value: sum(1 for r in self.generated_responses if r.style == style)
                for style in ResponseStyle
            },
            "avg_confidence": sum(r.confidence for r in self.generated_responses) / max(len(self.generated_responses), 1),
            "avg_semantic_similarity": sum(r.semantic_similarity for r in self.generated_responses) / max(len(self.generated_responses), 1)
        }
