"""
Sintetizador de Conocimiento
Versión: 0.6.0
Sintetiza conocimiento de múltiples respuestas de IAs
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Celebro_Synthesizer')

class KnowledgeType(Enum):
    """Tipos de conocimiento"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    EXPERIENTIAL = "experiential"
    METACOGNITIVE = "metacognitive"

class SynthesisMethod(Enum):
    """Métodos de síntesis"""
    AGGREGATION = "aggregation"
    INTEGRATION = "integration"
    COMPARISON = "comparison"
    EXTRAPOLATION = "extrapolation"
    GENERALIZATION = "generalization"

@dataclass
class KnowledgeChunk:
    """Fragmento de conocimiento"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    source_responses: List[str]
    confidence: float
    relevance_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class SynthesizedKnowledge:
    """Conocimiento sintetizado"""
    id: str
    topic: str
    knowledge_chunks: List[KnowledgeChunk]
    synthesis_method: SynthesisMethod
    overall_confidence: float
    coherence_score: float
    completeness_score: float
    timestamp: datetime

class KnowledgeSynthesizer:
    """
    Sintetizador de conocimiento para @celebro.
    Combina y sintetiza conocimiento de múltiples respuestas de IAs.
    """
    
    def __init__(self):
        self.knowledge_base = {}
        self.synthesis_rules = self._initialize_synthesis_rules()
        self.topic_clusters = {}
        self.knowledge_graph = {}
        
        logger.info("Sintetizador de conocimiento inicializado")
    
    def _initialize_synthesis_rules(self) -> Dict[str, Dict[str, Any]]:
        """Inicializa reglas de síntesis"""
        return {
            "factual_consolidation": {
                "method": SynthesisMethod.AGGREGATION,
                "conditions": ["factual_responses >= 2", "similar_topics"],
                "confidence_threshold": 0.7
            },
            "procedural_integration": {
                "method": SynthesisMethod.INTEGRATION,
                "conditions": ["procedural_responses >= 2", "complementary_steps"],
                "confidence_threshold": 0.8
            },
            "conceptual_synthesis": {
                "method": SynthesisMethod.GENERALIZATION,
                "conditions": ["conceptual_responses >= 3", "related_concepts"],
                "confidence_threshold": 0.6
            },
            "experiential_combination": {
                "method": SynthesisMethod.COMPARISON,
                "conditions": ["experiential_responses >= 2", "different_perspectives"],
                "confidence_threshold": 0.5
            }
        }
    
    async def synthesize_knowledge(self, responses: List[Any], 
                                 topic: str = None) -> SynthesizedKnowledge:
        """
        Sintetiza conocimiento de múltiples respuestas
        
        Args:
            responses: Lista de respuestas analizadas
            topic: Tema principal (opcional)
        
        Returns:
            Conocimiento sintetizado
        """
        try:
            if not responses:
                return self._create_empty_synthesis(topic)
            
            # Agrupar respuestas por similitud
            response_groups = await self._group_similar_responses(responses)
            
            # Extraer fragmentos de conocimiento
            knowledge_chunks = await self._extract_knowledge_chunks(responses, response_groups)
            
            # Determinar método de síntesis
            synthesis_method = self._determine_synthesis_method(knowledge_chunks)
            
            # Sintetizar conocimiento
            synthesized = await self._perform_synthesis(
                knowledge_chunks, synthesis_method, topic
            )
            
            # Guardar en base de conocimiento
            synthesis_id = f"synth_{int(datetime.now().timestamp())}"
            synthesized.id = synthesis_id
            self.knowledge_base[synthesis_id] = synthesized
            
            logger.info(f"Conocimiento sintetizado: {synthesis_id} - {len(knowledge_chunks)} fragmentos")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error sintetizando conocimiento: {e}")
            return self._create_empty_synthesis(topic)
    
    async def _group_similar_responses(self, responses: List[Any]) -> List[List[Any]]:
        """Agrupa respuestas similares"""
        try:
            groups = []
            used_responses = set()
            
            for i, response in enumerate(responses):
                if i in used_responses:
                    continue
                
                group = [response]
                used_responses.add(i)
                
                # Buscar respuestas similares
                for j, other_response in enumerate(responses[i+1:], i+1):
                    if j in used_responses:
                        continue
                    
                    similarity = self._calculate_response_similarity(response, other_response)
                    if similarity > 0.6:  # Umbral de similitud
                        group.append(other_response)
                        used_responses.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error agrupando respuestas: {e}")
            return [[r] for r in responses]
    
    def _calculate_response_similarity(self, response1: Any, response2: Any) -> float:
        """Calcula similitud entre dos respuestas"""
        try:
            # Similitud basada en conceptos clave
            concepts1 = set(response1.key_concepts if hasattr(response1, 'key_concepts') else [])
            concepts2 = set(response2.key_concepts if hasattr(response2, 'key_concepts') else [])
            
            if not concepts1 and not concepts2:
                return 0.5
            
            if not concepts1 or not concepts2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(concepts1.intersection(concepts2))
            union = len(concepts1.union(concepts2))
            
            concept_similarity = intersection / union if union > 0 else 0.0
            
            # Similitud basada en tipo de respuesta
            type_similarity = 1.0 if response1.response_type == response2.response_type else 0.0
            
            # Similitud basada en nivel técnico
            tech_diff = abs(response1.technical_level - response2.technical_level)
            tech_similarity = max(0.0, 1.0 - (tech_diff / 5.0))
            
            # Similitud combinada
            overall_similarity = (concept_similarity * 0.5 + type_similarity * 0.3 + tech_similarity * 0.2)
            
            return overall_similarity
            
        except Exception as e:
            logger.error(f"Error calculando similitud: {e}")
            return 0.0
    
    async def _extract_knowledge_chunks(self, responses: List[Any], 
                                      response_groups: List[List[Any]]) -> List[KnowledgeChunk]:
        """Extrae fragmentos de conocimiento de las respuestas"""
        try:
            knowledge_chunks = []
            
            for group in response_groups:
                # Determinar tipo de conocimiento del grupo
                knowledge_type = self._determine_group_knowledge_type(group)
                
                # Extraer contenido principal
                main_content = self._extract_main_content(group)
                
                # Calcular confianza del grupo
                group_confidence = self._calculate_group_confidence(group)
                
                # Calcular relevancia
                relevance_score = self._calculate_relevance_score(group)
                
                # Crear fragmento de conocimiento
                chunk = KnowledgeChunk(
                    id=f"chunk_{int(datetime.now().timestamp())}_{len(knowledge_chunks)}",
                    content=main_content,
                    knowledge_type=knowledge_type,
                    source_responses=[r.response_id for r in group if hasattr(r, 'response_id')],
                    confidence=group_confidence,
                    relevance_score=relevance_score,
                    timestamp=datetime.now(),
                    metadata={
                        "group_size": len(group),
                        "response_types": [r.response_type.value for r in group if hasattr(r, 'response_type')],
                        "avg_technical_level": sum(r.technical_level for r in group if hasattr(r, 'technical_level')) / len(group)
                    }
                )
                
                knowledge_chunks.append(chunk)
            
            return knowledge_chunks
            
        except Exception as e:
            logger.error(f"Error extrayendo fragmentos de conocimiento: {e}")
            return []
    
    def _determine_group_knowledge_type(self, group: List[Any]) -> KnowledgeType:
        """Determina el tipo de conocimiento de un grupo"""
        try:
            # Contar tipos de respuesta en el grupo
            type_counts = {}
            for response in group:
                if hasattr(response, 'response_type'):
                    response_type = response.response_type.value
                    type_counts[response_type] = type_counts.get(response_type, 0) + 1
            
            # Determinar tipo dominante
            if type_counts.get("factual", 0) > 0:
                return KnowledgeType.FACTUAL
            elif type_counts.get("instructive", 0) > 0:
                return KnowledgeType.PROCEDURAL
            elif type_counts.get("analytical", 0) > 0:
                return KnowledgeType.CONCEPTUAL
            elif type_counts.get("creative", 0) > 0:
                return KnowledgeType.EXPERIENTIAL
            else:
                return KnowledgeType.FACTUAL
                
        except Exception as e:
            logger.error(f"Error determinando tipo de conocimiento: {e}")
            return KnowledgeType.FACTUAL
    
    def _extract_main_content(self, group: List[Any]) -> str:
        """Extrae el contenido principal de un grupo de respuestas"""
        try:
            if not group:
                return ""
            
            # Usar la respuesta con mayor confianza como base
            best_response = max(group, key=lambda r: r.confidence if hasattr(r, 'confidence') else 0)
            
            # Combinar hechos extraídos de todas las respuestas
            all_facts = []
            for response in group:
                if hasattr(response, 'extracted_facts'):
                    all_facts.extend(response.extracted_facts)
            
            # Crear contenido combinado
            content_parts = [best_response.original_response]
            
            if all_facts:
                facts_text = "Hechos adicionales: " + "; ".join(all_facts[:3])
                content_parts.append(facts_text)
            
            return " ".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error extrayendo contenido principal: {e}")
            return group[0].original_response if group else ""
    
    def _calculate_group_confidence(self, group: List[Any]) -> float:
        """Calcula la confianza de un grupo de respuestas"""
        try:
            if not group:
                return 0.0
            
            # Promedio de confianza del grupo
            avg_confidence = sum(r.confidence for r in group if hasattr(r, 'confidence')) / len(group)
            
            # Bonificación por consenso
            consensus_bonus = 0.1 if len(group) > 1 else 0.0
            
            # Bonificación por similitud
            similarity_bonus = 0.1 if len(group) > 1 else 0.0
            
            total_confidence = avg_confidence + consensus_bonus + similarity_bonus
            
            return min(1.0, total_confidence)
            
        except Exception as e:
            logger.error(f"Error calculando confianza del grupo: {e}")
            return 0.5
    
    def _calculate_relevance_score(self, group: List[Any]) -> float:
        """Calcula la puntuación de relevancia de un grupo"""
        try:
            if not group:
                return 0.0
            
            # Factores de relevancia
            avg_confidence = sum(r.confidence for r in group if hasattr(r, 'confidence')) / len(group)
            group_size = len(group)
            
            # Relevancia basada en confianza y tamaño del grupo
            relevance = (avg_confidence * 0.7) + (min(1.0, group_size / 5) * 0.3)
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Error calculando relevancia: {e}")
            return 0.5
    
    def _determine_synthesis_method(self, knowledge_chunks: List[KnowledgeChunk]) -> SynthesisMethod:
        """Determina el método de síntesis apropiado"""
        try:
            if not knowledge_chunks:
                return SynthesisMethod.AGGREGATION
            
            # Contar tipos de conocimiento
            type_counts = {}
            for chunk in knowledge_chunks:
                chunk_type = chunk.knowledge_type.value
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            # Determinar método basado en tipos dominantes
            if type_counts.get("factual", 0) >= 2:
                return SynthesisMethod.AGGREGATION
            elif type_counts.get("procedural", 0) >= 2:
                return SynthesisMethod.INTEGRATION
            elif type_counts.get("conceptual", 0) >= 3:
                return SynthesisMethod.GENERALIZATION
            elif type_counts.get("experiential", 0) >= 2:
                return SynthesisMethod.COMPARISON
            else:
                return SynthesisMethod.AGGREGATION
                
        except Exception as e:
            logger.error(f"Error determinando método de síntesis: {e}")
            return SynthesisMethod.AGGREGATION
    
    async def _perform_synthesis(self, knowledge_chunks: List[KnowledgeChunk], 
                               method: SynthesisMethod, topic: str = None) -> SynthesizedKnowledge:
        """Realiza la síntesis de conocimiento"""
        try:
            # Determinar tema si no se proporciona
            if not topic:
                topic = self._extract_topic_from_chunks(knowledge_chunks)
            
            # Aplicar método de síntesis
            if method == SynthesisMethod.AGGREGATION:
                synthesized_content = self._aggregate_knowledge(knowledge_chunks)
            elif method == SynthesisMethod.INTEGRATION:
                synthesized_content = self._integrate_knowledge(knowledge_chunks)
            elif method == SynthesisMethod.GENERALIZATION:
                synthesized_content = self._generalize_knowledge(knowledge_chunks)
            elif method == SynthesisMethod.COMPARISON:
                synthesized_content = self._compare_knowledge(knowledge_chunks)
            else:
                synthesized_content = self._aggregate_knowledge(knowledge_chunks)
            
            # Calcular métricas de síntesis
            overall_confidence = self._calculate_overall_confidence(knowledge_chunks)
            coherence_score = self._calculate_coherence_score(knowledge_chunks)
            completeness_score = self._calculate_completeness_score(knowledge_chunks)
            
            # Crear conocimiento sintetizado
            synthesized = SynthesizedKnowledge(
                id="",  # Se asignará después
                topic=topic,
                knowledge_chunks=knowledge_chunks,
                synthesis_method=method,
                overall_confidence=overall_confidence,
                coherence_score=coherence_score,
                completeness_score=completeness_score,
                timestamp=datetime.now()
            )
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error realizando síntesis: {e}")
            return self._create_empty_synthesis(topic)
    
    def _extract_topic_from_chunks(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """Extrae el tema principal de los fragmentos de conocimiento"""
        try:
            if not knowledge_chunks:
                return "General"
            
            # Usar el fragmento con mayor relevancia
            best_chunk = max(knowledge_chunks, key=lambda c: c.relevance_score)
            
            # Extraer palabras clave del contenido
            content_words = best_chunk.content.split()
            important_words = [word for word in content_words if len(word) > 4]
            
            if important_words:
                return important_words[0].title()
            else:
                return "General"
                
        except Exception as e:
            logger.error(f"Error extrayendo tema: {e}")
            return "General"
    
    def _aggregate_knowledge(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """Agrega conocimiento de múltiples fragmentos"""
        try:
            # Ordenar por relevancia
            sorted_chunks = sorted(knowledge_chunks, key=lambda c: c.relevance_score, reverse=True)
            
            # Combinar contenido
            aggregated_parts = []
            for chunk in sorted_chunks[:3]:  # Top 3 fragmentos
                aggregated_parts.append(chunk.content)
            
            return " ".join(aggregated_parts)
            
        except Exception as e:
            logger.error(f"Error agregando conocimiento: {e}")
            return ""
    
    def _integrate_knowledge(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """Integra conocimiento de múltiples fragmentos"""
        try:
            # Crear narrativa integrada
            integration_parts = []
            
            for i, chunk in enumerate(knowledge_chunks):
                if i == 0:
                    integration_parts.append(f"En primer lugar, {chunk.content.lower()}")
                elif i == 1:
                    integration_parts.append(f"Además, {chunk.content.lower()}")
                else:
                    integration_parts.append(f"Finalmente, {chunk.content.lower()}")
            
            return " ".join(integration_parts)
            
        except Exception as e:
            logger.error(f"Error integrando conocimiento: {e}")
            return ""
    
    def _generalize_knowledge(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """Generaliza conocimiento de múltiples fragmentos"""
        try:
            # Crear generalización
            generalization_parts = []
            
            # Agrupar por tipo de conocimiento
            type_groups = {}
            for chunk in knowledge_chunks:
                chunk_type = chunk.knowledge_type.value
                if chunk_type not in type_groups:
                    type_groups[chunk_type] = []
                type_groups[chunk_type].append(chunk)
            
            # Crear generalización por tipo
            for chunk_type, chunks in type_groups.items():
                if chunk_type == "factual":
                    generalization_parts.append(f"En términos generales, {chunks[0].content.lower()}")
                elif chunk_type == "conceptual":
                    generalization_parts.append(f"Conceptualmente, {chunks[0].content.lower()}")
                elif chunk_type == "procedural":
                    generalization_parts.append(f"Procedimentalmente, {chunks[0].content.lower()}")
            
            return " ".join(generalization_parts)
            
        except Exception as e:
            logger.error(f"Error generalizando conocimiento: {e}")
            return ""
    
    def _compare_knowledge(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        """Compara conocimiento de múltiples fragmentos"""
        try:
            if len(knowledge_chunks) < 2:
                return knowledge_chunks[0].content if knowledge_chunks else ""
            
            # Crear comparación
            comparison_parts = []
            
            for i, chunk in enumerate(knowledge_chunks):
                if i == 0:
                    comparison_parts.append(f"Una perspectiva sugiere que {chunk.content.lower()}")
                elif i == 1:
                    comparison_parts.append(f"Otra perspectiva indica que {chunk.content.lower()}")
                else:
                    comparison_parts.append(f"Una tercera perspectiva propone que {chunk.content.lower()}")
            
            return " ".join(comparison_parts)
            
        except Exception as e:
            logger.error(f"Error comparando conocimiento: {e}")
            return ""
    
    def _calculate_overall_confidence(self, knowledge_chunks: List[KnowledgeChunk]) -> float:
        """Calcula la confianza general de la síntesis"""
        try:
            if not knowledge_chunks:
                return 0.0
            
            # Promedio ponderado por relevancia
            total_weighted_confidence = 0.0
            total_weight = 0.0
            
            for chunk in knowledge_chunks:
                weight = chunk.relevance_score
                total_weighted_confidence += chunk.confidence * weight
                total_weight += weight
            
            return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando confianza general: {e}")
            return 0.5
    
    def _calculate_coherence_score(self, knowledge_chunks: List[KnowledgeChunk]) -> float:
        """Calcula la puntuación de coherencia de la síntesis"""
        try:
            if len(knowledge_chunks) < 2:
                return 1.0
            
            # Coherencia basada en similitud de tipos de conocimiento
            type_counts = {}
            for chunk in knowledge_chunks:
                chunk_type = chunk.knowledge_type
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            # Calcular diversidad de tipos
            diversity = len(type_counts) / len(knowledge_chunks)
            
            # Coherencia inversamente proporcional a la diversidad
            coherence = 1.0 - (diversity - 0.5) if diversity > 0.5 else 1.0
            
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            logger.error(f"Error calculando coherencia: {e}")
            return 0.5
    
    def _calculate_completeness_score(self, knowledge_chunks: List[KnowledgeChunk]) -> float:
        """Calcula la puntuación de completitud de la síntesis"""
        try:
            if not knowledge_chunks:
                return 0.0
            
            # Completitud basada en número de fragmentos y su relevancia
            num_chunks = len(knowledge_chunks)
            avg_relevance = sum(c.relevance_score for c in knowledge_chunks) / num_chunks
            
            # Completitud combinada
            completeness = (min(1.0, num_chunks / 5) * 0.5) + (avg_relevance * 0.5)
            
            return completeness
            
        except Exception as e:
            logger.error(f"Error calculando completitud: {e}")
            return 0.5
    
    def _create_empty_synthesis(self, topic: str = None) -> SynthesizedKnowledge:
        """Crea una síntesis vacía"""
        return SynthesizedKnowledge(
            id=f"empty_{int(datetime.now().timestamp())}",
            topic=topic or "General",
            knowledge_chunks=[],
            synthesis_method=SynthesisMethod.AGGREGATION,
            overall_confidence=0.0,
            coherence_score=0.0,
            completeness_score=0.0,
            timestamp=datetime.now()
        )
    
    async def get_synthesis_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sintetizador"""
        return {
            "total_syntheses": len(self.knowledge_base),
            "total_knowledge_chunks": sum(len(s.knowledge_chunks) for s in self.knowledge_base.values()),
            "synthesis_methods": {
                method.value: sum(1 for s in self.knowledge_base.values() if s.synthesis_method == method)
                for method in SynthesisMethod
            },
            "knowledge_types": {
                kt.value: sum(1 for s in self.knowledge_base.values() 
                             for c in s.knowledge_chunks if c.knowledge_type == kt)
                for kt in KnowledgeType
            },
            "avg_confidence": sum(s.overall_confidence for s in self.knowledge_base.values()) / max(len(self.knowledge_base), 1)
        }
