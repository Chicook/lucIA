"""
Módulo 5: Sistema de Razonamiento
Versión: 0.6.0
Funcionalidad: Motor de razonamiento lógico, inferencia y toma de decisiones
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger('LucIA_Reasoning')

class ReasoningType(Enum):
    """Tipos de razonamiento"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"

class ConfidenceLevel(Enum):
    """Niveles de confianza"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class ReasoningStep:
    """Paso en el proceso de razonamiento"""
    step_id: str
    reasoning_type: ReasoningType
    premise: str
    conclusion: str
    confidence: float
    evidence: List[str]
    timestamp: datetime

@dataclass
class ReasoningChain:
    """Cadena de razonamiento"""
    chain_id: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    created_at: datetime

class ReasoningEngine:
    """
    Motor de razonamiento lógico para LucIA.
    Gestiona diferentes tipos de razonamiento y toma de decisiones.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.reasoning_chains = {}
        self.knowledge_base = {}
        self.rules = []
        self.patterns = {}
        self.confidence_threshold = 0.7
        self.max_reasoning_depth = 10
        
        # Estadísticas
        self.total_reasoning_cycles = 0
        self.successful_reasoning = 0
        self.failed_reasoning = 0
        
        # Inicializar patrones de razonamiento
        self._initialize_reasoning_patterns()
        
        logger.info("Motor de razonamiento inicializado")
    
    def _initialize_reasoning_patterns(self):
        """Inicializa patrones de razonamiento predefinidos"""
        self.patterns = {
            "if_then": {
                "pattern": r"if\s+(.+?)\s+then\s+(.+)",
                "type": ReasoningType.DEDUCTIVE,
                "confidence": 0.8
            },
            "causal": {
                "pattern": r"(.+?)\s+causes?\s+(.+)",
                "type": ReasoningType.CAUSAL,
                "confidence": 0.7
            },
            "temporal": {
                "pattern": r"(.+?)\s+before\s+(.+)",
                "type": ReasoningType.TEMPORAL,
                "confidence": 0.6
            },
            "analogical": {
                "pattern": r"(.+?)\s+is\s+like\s+(.+)",
                "type": ReasoningType.ANALOGICAL,
                "confidence": 0.5
            }
        }
    
    async def reason_about(self, query: str, context: Dict[str, Any] = None) -> ReasoningChain:
        """
        Realiza razonamiento sobre una consulta
        
        Args:
            query: Consulta a razonar
            context: Contexto adicional
        
        Returns:
            Cadena de razonamiento
        """
        try:
            self.total_reasoning_cycles += 1
            
            # Crear nueva cadena de razonamiento
            chain_id = f"reasoning_{int(datetime.now().timestamp())}"
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                steps=[],
                final_conclusion="",
                overall_confidence=0.0,
                created_at=datetime.now()
            )
            
            # Analizar la consulta
            analysis = await self._analyze_query(query, context or {})
            
            # Aplicar diferentes tipos de razonamiento
            reasoning_steps = []
            
            # Razonamiento deductivo
            deductive_steps = await self._deductive_reasoning(query, analysis)
            reasoning_steps.extend(deductive_steps)
            
            # Razonamiento inductivo
            inductive_steps = await self._inductive_reasoning(query, analysis)
            reasoning_steps.extend(inductive_steps)
            
            # Razonamiento abductivo
            abductive_steps = await self._abductive_reasoning(query, analysis)
            reasoning_steps.extend(abductive_steps)
            
            # Razonamiento analógico
            analogical_steps = await self._analogical_reasoning(query, analysis)
            reasoning_steps.extend(analogical_steps)
            
            # Evaluar y seleccionar mejores pasos
            best_steps = self._evaluate_reasoning_steps(reasoning_steps)
            
            # Construir cadena final
            reasoning_chain.steps = best_steps
            reasoning_chain.final_conclusion = self._synthesize_conclusion(best_steps)
            reasoning_chain.overall_confidence = self._calculate_overall_confidence(best_steps)
            
            # Guardar cadena
            self.reasoning_chains[chain_id] = reasoning_chain
            
            if reasoning_chain.overall_confidence >= self.confidence_threshold:
                self.successful_reasoning += 1
            else:
                self.failed_reasoning += 1
            
            logger.info(f"Razonamiento completado: {chain_id} (confianza: {reasoning_chain.overall_confidence:.2f})")
            
            return reasoning_chain
            
        except Exception as e:
            logger.error(f"Error en razonamiento: {e}")
            self.failed_reasoning += 1
            return ReasoningChain(
                chain_id="error",
                steps=[],
                final_conclusion="Error en el razonamiento",
                overall_confidence=0.0,
                created_at=datetime.now()
            )
    
    async def _analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la consulta para extraer información relevante"""
        analysis = {
            "keywords": [],
            "entities": [],
            "intent": "unknown",
            "complexity": "medium",
            "reasoning_type": "general"
        }
        
        # Extraer palabras clave
        words = re.findall(r'\b\w+\b', query.lower())
        analysis["keywords"] = words
        
        # Detectar entidades (nombres propios, números, etc.)
        entities = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', query)
        analysis["entities"] = entities
        
        # Detectar intención
        if any(word in query.lower() for word in ["what", "how", "why", "when", "where", "who"]):
            analysis["intent"] = "question"
        elif any(word in query.lower() for word in ["if", "then", "because", "therefore"]):
            analysis["intent"] = "logical"
        elif any(word in query.lower() for word in ["compare", "similar", "different"]):
            analysis["intent"] = "comparison"
        
        # Determinar complejidad
        if len(words) > 20:
            analysis["complexity"] = "high"
        elif len(words) < 5:
            analysis["complexity"] = "low"
        
        return analysis
    
    async def _deductive_reasoning(self, query: str, analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """Aplica razonamiento deductivo"""
        steps = []
        
        try:
            # Buscar reglas aplicables
            applicable_rules = self._find_applicable_rules(query)
            
            for rule in applicable_rules:
                step = ReasoningStep(
                    step_id=f"deductive_{len(steps)}",
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    premise=rule["premise"],
                    conclusion=rule["conclusion"],
                    confidence=rule.get("confidence", 0.8),
                    evidence=[rule["premise"]],
                    timestamp=datetime.now()
                )
                steps.append(step)
            
            # Aplicar patrones if-then
            if_then_matches = re.findall(self.patterns["if_then"]["pattern"], query, re.IGNORECASE)
            for premise, conclusion in if_then_matches:
                step = ReasoningStep(
                    step_id=f"deductive_pattern_{len(steps)}",
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    premise=premise.strip(),
                    conclusion=conclusion.strip(),
                    confidence=self.patterns["if_then"]["confidence"],
                    evidence=[premise.strip()],
                    timestamp=datetime.now()
                )
                steps.append(step)
            
        except Exception as e:
            logger.error(f"Error en razonamiento deductivo: {e}")
        
        return steps
    
    async def _inductive_reasoning(self, query: str, analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """Aplica razonamiento inductivo"""
        steps = []
        
        try:
            # Buscar patrones en datos históricos
            if self.core_engine and hasattr(self.core_engine, 'memory_system'):
                memories = await self.core_engine.memory_system.retrieve_memory(
                    query, memory_type="experience", limit=10
                )
                
                if memories:
                    # Analizar patrones en las memorias
                    patterns = self._find_patterns_in_memories(memories)
                    
                    for pattern in patterns:
                        step = ReasoningStep(
                            step_id=f"inductive_{len(steps)}",
                            reasoning_type=ReasoningType.INDUCTIVE,
                            premise=f"Basado en {len(pattern['evidence'])} casos observados",
                            conclusion=pattern["generalization"],
                            confidence=pattern["confidence"],
                            evidence=pattern["evidence"],
                            timestamp=datetime.now()
                        )
                        steps.append(step)
            
        except Exception as e:
            logger.error(f"Error en razonamiento inductivo: {e}")
        
        return steps
    
    async def _abductive_reasoning(self, query: str, analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """Aplica razonamiento abductivo (inferencia a la mejor explicación)"""
        steps = []
        
        try:
            # Generar hipótesis explicativas
            hypotheses = self._generate_explanatory_hypotheses(query, analysis)
            
            for i, hypothesis in enumerate(hypotheses):
                step = ReasoningStep(
                    step_id=f"abductive_{i}",
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    premise=f"Observación: {query}",
                    conclusion=f"Hipótesis: {hypothesis['explanation']}",
                    confidence=hypothesis["confidence"],
                    evidence=hypothesis["evidence"],
                    timestamp=datetime.now()
                )
                steps.append(step)
            
        except Exception as e:
            logger.error(f"Error en razonamiento abductivo: {e}")
        
        return steps
    
    async def _analogical_reasoning(self, query: str, analysis: Dict[str, Any]) -> List[ReasoningStep]:
        """Aplica razonamiento analógico"""
        steps = []
        
        try:
            # Buscar analogías en la base de conocimiento
            analogies = self._find_analogies(query)
            
            for analogy in analogies:
                step = ReasoningStep(
                    step_id=f"analogical_{len(steps)}",
                    reasoning_type=ReasoningType.ANALOGICAL,
                    premise=f"Analogía: {analogy['source']} es como {analogy['target']}",
                    conclusion=f"Por analogía: {analogy['conclusion']}",
                    confidence=analogy["confidence"],
                    evidence=[analogy["source"], analogy["target"]],
                    timestamp=datetime.now()
                )
                steps.append(step)
            
        except Exception as e:
            logger.error(f"Error en razonamiento analógico: {e}")
        
        return steps
    
    def _find_applicable_rules(self, query: str) -> List[Dict[str, Any]]:
        """Encuentra reglas aplicables a la consulta"""
        applicable_rules = []
        
        for rule in self.rules:
            if self._rule_matches_query(rule, query):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_matches_query(self, rule: Dict[str, Any], query: str) -> bool:
        """Verifica si una regla coincide con la consulta"""
        try:
            # Implementar lógica de coincidencia de reglas
            keywords = rule.get("keywords", [])
            return any(keyword.lower() in query.lower() for keyword in keywords)
        except:
            return False
    
    def _find_patterns_in_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra patrones en memorias para razonamiento inductivo"""
        patterns = []
        
        try:
            # Agrupar memorias por tipo
            memory_groups = defaultdict(list)
            for memory in memories:
                memory_type = memory.get("memory_type", "general")
                memory_groups[memory_type].append(memory)
            
            # Analizar patrones en cada grupo
            for memory_type, group_memories in memory_groups.items():
                if len(group_memories) >= 3:  # Mínimo para detectar patrones
                    pattern = self._extract_pattern_from_group(group_memories)
                    if pattern:
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error encontrando patrones en memorias: {e}")
        
        return patterns
    
    def _extract_pattern_from_group(self, memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extrae un patrón de un grupo de memorias"""
        try:
            # Implementar extracción de patrones
            # Por ahora, retornar un patrón simple
            if len(memories) >= 3:
                return {
                    "generalization": f"Patrón observado en {len(memories)} casos",
                    "confidence": min(0.8, len(memories) * 0.1),
                    "evidence": [mem["content"][:50] + "..." for mem in memories[:3]]
                }
        except Exception as e:
            logger.error(f"Error extrayendo patrón: {e}")
        
        return None
    
    def _generate_explanatory_hypotheses(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera hipótesis explicativas para razonamiento abductivo"""
        hypotheses = []
        
        try:
            # Generar hipótesis basadas en palabras clave
            keywords = analysis.get("keywords", [])
            
            for keyword in keywords[:3]:  # Limitar a 3 hipótesis
                hypothesis = {
                    "explanation": f"La causa más probable es {keyword}",
                    "confidence": 0.6,
                    "evidence": [f"Palabra clave: {keyword}"]
                }
                hypotheses.append(hypothesis)
            
        except Exception as e:
            logger.error(f"Error generando hipótesis: {e}")
        
        return hypotheses
    
    def _find_analogies(self, query: str) -> List[Dict[str, Any]]:
        """Encuentra analogías para razonamiento analógico"""
        analogies = []
        
        try:
            # Buscar analogías en la base de conocimiento
            # Por ahora, retornar analogías predefinidas
            predefined_analogies = [
                {
                    "source": "un sistema de IA",
                    "target": "un cerebro humano",
                    "conclusion": "puede aprender y adaptarse",
                    "confidence": 0.7
                }
            ]
            
            for analogy in predefined_analogies:
                if any(word in query.lower() for word in ["sistema", "ia", "inteligencia"]):
                    analogies.append(analogy)
            
        except Exception as e:
            logger.error(f"Error encontrando analogías: {e}")
        
        return analogies
    
    def _evaluate_reasoning_steps(self, steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """Evalúa y selecciona los mejores pasos de razonamiento"""
        try:
            # Ordenar por confianza
            sorted_steps = sorted(steps, key=lambda x: x.confidence, reverse=True)
            
            # Seleccionar los mejores pasos (máximo 5)
            best_steps = sorted_steps[:5]
            
            return best_steps
            
        except Exception as e:
            logger.error(f"Error evaluando pasos de razonamiento: {e}")
            return steps
    
    def _synthesize_conclusion(self, steps: List[ReasoningStep]) -> str:
        """Sintetiza una conclusión final de los pasos de razonamiento"""
        try:
            if not steps:
                return "No se pudo llegar a una conclusión"
            
            # Tomar la conclusión del paso con mayor confianza
            best_step = max(steps, key=lambda x: x.confidence)
            return best_step.conclusion
            
        except Exception as e:
            logger.error(f"Error sintetizando conclusión: {e}")
            return "Error en la síntesis de conclusión"
    
    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calcula la confianza general de la cadena de razonamiento"""
        try:
            if not steps:
                return 0.0
            
            # Promedio ponderado de confianzas
            total_confidence = sum(step.confidence for step in steps)
            return total_confidence / len(steps)
            
        except Exception as e:
            logger.error(f"Error calculando confianza general: {e}")
            return 0.0
    
    async def add_rule(self, rule: Dict[str, Any]):
        """Agrega una nueva regla de razonamiento"""
        try:
            self.rules.append(rule)
            logger.info(f"Regla agregada: {rule.get('name', 'Sin nombre')}")
        except Exception as e:
            logger.error(f"Error agregando regla: {e}")
    
    async def get_reasoning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de razonamiento"""
        return {
            "total_reasoning_cycles": self.total_reasoning_cycles,
            "successful_reasoning": self.successful_reasoning,
            "failed_reasoning": self.failed_reasoning,
            "success_rate": (self.successful_reasoning / max(self.total_reasoning_cycles, 1)) * 100,
            "total_rules": len(self.rules),
            "total_chains": len(self.reasoning_chains),
            "confidence_threshold": self.confidence_threshold
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de razonamiento"""
        try:
            state = {
                "reasoning_chains": {
                    chain_id: {
                        "chain_id": chain.chain_id,
                        "steps": [
                            {
                                "step_id": step.step_id,
                                "reasoning_type": step.reasoning_type.value,
                                "premise": step.premise,
                                "conclusion": step.conclusion,
                                "confidence": step.confidence,
                                "evidence": step.evidence,
                                "timestamp": step.timestamp.isoformat()
                            }
                            for step in chain.steps
                        ],
                        "final_conclusion": chain.final_conclusion,
                        "overall_confidence": chain.overall_confidence,
                        "created_at": chain.created_at.isoformat()
                    }
                    for chain_id, chain in self.reasoning_chains.items()
                },
                "rules": self.rules,
                "stats": {
                    "total_reasoning_cycles": self.total_reasoning_cycles,
                    "successful_reasoning": self.successful_reasoning,
                    "failed_reasoning": self.failed_reasoning
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/reasoning_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de razonamiento guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de razonamiento: {e}")

# Instancia global del motor de razonamiento
reasoning_engine = ReasoningEngine()

async def initialize_module(core_engine):
    """Inicializa el módulo de razonamiento"""
    global reasoning_engine
    reasoning_engine.core_engine = core_engine
    core_engine.reasoning_engine = reasoning_engine
    logger.info("Módulo de razonamiento inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de razonamiento"""
    if isinstance(input_data, str):
        # Realizar razonamiento sobre la entrada
        reasoning_chain = await reasoning_engine.reason_about(input_data, context)
        return {
            "query": input_data,
            "reasoning_chain": {
                "chain_id": reasoning_chain.chain_id,
                "conclusion": reasoning_chain.final_conclusion,
                "confidence": reasoning_chain.overall_confidence,
                "steps_count": len(reasoning_chain.steps)
            }
        }
    elif isinstance(input_data, dict) and "add_rule" in input_data:
        # Agregar nueva regla
        await reasoning_engine.add_rule(input_data["add_rule"])
        return {"rule_added": True}
    
    return input_data

def run_modulo5():
    """Función de compatibilidad con el sistema anterior"""
    print("🧠 Módulo 5: Sistema de Razonamiento")
    print("   - Razonamiento deductivo, inductivo y abductivo")
    print("   - Inferencia lógica y toma de decisiones")
    print("   - Evaluación de confianza")
    print("   - Cadenas de razonamiento")
    print("   ✅ Módulo inicializado correctamente")