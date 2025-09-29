#!/usr/bin/env python3
"""
Motor de Aprendizaje Profundo - @red_neuronal
Versión: 0.6.0
Sistema avanzado de aprendizaje que integra con Gemini para entrenamiento profundo
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import pickle
import hashlib

from .query_analyzer import DeepQueryAnalyzer, QueryAnalysis, QueryComplexity, QueryCategory
from .gemini_integration import GeminiIntegration
# from .neural_core import NeuralCore  # Evitar importación circular

logger = logging.getLogger('Neural_DeepLearning')

@dataclass
class LearningSession:
    """Sesión de aprendizaje profundo"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    queries: List[QueryAnalysis]
    learning_objectives: List[str]
    progress_metrics: Dict[str, Any]
    gemini_responses: List[Dict[str, Any]]
    knowledge_gaps: List[str]
    recommendations: List[str]
    status: str  # active, completed, paused

@dataclass
class LearningInsight:
    """Insight de aprendizaje generado"""
    insight_id: str
    session_id: str
    insight_type: str  # pattern, gap, recommendation, improvement
    content: str
    confidence: float
    evidence: List[str]
    timestamp: datetime
    actionable: bool

class DeepLearningEngine:
    """
    Motor de aprendizaje profundo que integra análisis de consultas con Gemini
    """
    
    def __init__(self):
        self.query_analyzer = DeepQueryAnalyzer()
        self.gemini_integration = GeminiIntegration()
        # self.neural_core = NeuralCore()  # Evitar importación circular
        
        self.active_sessions = {}
        self.completed_sessions = []
        self.learning_insights = []
        
        # Configuración de aprendizaje
        self.learning_config = {
            'max_session_duration': 3600,  # 1 hora
            'min_queries_per_session': 3,
            'insight_generation_threshold': 0.7,
            'knowledge_retention_days': 30,
            'adaptive_learning': True,
            'real_time_analysis': True
        }
        
        # Patrones de aprendizaje
        self.learning_patterns = {
            'progressive_complexity': [],
            'topic_transitions': [],
            'knowledge_gaps': [],
            'skill_development': []
        }
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'total_sessions': 0,
            'total_queries': 0,
            'average_session_duration': 0,
            'learning_efficiency': 0,
            'knowledge_retention_rate': 0,
            'insight_generation_rate': 0
        }
        
        logger.info("Motor de aprendizaje profundo inicializado")
    
    async def start_learning_session(self, initial_query: str, 
                                   learning_objectives: List[str] = None) -> str:
        """
        Inicia una nueva sesión de aprendizaje profundo
        
        Args:
            initial_query: Consulta inicial
            learning_objectives: Objetivos de aprendizaje
            
        Returns:
            ID de la sesión
        """
        try:
            # Generar ID de sesión
            session_id = self._generate_session_id()
            
            # Analizar consulta inicial
            initial_analysis = self.query_analyzer.analyze_query(initial_query)
            
            # Crear sesión
            session = LearningSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                queries=[initial_analysis],
                learning_objectives=learning_objectives or self._generate_learning_objectives(initial_analysis),
                progress_metrics={},
                gemini_responses=[],
                knowledge_gaps=[],
                recommendations=[],
                status='active'
            )
            
            # Guardar sesión activa
            self.active_sessions[session_id] = session
            
            # Procesar consulta inicial con Gemini
            await self._process_query_with_gemini(session, initial_query, initial_analysis)
            
            # Generar insights iniciales
            await self._generate_learning_insights(session)
            
            logger.info(f"Sesión de aprendizaje iniciada: {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error iniciando sesión de aprendizaje: {e}")
            raise
    
    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta en una sesión de aprendizaje activa
        
        Args:
            session_id: ID de la sesión
            query: Consulta a procesar
            
        Returns:
            Resultado del procesamiento
        """
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            session = self.active_sessions[session_id]
            
            # Analizar consulta
            query_analysis = self.query_analyzer.analyze_query(query, {
                'session_id': session_id,
                'previous_queries': len(session.queries),
                'session_duration': (datetime.now() - session.start_time).total_seconds()
            })
            
            # Agregar a la sesión
            session.queries.append(query_analysis)
            
            # Procesar con Gemini
            gemini_response = await self._process_query_with_gemini(session, query, query_analysis)
            
            # Actualizar métricas de progreso
            self._update_session_progress(session)
            
            # Generar insights si es necesario
            if self._should_generate_insights(session):
                await self._generate_learning_insights(session)
            
            # Verificar si la sesión debe continuar
            if self._should_end_session(session):
                await self._end_learning_session(session_id)
            
            return {
                'query_analysis': query_analysis,
                'gemini_response': gemini_response,
                'session_progress': session.progress_metrics,
                'insights': self._get_recent_insights(session_id),
                'recommendations': session.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error procesando consulta en sesión {session_id}: {e}")
            raise
    
    async def _process_query_with_gemini(self, session: LearningSession, 
                                       query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Procesa una consulta con Gemini de forma especializada"""
        try:
            # Crear prompt especializado basado en el análisis
            specialized_prompt = self._create_specialized_prompt(query, analysis, session)
            
            # Obtener respuesta de Gemini
            gemini_result = self.gemini_integration.generate_text(specialized_prompt)
            gemini_response = gemini_result.get('text', '') if isinstance(gemini_result, dict) else str(gemini_result)
            
            # Procesar respuesta
            processed_response = {
                'original_query': query,
                'analysis': {
                    'complexity': analysis.complexity.value,
                    'category': analysis.category.value,
                    'keywords': analysis.keywords,
                    'entities': analysis.entities,
                    'intent': analysis.intent,
                    'learning_potential': analysis.learning_potential
                },
                'gemini_response': gemini_response,
                'specialized_prompt': specialized_prompt,
                'timestamp': datetime.now().isoformat(),
                'session_context': {
                    'query_number': len(session.queries),
                    'session_duration': (datetime.now() - session.start_time).total_seconds(),
                    'learning_objectives': session.learning_objectives
                }
            }
            
            # Agregar a la sesión
            session.gemini_responses.append(processed_response)
            
            # Analizar respuesta para identificar gaps de conocimiento
            await self._analyze_response_for_gaps(session, processed_response)
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Error procesando consulta con Gemini: {e}")
            return {
                'error': str(e),
                'original_query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_specialized_prompt(self, query: str, analysis: QueryAnalysis, 
                                 session: LearningSession) -> str:
        """Crea un prompt especializado para Gemini"""
        # Base del prompt
        base_prompt = f"""Eres LucIA, un asistente especializado en ciberseguridad con capacidades de aprendizaje profundo.

CONTEXTO DE LA SESIÓN:
- Objetivos de aprendizaje: {', '.join(session.learning_objectives)}
- Consulta número: {len(session.queries)}
- Duración de sesión: {(datetime.now() - session.start_time).total_seconds():.0f} segundos

ANÁLISIS DE LA CONSULTA:
- Complejidad: {analysis.complexity.value}
- Categoría: {analysis.category.value}
- Palabras clave: {', '.join(analysis.keywords[:10])}
- Entidades técnicas: {', '.join(analysis.entities[:5])}
- Intención: {analysis.intent}
- Potencial de aprendizaje: {analysis.learning_potential:.2f}

CONSULTA DEL USUARIO: {query}

INSTRUCCIONES ESPECÍFICAS:
"""
        
        # Instrucciones específicas por complejidad
        if analysis.complexity == QueryComplexity.BASIC:
            base_prompt += """
- Explica de manera simple y clara
- Usa ejemplos prácticos y cotidianos
- Evita jerga técnica innecesaria
- Proporciona pasos claros y accesibles
"""
        elif analysis.complexity == QueryComplexity.INTERMEDIATE:
            base_prompt += """
- Proporciona información técnica pero accesible
- Incluye ejemplos de implementación
- Menciona mejores prácticas
- Explica el "por qué" detrás de las recomendaciones
"""
        elif analysis.complexity == QueryComplexity.ADVANCED:
            base_prompt += """
- Profundiza en aspectos técnicos avanzados
- Incluye análisis de arquitectura y diseño
- Menciona consideraciones de rendimiento y escalabilidad
- Proporciona ejemplos de código y configuraciones
"""
        else:  # EXPERT
            base_prompt += """
- Realiza un análisis forense y técnico profundo
- Incluye técnicas avanzadas y metodologías
- Menciona herramientas especializadas
- Proporciona análisis de casos reales y complejos
"""
        
        # Instrucciones específicas por categoría
        category_instructions = {
            QueryCategory.AUTHENTICATION: """
- Enfócate en mecanismos de autenticación seguros
- Menciona OAuth 2.0, SAML, JWT, 2FA, MFA
- Incluye consideraciones de implementación
- Explica vulnerabilidades comunes y mitigaciones
""",
            QueryCategory.ENCRYPTION: """
- Profundiza en algoritmos de cifrado (AES, RSA, ECC)
- Explica protocolos SSL/TLS y HTTPS
- Menciona gestión de claves y certificados
- Incluye consideraciones de rendimiento
""",
            QueryCategory.MALWARE: """
- Analiza técnicas de detección y prevención
- Menciona sandboxing, análisis de comportamiento
- Explica técnicas de evasión y contramedidas
- Incluye herramientas de análisis forense
""",
            QueryCategory.PHISHING: """
- Explica técnicas de detección y prevención
- Menciona SPF, DKIM, DMARC
- Incluye educación del usuario
- Explica herramientas de filtrado
""",
            QueryCategory.FIREWALL: """
- Profundiza en configuración y reglas
- Menciona NGFW, WAF, segmentación de red
- Explica monitoreo y logging
- Incluye consideraciones de rendimiento
""",
            QueryCategory.VULNERABILITY: """
- Analiza gestión de vulnerabilidades
- Menciona CVE, CVSS, parcheo
- Explica técnicas de explotación y mitigación
- Incluye herramientas de escaneo
""",
            QueryCategory.NETWORK_SECURITY: """
- Profundiza en seguridad de red
- Menciona IDS/IPS, SIEM, monitoreo
- Explica segmentación y microsegmentación
- Incluye análisis de tráfico
""",
            QueryCategory.INCIDENT_RESPONSE: """
- Explica respuesta a incidentes
- Menciona forensia digital, contención
- Incluye planes de recuperación
- Explica herramientas de análisis
""",
            QueryCategory.COMPLIANCE: """
- Profundiza en marcos de cumplimiento
- Menciona GDPR, SOX, HIPAA, PCI-DSS
- Explica auditorías y controles
- Incluye documentación y políticas
""",
            QueryCategory.CODE_SECURITY: """
- Analiza desarrollo seguro
- Menciona OWASP, SAST, DAST, SCA
- Explica DevSecOps y CI/CD
- Incluye herramientas de análisis de código
"""
        }
        
        base_prompt += category_instructions.get(analysis.category, "")
        
        # Instrucciones específicas por intención
        intent_instructions = {
            "explanatory": "\n- Proporciona una explicación clara y completa\n- Incluye contexto y antecedentes\n- Usa analogías cuando sea apropiado",
            "instructional": "\n- Proporciona pasos detallados y claros\n- Incluye ejemplos prácticos\n- Menciona herramientas y recursos necesarios",
            "troubleshooting": "\n- Diagnostica el problema paso a paso\n- Proporciona soluciones específicas\n- Incluye verificación y validación",
            "recommendation": "\n- Proporciona recomendaciones específicas\n- Justifica cada recomendación\n- Incluye consideraciones de implementación",
            "comparison": "\n- Compara opciones de manera objetiva\n- Incluye pros y contras\n- Proporciona recomendaciones basadas en contexto"
        }
        
        base_prompt += intent_instructions.get(analysis.intent, "")
        
        # Instrucciones finales
        base_prompt += f"""

RESPUESTA REQUERIDA:
- Responde en español de manera técnica pero accesible
- Incluye ejemplos prácticos y código cuando sea relevante
- Menciona herramientas específicas y recursos
- Proporciona información accionable y específica
- Adapta el nivel técnico a la complejidad identificada ({analysis.complexity.value})
- Enfócate en la categoría de ciberseguridad: {analysis.category.value}

CONSULTA: {query}"""
        
        return base_prompt
    
    async def _analyze_response_for_gaps(self, session: LearningSession, 
                                       response: Dict[str, Any]):
        """Analiza la respuesta para identificar gaps de conocimiento"""
        try:
            # Analizar la respuesta de Gemini
            gemini_text = response.get('gemini_response', '')
            
            # Patrones que indican gaps de conocimiento
            gap_indicators = [
                'no estoy seguro', 'no tengo información', 'no conozco',
                'no sé', 'no puedo', 'no tengo datos', 'no está claro',
                'requiere más investigación', 'necesita verificación'
            ]
            
            gaps_found = []
            for indicator in gap_indicators:
                if indicator.lower() in gemini_text.lower():
                    gaps_found.append(indicator)
            
            if gaps_found:
                gap = {
                    'query': response['original_query'],
                    'indicators': gaps_found,
                    'timestamp': datetime.now(),
                    'category': response['analysis']['category']
                }
                session.knowledge_gaps.append(gap)
                
                # Generar recomendación para llenar el gap
                recommendation = self._generate_gap_recommendation(gap)
                session.recommendations.append(recommendation)
            
        except Exception as e:
            logger.error(f"Error analizando gaps de conocimiento: {e}")
    
    def _generate_gap_recommendation(self, gap: Dict[str, Any]) -> str:
        """Genera una recomendación para llenar un gap de conocimiento"""
        category = gap['category']
        query = gap['query']
        
        recommendations = {
            'authentication': f"Para profundizar en autenticación, investiga sobre OAuth 2.0, SAML, JWT y autenticación multifactor relacionado con: {query}",
            'encryption': f"Para mejorar conocimientos en encriptación, estudia algoritmos AES, RSA y protocolos SSL/TLS en el contexto de: {query}",
            'malware': f"Para entender mejor malware, investiga técnicas de detección, sandboxing y análisis forense relacionado con: {query}",
            'phishing': f"Para combatir phishing, estudia SPF, DKIM, DMARC y técnicas de educación del usuario en: {query}",
            'firewall': f"Para configurar firewalls efectivamente, aprende sobre NGFW, reglas de filtrado y segmentación de red en: {query}",
            'vulnerability': f"Para gestionar vulnerabilidades, estudia CVE, CVSS, parcheo y herramientas de escaneo en: {query}"
        }
        
        return recommendations.get(category, f"Investiga más sobre {category} en el contexto de: {query}")
    
    async def _generate_learning_insights(self, session: LearningSession):
        """Genera insights de aprendizaje para la sesión"""
        try:
            if len(session.queries) < 2:
                return
            
            # Analizar patrones en las consultas
            patterns = self._analyze_query_patterns(session)
            
            # Generar insights basados en patrones
            for pattern in patterns:
                insight = LearningInsight(
                    insight_id=self._generate_insight_id(),
                    session_id=session.session_id,
                    insight_type=pattern['type'],
                    content=pattern['content'],
                    confidence=pattern['confidence'],
                    evidence=pattern['evidence'],
                    timestamp=datetime.now(),
                    actionable=pattern['actionable']
                )
                
                self.learning_insights.append(insight)
                session.recommendations.append(insight.content)
            
        except Exception as e:
            logger.error(f"Error generando insights de aprendizaje: {e}")
    
    def _analyze_query_patterns(self, session: LearningSession) -> List[Dict[str, Any]]:
        """Analiza patrones en las consultas de la sesión"""
        patterns = []
        
        if len(session.queries) < 2:
            return patterns
        
        # Patrón: Progresión de complejidad
        complexity_progression = self._analyze_complexity_progression(session.queries)
        if complexity_progression:
            patterns.append(complexity_progression)
        
        # Patrón: Transiciones de tema
        topic_transitions = self._analyze_topic_transitions(session.queries)
        if topic_transitions:
            patterns.append(topic_transitions)
        
        # Patrón: Gaps de conocimiento
        knowledge_gaps = self._analyze_knowledge_gaps(session)
        if knowledge_gaps:
            patterns.append(knowledge_gaps)
        
        # Patrón: Desarrollo de habilidades
        skill_development = self._analyze_skill_development(session.queries)
        if skill_development:
            patterns.append(skill_development)
        
        return patterns
    
    def _analyze_complexity_progression(self, queries: List[QueryAnalysis]) -> Optional[Dict[str, Any]]:
        """Analiza la progresión de complejidad en las consultas"""
        if len(queries) < 3:
            return None
        
        complexity_scores = {
            QueryComplexity.BASIC: 1,
            QueryComplexity.INTERMEDIATE: 2,
            QueryComplexity.ADVANCED: 3,
            QueryComplexity.EXPERT: 4
        }
        
        scores = [complexity_scores[q.complexity] for q in queries]
        
        # Calcular tendencia
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            early_avg = sum(scores[:3]) / 3
            
            if recent_avg > early_avg + 0.5:
                return {
                    'type': 'progressive_complexity',
                    'content': f"El usuario está progresando hacia consultas más complejas (de {early_avg:.1f} a {recent_avg:.1f})",
                    'confidence': 0.8,
                    'evidence': [f"Complejidad actual: {scores[-1]}", f"Promedio reciente: {recent_avg:.1f}"],
                    'actionable': True
                }
        
        return None
    
    def _analyze_topic_transitions(self, queries: List[QueryAnalysis]) -> Optional[Dict[str, Any]]:
        """Analiza las transiciones entre temas"""
        if len(queries) < 2:
            return None
        
        categories = [q.category.value for q in queries]
        unique_categories = len(set(categories))
        
        if unique_categories > 1:
            transitions = []
            for i in range(1, len(categories)):
                if categories[i] != categories[i-1]:
                    transitions.append(f"{categories[i-1]} → {categories[i]}")
            
            if transitions:
                return {
                    'type': 'topic_transitions',
                    'content': f"El usuario está explorando múltiples temas: {', '.join(transitions)}",
                    'confidence': 0.9,
                    'evidence': transitions,
                    'actionable': True
                }
        
        return None
    
    def _analyze_knowledge_gaps(self, session: LearningSession) -> Optional[Dict[str, Any]]:
        """Analiza los gaps de conocimiento identificados"""
        if not session.knowledge_gaps:
            return None
        
        gap_categories = [gap['category'] for gap in session.knowledge_gaps]
        most_common_gap = max(set(gap_categories), key=gap_categories.count)
        
        return {
            'type': 'knowledge_gaps',
            'content': f"Se identificaron gaps de conocimiento en {most_common_gap} ({len(session.knowledge_gaps)} gaps)",
            'confidence': 0.7,
            'evidence': [gap['query'] for gap in session.knowledge_gaps[:3]],
            'actionable': True
        }
    
    def _analyze_skill_development(self, queries: List[QueryAnalysis]) -> Optional[Dict[str, Any]]:
        """Analiza el desarrollo de habilidades"""
        if len(queries) < 3:
            return None
        
        # Analizar evolución del potencial de aprendizaje
        learning_potentials = [q.learning_potential for q in queries]
        
        if len(learning_potentials) >= 3:
            recent_avg = sum(learning_potentials[-3:]) / 3
            early_avg = sum(learning_potentials[:3]) / 3
            
            if recent_avg > early_avg + 0.1:
                return {
                    'type': 'skill_development',
                    'content': f"El usuario está desarrollando habilidades más profundas (potencial de aprendizaje: {early_avg:.2f} → {recent_avg:.2f})",
                    'confidence': 0.6,
                    'evidence': [f"Potencial actual: {learning_potentials[-1]:.2f}", f"Promedio reciente: {recent_avg:.2f}"],
                    'actionable': True
                }
        
        return None
    
    def _should_generate_insights(self, session: LearningSession) -> bool:
        """Determina si se deben generar insights"""
        return (
            len(session.queries) >= 3 and
            len(session.queries) % 2 == 0 and  # Cada 2 consultas
            session.status == 'active'
        )
    
    def _should_end_session(self, session: LearningSession) -> bool:
        """Determina si la sesión debe terminar"""
        duration = (datetime.now() - session.start_time).total_seconds()
        
        return (
            duration > self.learning_config['max_session_duration'] or
            len(session.queries) > 20 or  # Máximo 20 consultas por sesión
            session.status != 'active'
        )
    
    async def _end_learning_session(self, session_id: str):
        """Termina una sesión de aprendizaje"""
        try:
            if session_id not in self.active_sessions:
                return
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.status = 'completed'
            
            # Calcular métricas finales
            self._calculate_final_metrics(session)
            
            # Mover a sesiones completadas
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            # Actualizar métricas globales
            self._update_global_metrics()
            
            logger.info(f"Sesión de aprendizaje completada: {session_id}")
            
        except Exception as e:
            logger.error(f"Error terminando sesión {session_id}: {e}")
    
    def _calculate_final_metrics(self, session: LearningSession):
        """Calcula las métricas finales de una sesión"""
        duration = (session.end_time - session.start_time).total_seconds()
        
        session.progress_metrics = {
            'total_queries': len(session.queries),
            'session_duration': duration,
            'average_complexity': sum(1 if q.complexity in [QueryComplexity.ADVANCED, QueryComplexity.EXPERT] else 0 
                                    for q in session.queries) / len(session.queries),
            'knowledge_gaps_identified': len(session.knowledge_gaps),
            'insights_generated': len([i for i in self.learning_insights if i.session_id == session.session_id]),
            'learning_efficiency': self._calculate_learning_efficiency(session),
            'objectives_achieved': self._calculate_objectives_achievement(session)
        }
    
    def _calculate_learning_efficiency(self, session: LearningSession) -> float:
        """Calcula la eficiencia de aprendizaje de la sesión"""
        if not session.queries:
            return 0.0
        
        # Factor 1: Progresión de complejidad
        complexity_scores = {
            QueryComplexity.BASIC: 1,
            QueryComplexity.INTERMEDIATE: 2,
            QueryComplexity.ADVANCED: 3,
            QueryComplexity.EXPERT: 4
        }
        
        scores = [complexity_scores[q.complexity] for q in session.queries]
        complexity_progression = (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0
        
        # Factor 2: Potencial de aprendizaje promedio
        avg_learning_potential = sum(q.learning_potential for q in session.queries) / len(session.queries)
        
        # Factor 3: Diversidad de temas
        unique_categories = len(set(q.category.value for q in session.queries))
        topic_diversity = unique_categories / len(session.queries)
        
        # Calcular eficiencia (0-1)
        efficiency = (complexity_progression * 0.4 + avg_learning_potential * 0.4 + topic_diversity * 0.2)
        return min(efficiency, 1.0)
    
    def _calculate_objectives_achievement(self, session: LearningSession) -> float:
        """Calcula el porcentaje de objetivos logrados"""
        if not session.learning_objectives:
            return 1.0
        
        # Análisis simple basado en categorías cubiertas
        covered_categories = set(q.category.value for q in session.queries)
        objective_categories = set()
        
        for objective in session.learning_objectives:
            for category in QueryCategory:
                if category.value in objective.lower():
                    objective_categories.add(category.value)
        
        if not objective_categories:
            return 1.0
        
        achievement = len(covered_categories.intersection(objective_categories)) / len(objective_categories)
        return min(achievement, 1.0)
    
    def _update_session_progress(self, session: LearningSession):
        """Actualiza el progreso de la sesión"""
        session.progress_metrics.update({
            'queries_processed': len(session.queries),
            'session_duration': (datetime.now() - session.start_time).total_seconds(),
            'knowledge_gaps': len(session.knowledge_gaps),
            'recommendations': len(session.recommendations)
        })
    
    def _update_global_metrics(self):
        """Actualiza las métricas globales del sistema"""
        total_sessions = len(self.completed_sessions)
        if total_sessions == 0:
            return
        
        # Métricas básicas
        self.performance_metrics['total_sessions'] = total_sessions
        self.performance_metrics['total_queries'] = sum(len(s.queries) for s in self.completed_sessions)
        
        # Duración promedio
        durations = [s.progress_metrics.get('session_duration', 0) for s in self.completed_sessions]
        self.performance_metrics['average_session_duration'] = sum(durations) / len(durations)
        
        # Eficiencia de aprendizaje
        efficiencies = [s.progress_metrics.get('learning_efficiency', 0) for s in self.completed_sessions]
        self.performance_metrics['learning_efficiency'] = sum(efficiencies) / len(efficiencies)
        
        # Tasa de generación de insights
        total_insights = len(self.learning_insights)
        self.performance_metrics['insight_generation_rate'] = total_insights / total_sessions
    
    def _generate_session_id(self) -> str:
        """Genera un ID único para la sesión"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_hash}"
    
    def _generate_insight_id(self) -> str:
        """Genera un ID único para un insight"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"insight_{timestamp}_{random_hash}"
    
    def _generate_learning_objectives(self, analysis: QueryAnalysis) -> List[str]:
        """Genera objetivos de aprendizaje basados en el análisis"""
        objectives = []
        
        category = analysis.category.value
        complexity = analysis.complexity.value
        
        base_objectives = {
            'authentication': [
                f"Dominar conceptos de {complexity} en autenticación",
                "Implementar sistemas de autenticación seguros",
                "Entender vulnerabilidades comunes en autenticación"
            ],
            'encryption': [
                f"Profundizar en {complexity} de encriptación",
                "Implementar algoritmos de cifrado apropiados",
                "Gestionar claves y certificados de forma segura"
            ],
            'malware': [
                f"Desarrollar habilidades {complexity} en detección de malware",
                "Implementar sistemas de prevención",
                "Realizar análisis forense básico"
            ]
        }
        
        objectives.extend(base_objectives.get(category, [
            f"Mejorar conocimientos en {category}",
            f"Desarrollar habilidades {complexity} en ciberseguridad"
        ]))
        
        return objectives[:3]  # Máximo 3 objetivos
    
    def _get_recent_insights(self, session_id: str) -> List[Dict[str, Any]]:
        """Obtiene insights recientes para una sesión"""
        session_insights = [i for i in self.learning_insights if i.session_id == session_id]
        return [
            {
                'type': i.insight_type,
                'content': i.content,
                'confidence': i.confidence,
                'actionable': i.actionable,
                'timestamp': i.timestamp.isoformat()
            }
            for i in session_insights[-5:]  # Últimos 5 insights
        ]
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Obtiene analytics completos del aprendizaje"""
        return {
            'performance_metrics': self.performance_metrics,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'total_insights': len(self.learning_insights),
            'recent_insights': [
                {
                    'type': i.insight_type,
                    'content': i.content,
                    'confidence': i.confidence,
                    'timestamp': i.timestamp.isoformat()
                }
                for i in self.learning_insights[-10:]  # Últimos 10 insights
            ],
            'learning_patterns': self.learning_patterns,
            'query_analyzer_insights': self.query_analyzer.get_learning_insights()
        }
    
    def save_learning_data(self, filepath: str = "data/deep_learning_data.json"):
        """Guarda los datos de aprendizaje"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data = {
                'performance_metrics': self.performance_metrics,
                'completed_sessions': [
                    {
                        'session_id': s.session_id,
                        'start_time': s.start_time.isoformat(),
                        'end_time': s.end_time.isoformat() if s.end_time else None,
                        'queries_count': len(s.queries),
                        'learning_objectives': s.learning_objectives,
                        'progress_metrics': s.progress_metrics,
                        'knowledge_gaps_count': len(s.knowledge_gaps),
                        'recommendations_count': len(s.recommendations)
                    }
                    for s in self.completed_sessions
                ],
                'insights': [
                    {
                        'insight_id': i.insight_id,
                        'session_id': i.session_id,
                        'insight_type': i.insight_type,
                        'content': i.content,
                        'confidence': i.confidence,
                        'timestamp': i.timestamp.isoformat(),
                        'actionable': i.actionable
                    }
                    for i in self.learning_insights
                ],
                'learning_patterns': self.learning_patterns
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Datos de aprendizaje guardados en {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")

# Instancia global del motor de aprendizaje profundo
deep_learning_engine = DeepLearningEngine()
