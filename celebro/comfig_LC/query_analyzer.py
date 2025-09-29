#!/usr/bin/env python3
"""
Analizador Profundo de Consultas - @red_neuronal
Versión: 0.6.0
Sistema avanzado para analizar y procesar consultas de entrenamiento con Gemini
"""

import numpy as np
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
import os

logger = logging.getLogger('Neural_QueryAnalyzer')

class QueryComplexity(Enum):
    """Niveles de complejidad de consultas"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class QueryCategory(Enum):
    """Categorías de consultas de ciberseguridad"""
    AUTHENTICATION = "authentication"
    ENCRYPTION = "encryption"
    MALWARE = "malware"
    PHISHING = "phishing"
    FIREWALL = "firewall"
    VULNERABILITY = "vulnerability"
    NETWORK_SECURITY = "network_security"
    INCIDENT_RESPONSE = "incident_response"
    COMPLIANCE = "compliance"
    CODE_SECURITY = "code_security"

@dataclass
class QueryAnalysis:
    """Resultado del análisis de una consulta"""
    query_id: str
    original_query: str
    complexity: QueryComplexity
    category: QueryCategory
    keywords: List[str]
    entities: List[str]
    intent: str
    context: Dict[str, Any]
    learning_potential: float
    suggested_prompts: List[str]
    timestamp: datetime
    analysis_metadata: Dict[str, Any]

class DeepQueryAnalyzer:
    """
    Analizador profundo de consultas para entrenamiento con Gemini
    """
    
    def __init__(self):
        self.analysis_cache = {}
        self.learning_patterns = {}
        self.query_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'complexity_distribution': {},
            'category_distribution': {},
            'learning_improvements': []
        }
        
        # Patrones de análisis
        self.complexity_patterns = {
            QueryComplexity.BASIC: [
                r'\b(qué|what|how|como|cómo)\b',
                r'\b(explain|explica|explicar)\b',
                r'\b(define|define|definir)\b'
            ],
            QueryComplexity.INTERMEDIATE: [
                r'\b(implement|implementar|implementación)\b',
                r'\b(configure|configurar|configuración)\b',
                r'\b(secure|seguro|seguridad)\b'
            ],
            QueryComplexity.ADVANCED: [
                r'\b(architecture|arquitectura|diseño)\b',
                r'\b(optimize|optimizar|optimización)\b',
                r'\b(penetration|penetración|testing)\b'
            ],
            QueryComplexity.EXPERT: [
                r'\b(exploit|exploit|vulnerabilidad)\b',
                r'\b(forensics|forense|análisis)\b',
                r'\b(reverse|ingeniería|inversa)\b'
            ]
        }
        
        self.category_patterns = {
            QueryCategory.AUTHENTICATION: [
                r'\b(autenticación|authentication|login|password|2fa|mfa)\b',
                r'\b(oauth|saml|jwt|token)\b',
                r'\b(multi-factor|dos factores|biometric)\b'
            ],
            QueryCategory.ENCRYPTION: [
                r'\b(encriptación|encryption|cifrado|cipher)\b',
                r'\b(aes|rsa|ssl|tls|https)\b',
                r'\b(key|clave|hash|digest)\b'
            ],
            QueryCategory.MALWARE: [
                r'\b(malware|virus|ransomware|trojan)\b',
                r'\b(antivirus|sandbox|behavioral)\b',
                r'\b(detection|detección|prevention)\b'
            ],
            QueryCategory.PHISHING: [
                r'\b(phishing|estafa|fraude|scam)\b',
                r'\b(email|correo|spam|filter)\b',
                r'\b(spf|dkim|dmarc)\b'
            ],
            QueryCategory.FIREWALL: [
                r'\b(firewall|cortafuegos|iptables)\b',
                r'\b(ngfw|waf|network|red)\b',
                r'\b(rules|reglas|filtering)\b'
            ],
            QueryCategory.VULNERABILITY: [
                r'\b(vulnerabilidad|vulnerability|cve)\b',
                r'\b(exploit|exploit|poc)\b',
                r'\b(patch|parche|update)\b'
            ],
            QueryCategory.NETWORK_SECURITY: [
                r'\b(network|red|seguridad|security)\b',
                r'\b(ids|ips|monitoring|monitoreo)\b',
                r'\b(segmentation|segmentación|vlan)\b'
            ],
            QueryCategory.INCIDENT_RESPONSE: [
                r'\b(incident|incidente|response|respuesta)\b',
                r'\b(forensics|forense|investigation)\b',
                r'\b(containment|contención|recovery)\b'
            ],
            QueryCategory.COMPLIANCE: [
                r'\b(compliance|cumplimiento|gdpr|sox)\b',
                r'\b(audit|auditoría|control)\b',
                r'\b(policy|política|governance)\b'
            ],
            QueryCategory.CODE_SECURITY: [
                r'\b(code|código|secure|seguro)\b',
                r'\b(owasp|sast|dast|sca)\b',
                r'\b(devsecops|ci/cd|pipeline)\b'
            ]
        }
        
        # Entidades de ciberseguridad
        self.security_entities = [
            'API', 'HTTPS', 'SSL', 'TLS', 'AES', 'RSA', 'SHA', 'MD5',
            'JWT', 'OAuth', 'SAML', 'LDAP', 'Kerberos', 'PKI',
            'WAF', 'IDS', 'IPS', 'SIEM', 'SOC', 'NOC',
            'CVE', 'CVSS', 'OWASP', 'NIST', 'ISO', 'PCI-DSS',
            'GDPR', 'SOX', 'HIPAA', 'FISMA', 'COBIT',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
            'Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'
        ]
        
        logger.info("Analizador profundo de consultas inicializado")
    
    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """
        Analiza una consulta de forma profunda
        
        Args:
            query: Consulta a analizar
            context: Contexto adicional
            
        Returns:
            Análisis detallado de la consulta
        """
        try:
            # Generar ID único para la consulta
            query_id = self._generate_query_id(query)
            
            # Verificar cache
            if query_id in self.analysis_cache:
                cached_analysis = self.analysis_cache[query_id]
                logger.debug(f"Consulta encontrada en cache: {query_id}")
                return cached_analysis
            
            # Análisis de complejidad
            complexity = self._analyze_complexity(query)
            
            # Análisis de categoría
            category = self._analyze_category(query)
            
            # Extracción de keywords
            keywords = self._extract_keywords(query)
            
            # Extracción de entidades
            entities = self._extract_entities(query)
            
            # Análisis de intención
            intent = self._analyze_intent(query, complexity, category)
            
            # Análisis de contexto
            context_analysis = self._analyze_context(query, context or {})
            
            # Potencial de aprendizaje
            learning_potential = self._calculate_learning_potential(
                query, complexity, category, keywords, entities
            )
            
            # Generar prompts sugeridos
            suggested_prompts = self._generate_suggested_prompts(
                query, complexity, category, intent
            )
            
            # Crear análisis completo
            analysis = QueryAnalysis(
                query_id=query_id,
                original_query=query,
                complexity=complexity,
                category=category,
                keywords=keywords,
                entities=entities,
                intent=intent,
                context=context_analysis,
                learning_potential=learning_potential,
                suggested_prompts=suggested_prompts,
                timestamp=datetime.now(),
                analysis_metadata={
                    'word_count': len(query.split()),
                    'character_count': len(query),
                    'language': self._detect_language(query),
                    'technical_terms': len([w for w in keywords if w.lower() in [e.lower() for e in self.security_entities]]),
                    'question_type': self._classify_question_type(query)
                }
            )
            
            # Guardar en cache
            self.analysis_cache[query_id] = analysis
            
            # Actualizar métricas
            self._update_metrics(analysis)
            
            # Agregar a historial
            self.query_history.append(analysis)
            
            logger.info(f"Consulta analizada: {query_id} - {complexity.value} - {category.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando consulta: {e}")
            # Retornar análisis básico en caso de error
            return self._create_basic_analysis(query)
    
    def _generate_query_id(self, query: str) -> str:
        """Genera un ID único para la consulta"""
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"query_{timestamp}_{query_hash}"
    
    def _analyze_complexity(self, query: str) -> QueryComplexity:
        """Analiza la complejidad de la consulta"""
        query_lower = query.lower()
        scores = {}
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            scores[complexity] = score
        
        # Determinar complejidad basada en puntuación
        if scores[QueryComplexity.EXPERT] > 0:
            return QueryComplexity.EXPERT
        elif scores[QueryComplexity.ADVANCED] > 0:
            return QueryComplexity.ADVANCED
        elif scores[QueryComplexity.INTERMEDIATE] > 0:
            return QueryComplexity.INTERMEDIATE
        else:
            return QueryComplexity.BASIC
    
    def _analyze_category(self, query: str) -> QueryCategory:
        """Analiza la categoría de la consulta"""
        query_lower = query.lower()
        scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            scores[category] = score
        
        # Retornar categoría con mayor puntuación
        if scores:
            return max(scores, key=scores.get)
        else:
            return QueryCategory.CODE_SECURITY  # Categoría por defecto
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extrae palabras clave de la consulta"""
        # Limpiar y tokenizar
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filtrar palabras comunes
        stop_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'sido', 'estado', 'estaba', 'están', 'como', 'más', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'sido', 'estado', 'estaba', 'están'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Agregar términos técnicos encontrados
        technical_terms = [term for term in self.security_entities if term.lower() in query.lower()]
        keywords.extend(technical_terms)
        
        return list(set(keywords))  # Eliminar duplicados
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extrae entidades de ciberseguridad de la consulta"""
        entities = []
        query_upper = query.upper()
        
        for entity in self.security_entities:
            if entity.upper() in query_upper:
                entities.append(entity)
        
        return entities
    
    def _analyze_intent(self, query: str, complexity: QueryComplexity, category: QueryCategory) -> str:
        """Analiza la intención de la consulta"""
        query_lower = query.lower()
        
        # Patrones de intención
        if any(word in query_lower for word in ['qué', 'what', 'cómo', 'how', 'por qué', 'why']):
            return "explanatory"
        elif any(word in query_lower for word in ['cómo', 'how', 'implementar', 'implement', 'crear', 'create']):
            return "instructional"
        elif any(word in query_lower for word in ['problema', 'problem', 'error', 'fallo', 'issue']):
            return "troubleshooting"
        elif any(word in query_lower for word in ['mejor', 'best', 'recomendar', 'recommend', 'sugerir', 'suggest']):
            return "recommendation"
        elif any(word in query_lower for word in ['comparar', 'compare', 'vs', 'versus', 'diferencia', 'difference']):
            return "comparison"
        else:
            return "general"
    
    def _analyze_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el contexto de la consulta"""
        context_analysis = {
            'has_previous_queries': len(self.query_history) > 0,
            'query_sequence': len(self.query_history),
            'user_experience_level': self._estimate_user_level(),
            'related_topics': self._find_related_topics(query),
            'temporal_context': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_data': context
        }
        
        return context_analysis
    
    def _calculate_learning_potential(self, query: str, complexity: QueryComplexity, 
                                    category: QueryCategory, keywords: List[str], 
                                    entities: List[str]) -> float:
        """Calcula el potencial de aprendizaje de la consulta"""
        score = 0.0
        
        # Puntuación por complejidad
        complexity_scores = {
            QueryComplexity.BASIC: 0.2,
            QueryComplexity.INTERMEDIATE: 0.5,
            QueryComplexity.ADVANCED: 0.8,
            QueryComplexity.EXPERT: 1.0
        }
        score += complexity_scores[complexity]
        
        # Puntuación por términos técnicos
        technical_terms = len([w for w in keywords if w.lower() in [e.lower() for e in self.security_entities]])
        score += min(technical_terms * 0.1, 0.3)
        
        # Puntuación por entidades
        score += min(len(entities) * 0.05, 0.2)
        
        # Puntuación por longitud de consulta
        word_count = len(query.split())
        if word_count > 10:
            score += 0.1
        if word_count > 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_suggested_prompts(self, query: str, complexity: QueryComplexity, 
                                  category: QueryCategory, intent: str) -> List[str]:
        """Genera prompts sugeridos para mejorar el aprendizaje"""
        prompts = []
        
        # Prompt base
        base_prompt = f"Como experto en ciberseguridad, analiza esta consulta sobre {category.value}: {query}"
        
        # Prompts específicos por complejidad
        if complexity == QueryComplexity.BASIC:
            prompts.append(f"Explica de manera simple y clara: {query}")
            prompts.append(f"¿Cuáles son los conceptos básicos relacionados con {category.value}?")
        elif complexity == QueryComplexity.INTERMEDIATE:
            prompts.append(f"Proporciona una guía paso a paso para: {query}")
            prompts.append(f"¿Cuáles son las mejores prácticas en {category.value}?")
        elif complexity == QueryComplexity.ADVANCED:
            prompts.append(f"Analiza en profundidad los aspectos técnicos de: {query}")
            prompts.append(f"¿Cuáles son los desafíos avanzados en {category.value}?")
        else:  # EXPERT
            prompts.append(f"Realiza un análisis forense completo de: {query}")
            prompts.append(f"¿Cuáles son las técnicas de ataque y defensa en {category.value}?")
        
        # Prompts específicos por intención
        if intent == "instructional":
            prompts.append(f"Crea un tutorial detallado sobre: {query}")
        elif intent == "troubleshooting":
            prompts.append(f"Diagnostica y resuelve el problema: {query}")
        elif intent == "recommendation":
            prompts.append(f"Recomienda las mejores soluciones para: {query}")
        
        return prompts[:5]  # Limitar a 5 prompts
    
    def _detect_language(self, query: str) -> str:
        """Detecta el idioma de la consulta"""
        spanish_words = ['qué', 'cómo', 'por', 'para', 'con', 'del', 'los', 'las', 'una', 'como']
        english_words = ['what', 'how', 'for', 'with', 'the', 'and', 'or', 'but', 'in', 'on']
        
        spanish_count = sum(1 for word in spanish_words if word in query.lower())
        english_count = sum(1 for word in english_words if word in query.lower())
        
        return 'spanish' if spanish_count > english_count else 'english'
    
    def _classify_question_type(self, query: str) -> str:
        """Clasifica el tipo de pregunta"""
        query_lower = query.lower()
        
        if query_lower.startswith(('qué', 'what')):
            return 'definition'
        elif query_lower.startswith(('cómo', 'how')):
            return 'procedure'
        elif query_lower.startswith(('por qué', 'why')):
            return 'explanation'
        elif query_lower.startswith(('cuándo', 'when')):
            return 'temporal'
        elif query_lower.startswith(('dónde', 'where')):
            return 'location'
        elif query_lower.startswith(('quién', 'who')):
            return 'person'
        else:
            return 'general'
    
    def _estimate_user_level(self) -> str:
        """Estima el nivel de experiencia del usuario"""
        if not self.query_history:
            return 'beginner'
        
        recent_queries = self.query_history[-5:]  # Últimas 5 consultas
        avg_complexity = sum(1 if q.complexity in [QueryComplexity.ADVANCED, QueryComplexity.EXPERT] else 0 
                           for q in recent_queries) / len(recent_queries)
        
        if avg_complexity > 0.6:
            return 'expert'
        elif avg_complexity > 0.3:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _find_related_topics(self, query: str) -> List[str]:
        """Encuentra temas relacionados basados en el historial"""
        if not self.query_history:
            return []
        
        current_keywords = set(self._extract_keywords(query))
        related_topics = []
        
        for past_query in self.query_history[-10:]:  # Últimas 10 consultas
            past_keywords = set(past_query.keywords)
            overlap = current_keywords.intersection(past_keywords)
            if len(overlap) > 0:
                related_topics.append(past_query.category.value)
        
        return list(set(related_topics))
    
    def _update_metrics(self, analysis: QueryAnalysis):
        """Actualiza las métricas de rendimiento"""
        self.performance_metrics['total_queries'] += 1
        
        # Distribución de complejidad
        complexity = analysis.complexity.value
        if complexity not in self.performance_metrics['complexity_distribution']:
            self.performance_metrics['complexity_distribution'][complexity] = 0
        self.performance_metrics['complexity_distribution'][complexity] += 1
        
        # Distribución de categorías
        category = analysis.category.value
        if category not in self.performance_metrics['category_distribution']:
            self.performance_metrics['category_distribution'][category] = 0
        self.performance_metrics['category_distribution'][category] += 1
    
    def _create_basic_analysis(self, query: str) -> QueryAnalysis:
        """Crea un análisis básico en caso de error"""
        return QueryAnalysis(
            query_id=self._generate_query_id(query),
            original_query=query,
            complexity=QueryComplexity.BASIC,
            category=QueryCategory.CODE_SECURITY,
            keywords=self._extract_keywords(query),
            entities=self._extract_entities(query),
            intent="general",
            context={},
            learning_potential=0.5,
            suggested_prompts=[f"Analiza esta consulta: {query}"],
            timestamp=datetime.now(),
            analysis_metadata={'error': True}
        )
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Obtiene insights sobre el aprendizaje"""
        if not self.query_history:
            return {'message': 'No hay historial de consultas'}
        
        recent_queries = self.query_history[-20:]  # Últimas 20 consultas
        
        insights = {
            'total_queries': len(self.query_history),
            'recent_queries': len(recent_queries),
            'complexity_trend': self._calculate_complexity_trend(),
            'category_preferences': self.performance_metrics['category_distribution'],
            'learning_progression': self._calculate_learning_progression(),
            'recommended_focus_areas': self._get_recommended_focus_areas(),
            'user_level_evolution': self._track_user_level_evolution()
        }
        
        return insights
    
    def _calculate_complexity_trend(self) -> Dict[str, Any]:
        """Calcula la tendencia de complejidad"""
        if len(self.query_history) < 5:
            return {'trend': 'insufficient_data'}
        
        recent = self.query_history[-10:]
        complexity_scores = {
            QueryComplexity.BASIC: 1,
            QueryComplexity.INTERMEDIATE: 2,
            QueryComplexity.ADVANCED: 3,
            QueryComplexity.EXPERT: 4
        }
        
        scores = [complexity_scores[q.complexity] for q in recent]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 3:
            return {'trend': 'increasing', 'level': 'expert'}
        elif avg_score > 2:
            return {'trend': 'stable', 'level': 'intermediate'}
        else:
            return {'trend': 'beginner', 'level': 'basic'}
    
    def _calculate_learning_progression(self) -> Dict[str, Any]:
        """Calcula la progresión del aprendizaje"""
        if len(self.query_history) < 10:
            return {'progression': 'insufficient_data'}
        
        # Dividir en períodos
        total = len(self.query_history)
        first_half = self.query_history[:total//2]
        second_half = self.query_history[total//2:]
        
        first_avg_potential = sum(q.learning_potential for q in first_half) / len(first_half)
        second_avg_potential = sum(q.learning_potential for q in second_half) / len(second_half)
        
        improvement = second_avg_potential - first_avg_potential
        
        return {
            'progression': 'improving' if improvement > 0.1 else 'stable',
            'improvement_rate': improvement,
            'current_level': second_avg_potential
        }
    
    def _get_recommended_focus_areas(self) -> List[str]:
        """Obtiene áreas de enfoque recomendadas"""
        if not self.query_history:
            return []
        
        category_counts = {}
        for query in self.query_history[-20:]:
            category = query.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Ordenar por frecuencia (menos consultadas = más recomendadas)
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1])
        return [cat[0] for cat in sorted_categories[:3]]
    
    def _track_user_level_evolution(self) -> Dict[str, Any]:
        """Rastrea la evolución del nivel del usuario"""
        if len(self.query_history) < 5:
            return {'evolution': 'insufficient_data'}
        
        # Dividir en períodos
        periods = 3
        period_size = len(self.query_history) // periods
        period_levels = []
        
        for i in range(periods):
            start = i * period_size
            end = start + period_size if i < periods - 1 else len(self.query_history)
            period_queries = self.query_history[start:end]
            
            if period_queries:
                avg_complexity = sum(1 if q.complexity in [QueryComplexity.ADVANCED, QueryComplexity.EXPERT] else 0 
                                   for q in period_queries) / len(period_queries)
                period_levels.append(avg_complexity)
        
        return {
            'evolution': 'improving' if period_levels[-1] > period_levels[0] else 'stable',
            'period_levels': period_levels,
            'current_level': period_levels[-1] if period_levels else 0
        }
    
    def save_analysis_data(self, filepath: str = "data/query_analysis.json"):
        """Guarda los datos de análisis"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            data = {
                'analysis_cache': {k: {
                    'query_id': v.query_id,
                    'original_query': v.original_query,
                    'complexity': v.complexity.value,
                    'category': v.category.value,
                    'keywords': v.keywords,
                    'entities': v.entities,
                    'intent': v.intent,
                    'learning_potential': v.learning_potential,
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.analysis_cache.items()},
                'performance_metrics': self.performance_metrics,
                'query_history_count': len(self.query_history)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Datos de análisis guardados en {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando datos de análisis: {e}")
    
    def load_analysis_data(self, filepath: str = "data/query_analysis.json"):
        """Carga los datos de análisis"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Restaurar cache
                for k, v in data.get('analysis_cache', {}).items():
                    analysis = QueryAnalysis(
                        query_id=v['query_id'],
                        original_query=v['original_query'],
                        complexity=QueryComplexity(v['complexity']),
                        category=QueryCategory(v['category']),
                        keywords=v['keywords'],
                        entities=v['entities'],
                        intent=v['intent'],
                        context={},
                        learning_potential=v['learning_potential'],
                        suggested_prompts=[],
                        timestamp=datetime.fromisoformat(v['timestamp']),
                        analysis_metadata={}
                    )
                    self.analysis_cache[k] = analysis
                
                # Restaurar métricas
                self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
                
                logger.info(f"Datos de análisis cargados desde {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando datos de análisis: {e}")

# Instancia global del analizador
query_analyzer = DeepQueryAnalyzer()
