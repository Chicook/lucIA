"""
Currículum de Aprendizaje para @red_neuronal
Versión: 0.7.0
Sistema de currículum estructurado para aprendizaje profundo en ciberseguridad
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps

from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel, TopicCategory
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel
from .knowledge_base import KnowledgeBase, LearningSession

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningPath(Enum):
    """Rutas de aprendizaje disponibles"""
    FUNDAMENTOS = "fundamentos"
    DESARROLLADOR = "desarrollador"
    ADMINISTRADOR = "administrador"
    ANALISTA = "analista"
    AUDITOR = "auditor"
    COMPLETO = "completo"

class LearningPhase(Enum):
    """Fases del aprendizaje con orden jerárquico"""
    INTRODUCCION = ("introduccion", 1)
    CONCEPTOS = ("conceptos", 2)
    PRACTICA = ("practica", 3)
    APLICACION = ("aplicacion", 4)
    MAESTRIA = ("maestria", 5)
    
    def __init__(self, value, order):
        self.value = value
        self.order = order
    
    def __lt__(self, other):
        return self.order < other.order

class AssessmentType(Enum):
    """Tipos de evaluación"""
    QUIZ = "quiz"
    PRACTICA = "evaluacion_practica"
    PROYECTO = "proyecto_practico"
    SIMULACION = "simulacion_ataque"
    EXAMEN = "examen_completo"

@dataclass
class LearningObjective:
    """Objetivo de aprendizaje específico"""
    id: str
    description: str
    skill_type: str  # cognitive, procedural, affective
    difficulty: DifficultyLevel
    assessment_method: AssessmentType

@dataclass
class CurriculumModule:
    """Módulo del currículum mejorado"""
    id: str
    title: str
    description: str
    phase: LearningPhase
    topics: List[str]
    estimated_duration_hours: int
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIO
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if self.estimated_duration_hours <= 0:
            raise ValueError("La duración estimada debe ser positiva")
        if not self.topics:
            raise ValueError("El módulo debe tener al menos un tema")

@dataclass
class LearningMilestone:
    """Hito de aprendizaje mejorado"""
    id: str
    title: str
    description: str
    phase: LearningPhase
    required_topics: List[str]
    assessment_type: AssessmentType
    passing_score: float
    rewards: List[str] = field(default_factory=list)
    badge_icon: str = "🎖️"
    xp_points: int = 100
    
    def __post_init__(self):
        if not 0 <= self.passing_score <= 100:
            raise ValueError("El puntaje debe estar entre 0 y 100")

def cache_with_ttl(ttl_seconds: int = 3600):
    """Decorator para cache con TTL"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = datetime.now()
            
            if key in cache:
                result, timestamp = cache[key]
                if (current_time - timestamp).seconds < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        return wrapper
    return decorator

class ModuleBuilder:
    """Builder pattern para crear módulos"""
    
    def __init__(self, module_id: str, title: str):
        self.module = CurriculumModule(
            id=module_id,
            title=title,
            description="",
            phase=LearningPhase.INTRODUCCION,
            topics=[],
            estimated_duration_hours=1
        )
    
    def description(self, desc: str) -> 'ModuleBuilder':
        self.module.description = desc
        return self
    
    def phase(self, phase: LearningPhase) -> 'ModuleBuilder':
        self.module.phase = phase
        return self
    
    def topics(self, topics: List[str]) -> 'ModuleBuilder':
        self.module.topics = topics
        return self
    
    def duration(self, hours: int) -> 'ModuleBuilder':
        self.module.estimated_duration_hours = hours
        return self
    
    def prerequisites(self, prereqs: List[str]) -> 'ModuleBuilder':
        self.module.prerequisites = prereqs
        return self
    
    def objectives(self, objectives: List[LearningObjective]) -> 'ModuleBuilder':
        self.module.learning_objectives = objectives
        return self
    
    def resources(self, resources: List[str]) -> 'ModuleBuilder':
        self.module.resources = resources
        return self
    
    def tags(self, tags: Set[str]) -> 'ModuleBuilder':
        self.module.tags = tags
        return self
    
    def difficulty(self, level: DifficultyLevel) -> 'ModuleBuilder':
        self.module.difficulty_level = level
        return self
    
    def build(self) -> CurriculumModule:
        return self.module

class LearningCurriculum:
    """Sistema de currículum de aprendizaje estructurado y optimizado"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.security_topics = SecurityTopics()
        self.prompt_generator = PromptGenerator()
        self.knowledge_base = KnowledgeBase()
        
        # Cache para mejorar rendimiento
        self._modules_cache = {}
        self._paths_cache = {}
        self._milestones_cache = {}
        
        # Inicialización lazy loading
        self._modules_initialized = False
        self._milestones_initialized = False
        self._paths_initialized = False
        
        # Thread pool para operaciones paralelas
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def curriculum_modules(self) -> Dict[str, CurriculumModule]:
        """Lazy loading de módulos"""
        if not self._modules_initialized:
            self._modules_cache = self._initialize_modules()
            self._modules_initialized = True
        return self._modules_cache
    
    @property
    def learning_milestones(self) -> Dict[str, LearningMilestone]:
        """Lazy loading de hitos"""
        if not self._milestones_initialized:
            self._milestones_cache = self._initialize_milestones()
            self._milestones_initialized = True
        return self._milestones_cache
    
    @property
    def learning_paths(self) -> Dict[LearningPath, List[str]]:
        """Lazy loading de rutas"""
        if not self._paths_initialized:
            self._paths_cache = self._initialize_learning_paths()
            self._paths_initialized = True
        return self._paths_cache
    
    def _create_learning_objective(self, id: str, description: str, 
                                 skill_type: str = "cognitive",
                                 difficulty: DifficultyLevel = DifficultyLevel.MEDIO,
                                 assessment: AssessmentType = AssessmentType.QUIZ) -> LearningObjective:
        """Helper para crear objetivos de aprendizaje"""
        return LearningObjective(
            id=id,
            description=description,
            skill_type=skill_type,
            difficulty=difficulty,
            assessment_method=assessment
        )
    
    def _initialize_modules(self) -> Dict[str, CurriculumModule]:
        """Inicializa módulos usando el builder pattern"""
        modules = {}
        
        # MÓDULO 1: FUNDAMENTOS
        modules["fundamentos"] = (
            ModuleBuilder("fundamentos", "Fundamentos de Ciberseguridad")
            .description("Conceptos básicos y principios fundamentales de la ciberseguridad")
            .phase(LearningPhase.INTRODUCCION)
            .topics(["autenticacion", "encriptacion"])
            .duration(8)
            .objectives([
                self._create_learning_objective("fund_01", "Comprender conceptos básicos de ciberseguridad"),
                self._create_learning_objective("fund_02", "Identificar amenazas comunes", assessment=AssessmentType.PRACTICA),
                self._create_learning_objective("fund_03", "Aplicar principios de autenticación"),
                self._create_learning_objective("fund_04", "Implementar encriptación básica", skill_type="procedural")
            ])
            .resources(["OWASP Top 10", "NIST Cybersecurity Framework", "CIS Controls", "ISO 27001"])
            .tags({"fundamentos", "básico", "conceptos"})
            .difficulty(DifficultyLevel.BASICO)
            .build()
        )
        
        # MÓDULO 2: DESARROLLO SEGURO
        modules["desarrollo_seguro"] = (
            ModuleBuilder("desarrollo_seguro", "Desarrollo de Código Seguro")
            .description("Principios y prácticas para desarrollar aplicaciones seguras")
            .phase(LearningPhase.CONCEPTOS)
            .topics(["secure_coding", "web_security"])
            .duration(12)
            .prerequisites(["fundamentos"])
            .objectives([
                self._create_learning_objective("dev_01", "Aplicar principios de código seguro", skill_type="procedural"),
                self._create_learning_objective("dev_02", "Prevenir vulnerabilidades OWASP Top 10", assessment=AssessmentType.PROYECTO),
                self._create_learning_objective("dev_03", "Implementar validación de entrada"),
                self._create_learning_objective("dev_04", "Realizar code review de seguridad", difficulty=DifficultyLevel.INTERMEDIO)
            ])
            .resources(["OWASP Secure Coding Practices", "CERT Secure Coding Standards"])
            .tags({"desarrollo", "código", "web"})
            .difficulty(DifficultyLevel.INTERMEDIO)
            .build()
        )
        
        # MÓDULO 3: AMENAZAS Y VULNERABILIDADES
        modules["amenazas"] = (
            ModuleBuilder("amenazas", "Amenazas y Vulnerabilidades")
            .description("Identificación, análisis y mitigación de amenazas cibernéticas")
            .phase(LearningPhase.CONCEPTOS)
            .topics(["malware", "phishing"])
            .duration(10)
            .prerequisites(["fundamentos"])
            .objectives([
                self._create_learning_objective("amz_01", "Identificar tipos de malware"),
                self._create_learning_objective("amz_02", "Reconocer ataques de phishing", assessment=AssessmentType.SIMULACION),
                self._create_learning_objective("amz_03", "Implementar detección de amenazas", skill_type="procedural"),
                self._create_learning_objective("amz_04", "Desarrollar estrategias de prevención", difficulty=DifficultyLevel.AVANZADO)
            ])
            .resources(["MITRE ATT&CK Framework", "Malware Analysis Techniques"])
            .tags({"amenazas", "malware", "phishing"})
            .build()
        )
        
        # Continuar con otros módulos usando el mismo patrón...
        # (Por brevedad, incluyo solo algunos módulos. El patrón se repite)
        
        return modules
    
    def _initialize_milestones(self) -> Dict[str, LearningMilestone]:
        """Inicializa hitos con validación mejorada"""
        milestones = {}
        
        milestones["fundamentos_completados"] = LearningMilestone(
            id="fundamentos_completados",
            title="Fundamentos de Ciberseguridad Dominados",
            description="Has completado exitosamente los conceptos básicos de ciberseguridad",
            phase=LearningPhase.CONCEPTOS,
            required_topics=["autenticacion", "encriptacion"],
            assessment_type=AssessmentType.PRACTICA,
            passing_score=80.0,
            rewards=["Badge: Fundamentos de Ciberseguridad", "Acceso a módulos avanzados"],
            badge_icon="🛡️",
            xp_points=150
        )
        
        milestones["desarrollador_seguro"] = LearningMilestone(
            id="desarrollador_seguro",
            title="Desarrollador de Código Seguro",
            description="Has demostrado competencia en desarrollo de código seguro",
            phase=LearningPhase.PRACTICA,
            required_topics=["secure_coding", "web_security"],
            assessment_type=AssessmentType.PROYECTO,
            passing_score=85.0,
            rewards=["Certificado: Desarrollador Seguro", "Herramientas de desarrollo"],
            badge_icon="👨‍💻",
            xp_points=250
        )
        
        return milestones
    
    def _initialize_learning_paths(self) -> Dict[LearningPath, List[str]]:
        """Inicializa rutas de aprendizaje optimizadas"""
        return {
            LearningPath.FUNDAMENTOS: ["fundamentos"],
            LearningPath.DESARROLLADOR: ["fundamentos", "desarrollo_seguro"],
            LearningPath.ADMINISTRADOR: ["fundamentos", "defensas_red", "evaluacion_vuln"],
            LearningPath.ANALISTA: ["fundamentos", "amenazas", "ids_ips", "vulnerability_assessment"],
            LearningPath.AUDITOR: ["fundamentos", "evaluacion_vuln", "cumplimiento"],
            LearningPath.COMPLETO: ["fundamentos", "desarrollo_seguro", "amenazas", 
                                  "defensas_red", "evaluacion_vuln", "cumplimiento"]
        }
    
    @cache_with_ttl(1800)  # Cache por 30 minutos
    def get_learning_path(self, path: LearningPath) -> List[CurriculumModule]:
        """Obtiene módulos de una ruta con cache"""
        try:
            module_ids = self.learning_paths.get(path, [])
            modules = []
            
            for module_id in module_ids:
                if module_id in self.curriculum_modules:
                    modules.append(self.curriculum_modules[module_id])
                else:
                    logger.warning(f"Módulo no encontrado: {module_id}")
            
            # Ordenar por fase y prerequisitos
            return self._sort_modules_by_dependency(modules)
            
        except Exception as e:
            logger.error(f"Error obteniendo ruta de aprendizaje {path}: {e}")
            return []
    
    def _sort_modules_by_dependency(self, modules: List[CurriculumModule]) -> List[CurriculumModule]:
        """Ordena módulos por dependencias y fase"""
        sorted_modules = []
        remaining = modules.copy()
        
        while remaining:
            # Buscar módulos sin prerequisitos pendientes
            available = []
            for module in remaining:
                if all(prereq in [m.id for m in sorted_modules] for prereq in module.prerequisites):
                    available.append(module)
            
            if not available:
                # Si no hay módulos disponibles, hay dependencias circulares
                logger.warning("Posibles dependencias circulares detectadas")
                available = remaining  # Agregar todos los restantes
            
            # Ordenar por fase
            available.sort(key=lambda x: x.phase)
            
            # Agregar el primero disponible
            next_module = available[0]
            sorted_modules.append(next_module)
            remaining.remove(next_module)
        
        return sorted_modules
    
    def get_module(self, module_id: str) -> Optional[CurriculumModule]:
        """Obtiene un módulo específico con validación"""
        return self.curriculum_modules.get(module_id)
    
    def search_modules(self, query: str, tags: Set[str] = None, 
                      phase: LearningPhase = None, 
                      difficulty: DifficultyLevel = None) -> List[CurriculumModule]:
        """Búsqueda avanzada de módulos"""
        results = []
        query_lower = query.lower()
        
        for module in self.curriculum_modules.values():
            # Búsqueda por texto
            text_match = (query_lower in module.title.lower() or 
                         query_lower in module.description.lower())
            
            # Filtros adicionales
            tag_match = not tags or tags.intersection(module.tags)
            phase_match = not phase or module.phase == phase
            difficulty_match = not difficulty or module.difficulty_level == difficulty
            
            if text_match and tag_match and phase_match and difficulty_match:
                results.append(module)
        
        return results
    
    def get_available_modules(self, completed_modules: Set[str] = None) -> List[CurriculumModule]:
        """Obtiene módulos disponibles optimizado"""
        if completed_modules is None:
            completed_modules = set()
        
        available = []
        for module in self.curriculum_modules.values():
            if set(module.prerequisites).issubset(completed_modules):
                available.append(module)
        
        return self._sort_modules_by_dependency(available)
    
    def generate_learning_plan(self, path: LearningPath, 
                             user_level: SecurityLevel = SecurityLevel.BASICO,
                             target_weeks: int = None,
                             daily_hours: int = 2) -> Dict[str, Any]:
        """Genera plan de aprendizaje personalizado y optimizado"""
        try:
            modules = self.get_learning_path(path)
            
            # Filtrar por nivel usando comprehension
            phase_filters = {
                SecurityLevel.BASICO: {LearningPhase.INTRODUCCION, LearningPhase.CONCEPTOS},
                SecurityLevel.INTERMEDIO: {LearningPhase.CONCEPTOS, LearningPhase.PRACTICA},
                SecurityLevel.AVANZADO: {LearningPhase.PRACTICA, LearningPhase.APLICACION},
                SecurityLevel.EXPERTO: set(LearningPhase)
            }
            
            allowed_phases = phase_filters.get(user_level, set(LearningPhase))
            filtered_modules = [m for m in modules if m.phase in allowed_phases]
            
            # Calcular duración y cronograma optimizado
            total_duration = sum(m.estimated_duration_hours for m in filtered_modules)
            
            if target_weeks:
                weekly_hours = total_duration / target_weeks
            else:
                weekly_hours = daily_hours * 7
                target_weeks = max(1, int(total_duration / weekly_hours))
            
            # Generar cronograma inteligente
            schedule = self._generate_smart_schedule(filtered_modules, target_weeks, weekly_hours)
            
            # Generar prompts en paralelo
            learning_prompts = self._generate_prompts_parallel(filtered_modules)
            
            # Encontrar hitos relevantes
            relevant_milestones = self._find_relevant_milestones(filtered_modules)
            
            return {
                "learning_path": path.value,
                "user_level": user_level.value,
                "total_modules": len(filtered_modules),
                "total_duration_hours": total_duration,
                "estimated_weeks": target_weeks,
                "weekly_hours": weekly_hours,
                "schedule": schedule,
                "learning_prompts_count": len(learning_prompts),
                "milestones": relevant_milestones,
                "completion_rate_estimate": self._estimate_completion_rate(user_level),
                "difficulty_distribution": self._analyze_difficulty_distribution(filtered_modules)
            }
            
        except Exception as e:
            logger.error(f"Error generando plan de aprendizaje: {e}")
            raise Exception(f"Error generando plan de aprendizaje: {e}")
    
    def _generate_smart_schedule(self, modules: List[CurriculumModule], 
                               weeks: int, weekly_hours: float) -> List[Dict[str, Any]]:
        """Genera cronograma inteligente basado en dificultad y duración"""
        schedule = []
        current_date = datetime.now()
        
        # Distribuir módulos a lo largo de las semanas
        modules_per_week = max(1, len(modules) // weeks)
        
        for week in range(weeks):
            week_modules = modules[week * modules_per_week:(week + 1) * modules_per_week]
            
            # Si es la última semana, incluir módulos restantes
            if week == weeks - 1:
                week_modules = modules[week * modules_per_week:]
            
            if not week_modules:
                continue
                
            start_date = current_date + timedelta(weeks=week)
            end_date = start_date + timedelta(days=6)
            
            week_duration = sum(m.estimated_duration_hours for m in week_modules)
            
            schedule.append({
                "week": week + 1,
                "modules": [{"id": m.id, "title": m.title, "hours": m.estimated_duration_hours} 
                           for m in week_modules],
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_hours": week_duration,
                "intensity": "high" if week_duration > weekly_hours * 1.2 else 
                           "low" if week_duration < weekly_hours * 0.8 else "normal"
            })
        
        return schedule
    
    def _generate_prompts_parallel(self, modules: List[CurriculumModule]) -> List[LearningPrompt]:
        """Genera prompts en paralelo para mejor rendimiento"""
        def generate_module_prompts(module):
            prompts = []
            for topic_id in module.topics:
                for prompt_type in [PromptType.CONCEPTUAL, PromptType.PRACTICO]:
                    try:
                        prompt = self.prompt_generator.generate_prompt(
                            topic_id, prompt_type, DifficultyLevel.MEDIO
                        )
                        prompts.append(prompt)
                    except Exception as e:
                        logger.warning(f"Error generando prompt para {topic_id}: {e}")
            return prompts
        
        # Usar ThreadPoolExecutor para generar prompts en paralelo
        all_prompts = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_module = {executor.submit(generate_module_prompts, module): module 
                              for module in modules}
            
            for future in future_to_module:
                try:
                    prompts = future.result(timeout=30)
                    all_prompts.extend(prompts)
                except Exception as e:
                    logger.error(f"Error en generación paralela de prompts: {e}")
        
        return all_prompts
    
    def _find_relevant_milestones(self, modules: List[CurriculumModule]) -> List[LearningMilestone]:
        """Encuentra hitos relevantes para los módulos dados"""
        module_topics = set(topic for module in modules for topic in module.topics)
        
        relevant_milestones = []
        for milestone in self.learning_milestones.values():
            if set(milestone.required_topics).intersection(module_topics):
                relevant_milestones.append(milestone)
        
        return sorted(relevant_milestones, key=lambda x: x.phase.order)
    
    def _estimate_completion_rate(self, user_level: SecurityLevel) -> float:
        """Estima la tasa de finalización basada en el nivel del usuario"""
        completion_rates = {
            SecurityLevel.BASICO: 0.85,
            SecurityLevel.INTERMEDIO: 0.90,
            SecurityLevel.AVANZADO: 0.95,
            SecurityLevel.EXPERTO: 0.98
        }
        return completion_rates.get(user_level, 0.85)
    
    def _analyze_difficulty_distribution(self, modules: List[CurriculumModule]) -> Dict[str, int]:
        """Analiza la distribución de dificultad de los módulos"""
        distribution = {level.value: 0 for level in DifficultyLevel}
        
        for module in modules:
            distribution[module.difficulty_level.value] += 1
        
        return distribution
    
    def get_personalized_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones personalizadas avanzadas"""
        try:
            recommendations = []
            
            # Obtener progreso con cache
            milestone_progress = self.check_milestone_progress(user_id)
            
            # Análisis de patrones de aprendizaje
            learning_patterns = self._analyze_learning_patterns(user_id)
            
            # Recomendaciones basadas en IA
            ai_recommendations = self._generate_ai_recommendations(user_id, learning_patterns)
            recommendations.extend(ai_recommendations)
            
            # Recomendaciones por pares (peer recommendations)
            peer_recommendations = self._get_peer_recommendations(user_id)
            recommendations.extend(peer_recommendations)
            
            # Ordenar por prioridad y relevancia
            recommendations.sort(key=lambda x: (x["priority_score"], x["relevance_score"]), reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error obteniendo recomendaciones: {e}")
            return []
    
    def _analyze_learning_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analiza patrones de aprendizaje del usuario"""
        # Implementación simplificada - en producción usaría ML
        return {
            "preferred_difficulty": DifficultyLevel.MEDIO,
            "optimal_session_length": 45,  # minutos
            "best_learning_time": "morning",
            "completion_rate": 0.85,
            "struggle_topics": ["encriptacion", "malware"],
            "strong_topics": ["autenticacion", "firewall"]
        }
    
    def _generate_ai_recommendations(self, user_id: str, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recomendaciones usando patrones de IA"""
        recommendations = []
        
        # Recomendación basada en temas débiles
        if patterns["struggle_topics"]:
            recommendations.append({
                "type": "skill_reinforcement",
                "title": "Refuerza tus temas débiles",
                "description": f"Practica más en: {', '.join(patterns['struggle_topics'][:3])}",
                "priority_score": 90,
                "relevance_score": 95,
                "action": "realizar_ejercicios_adicionales",
                "estimated_time": "30-45 minutos",
                "difficulty": patterns["preferred_difficulty"].value
            })
        
        return recommendations
    
    def _get_peer_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones basadas en usuarios similares"""
        # Implementación simplificada - en producción usaría collaborative filtering
        return [{
            "type": "peer_suggestion",
            "title": "Módulo popular entre usuarios similares",
            "description": "El 85% de usuarios con tu perfil completaron este módulo",
            "priority_score": 70,
            "relevance_score": 80,
            "module_id": "desarrollo_seguro",
            "social_proof": "156 usuarios completaron"
        }]
    
    def export_curriculum_config(self, file_path: Path) -> None:
        """Exporta configuración del currículum"""
        try:
            config = {
                "modules": {k: {
                    "id": v.id,
                    "title": v.title,
                    "description": v.description,
                    "phase": v.phase.value,
                    "topics": v.topics,
                    "duration": v.estimated_duration_hours,
                    "prerequisites": v.prerequisites,
                    "tags": list(v.tags),
                    "difficulty": v.difficulty_level.value
                } for k, v in self.curriculum_modules.items()},
                "milestones": {k: {
                    "id": v.id,
                    "title": v.title,
                    "description": v.description,
                    "phase": v.phase.value,
                    "required_topics": v.required_topics,
                    "passing_score": v.passing_score,
                    "xp_points": v.xp_points
                } for k, v in self.learning_milestones.items()},
                "paths": {k.value: v for k, v in self.learning_paths.items()}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Configuración exportada a {file_path}")
            
        except Exception as e:
            logger.error(f"Error exportando configuración: {e}")
            raise
    
    def get_curriculum_analytics(self) -> Dict[str, Any]:
        """Obtiene análisis avanzados del currículum"""
        try:
            modules = list(self.curriculum_modules.values())
            
            # Estadísticas básicas
            total_modules = len(modules)
            total_hours = sum(m.estimated_duration_hours for m in modules)
            
            # Análisis de complejidad
            complexity_analysis = self._analyze_curriculum_complexity(modules)
            
            # Análisis de rutas
            path_analysis = {
                path.value: {
                    "modules": len(self.get_learning_path(path)),
                    "total_hours": sum(m.estimated_duration_hours for m in self.get_learning_path(path)),
                    "difficulty_spread": self._calculate_difficulty_spread(self.get_learning_path(path))
                }
                for path in LearningPath
            }
            
            return {
                "overview": {
                    "total_modules": total_modules,
                    "total_hours": total_hours,
                    "average_module_duration": total_hours / max(1, total_modules),
                    "total_milestones": len(self.learning_milestones)
                },
                "complexity_analysis": complexity_analysis,
                "path_analysis": path_analysis,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en análisis del currículum: {e}")
            raise
    
    def _analyze_curriculum_complexity(self, modules: List[CurriculumModule]) -> Dict[str, Any]:
        """Analiza la complejidad del currículum"""
        prerequisite_depths = []
        
        for module in modules:
            depth = self._calculate_prerequisite_depth(module.id, set())
            prerequisite_depths.append(depth)
        
        return {
            "max_prerequisite_depth": max(prerequisite_depths) if prerequisite_depths else 0,
            "average_prerequisite_depth": sum(prerequisite_depths) / len(prerequisite_depths) if prerequisite_depths else 0,
            "modules_without_prerequisites": sum(1 for m in modules if not m.prerequisites),
            "dependency_complexity": "high" if max(prerequisite_depths, default=0) > 3 else "medium" if max(prerequisite_depths, default=0) > 1 else "low"
        }
    
    def _calculate_prerequisite_depth(self, module_id: str, visited: Set[str]) -> int:
        """Calcula la profundidad de prerequisitos recursivamente"""
        if module_id in visited:
            return 0  # Evitar ciclos infinitos
        
        module = self.curriculum_modules.get(module_id)
        if not module or not module.prerequisites:
            return 0
        
        visited.add(module_id)
        depths = [self._calculate_prerequisite_depth(prereq, visited.copy()) 
                 for prereq in module.prerequisites]
        
        return 1 + max(depths) if depths else 1
    
    def _calculate_difficulty_spread(self, modules: List[CurriculumModule]) -> Dict[str, float]:
        """Calcula la distribución de dificultad"""
        if not modules:
            return {}
        
        difficulty_counts = {}
        for module in modules:
            difficulty = module.difficulty_level.value
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        total = len(modules)
        return {k: v / total for k, v in difficulty_counts.items()}
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)