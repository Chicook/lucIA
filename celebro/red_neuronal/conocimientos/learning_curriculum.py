"""
Currículum de Aprendizaje para @red_neuronal
Versión: 0.6.0
Sistema de currículum estructurado para aprendizaje profundo en ciberseguridad
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel, TopicCategory
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel
from .knowledge_base import KnowledgeBase, LearningSession

class LearningPath(Enum):
    """Rutas de aprendizaje disponibles"""
    FUNDAMENTOS = "fundamentos"
    DESARROLLADOR = "desarrollador"
    ADMINISTRADOR = "administrador"
    ANALISTA = "analista"
    AUDITOR = "auditor"
    COMPLETO = "completo"

class LearningPhase(Enum):
    """Fases del aprendizaje"""
    INTRODUCCION = "introduccion"
    CONCEPTOS = "conceptos"
    PRACTICA = "practica"
    APLICACION = "aplicacion"
    MAESTRIA = "maestria"

@dataclass
class CurriculumModule:
    """Módulo del currículum"""
    id: str
    title: str
    description: str
    phase: LearningPhase
    topics: List[str]
    estimated_duration_hours: int
    prerequisites: List[str]
    learning_objectives: List[str]
    assessment_criteria: List[str]
    resources: List[str]

@dataclass
class LearningMilestone:
    """Hito de aprendizaje"""
    id: str
    title: str
    description: str
    phase: LearningPhase
    required_topics: List[str]
    assessment_type: str
    passing_score: float
    rewards: List[str]

class LearningCurriculum:
    """Sistema de currículum de aprendizaje estructurado"""
    
    def __init__(self):
        self.security_topics = SecurityTopics()
        self.prompt_generator = PromptGenerator()
        self.knowledge_base = KnowledgeBase()
        self.curriculum_modules = self._initialize_modules()
        self.learning_milestones = self._initialize_milestones()
        self.learning_paths = self._initialize_learning_paths()
    
    def _initialize_modules(self) -> Dict[str, CurriculumModule]:
        """Inicializa los módulos del currículum"""
        modules = {}
        
        # MÓDULO 1: FUNDAMENTOS DE CIBERSEGURIDAD
        modules["fundamentos"] = CurriculumModule(
            id="fundamentos",
            title="Fundamentos de Ciberseguridad",
            description="Conceptos básicos y principios fundamentales de la ciberseguridad",
            phase=LearningPhase.INTRODUCCION,
            topics=["autenticacion", "encriptacion"],
            estimated_duration_hours=8,
            prerequisites=[],
            learning_objectives=[
                "Comprender los conceptos básicos de ciberseguridad",
                "Identificar amenazas comunes y vulnerabilidades",
                "Aplicar principios de autenticación y autorización",
                "Implementar técnicas básicas de encriptación"
            ],
            assessment_criteria=[
                "Explicar conceptos clave de ciberseguridad",
                "Identificar vulnerabilidades en sistemas",
                "Implementar medidas de autenticación básicas",
                "Aplicar encriptación a datos sensibles"
            ],
            resources=[
                "OWASP Top 10",
                "NIST Cybersecurity Framework",
                "CIS Controls",
                "ISO 27001"
            ]
        )
        
        # MÓDULO 2: DESARROLLO SEGURO
        modules["desarrollo_seguro"] = CurriculumModule(
            id="desarrollo_seguro",
            title="Desarrollo de Código Seguro",
            description="Principios y prácticas para desarrollar aplicaciones seguras",
            phase=LearningPhase.CONCEPTOS,
            topics=["secure_coding", "web_security"],
            estimated_duration_hours=12,
            prerequisites=["fundamentos"],
            learning_objectives=[
                "Aplicar principios de código seguro",
                "Prevenir vulnerabilidades OWASP Top 10",
                "Implementar validación de entrada",
                "Realizar code review de seguridad"
            ],
            assessment_criteria=[
                "Escribir código libre de vulnerabilidades comunes",
                "Implementar controles de seguridad web",
                "Realizar análisis estático de código",
                "Aplicar técnicas de testing de seguridad"
            ],
            resources=[
                "OWASP Secure Coding Practices",
                "CERT Secure Coding Standards",
                "SANS Secure Coding",
                "Microsoft Secure Development Lifecycle"
            ]
        )
        
        # MÓDULO 3: AMENAZAS Y VULNERABILIDADES
        modules["amenazas"] = CurriculumModule(
            id="amenazas",
            title="Amenazas y Vulnerabilidades",
            description="Identificación, análisis y mitigación de amenazas cibernéticas",
            phase=LearningPhase.CONCEPTOS,
            topics=["malware", "phishing"],
            estimated_duration_hours=10,
            prerequisites=["fundamentos"],
            learning_objectives=[
                "Identificar diferentes tipos de malware",
                "Reconocer ataques de phishing",
                "Implementar sistemas de detección",
                "Desarrollar estrategias de prevención"
            ],
            assessment_criteria=[
                "Clasificar tipos de malware",
                "Detectar emails de phishing",
                "Configurar sistemas de detección",
                "Implementar medidas preventivas"
            ],
            resources=[
                "MITRE ATT&CK Framework",
                "Malware Analysis Techniques",
                "Anti-Phishing Working Group",
                "NIST Cybersecurity Framework"
            ]
        )
        
        # MÓDULO 4: DEFENSAS DE RED
        modules["defensas_red"] = CurriculumModule(
            id="defensas_red",
            title="Defensas de Red y Sistemas",
            description="Implementación de controles de seguridad en redes y sistemas",
            phase=LearningPhase.PRACTICA,
            topics=["firewall", "ids_ips"],
            estimated_duration_hours=14,
            prerequisites=["fundamentos", "amenazas"],
            learning_objectives=[
                "Configurar firewalls de red y host",
                "Implementar sistemas IDS/IPS",
                "Monitorear tráfico de red",
                "Responder a incidentes de seguridad"
            ],
            assessment_criteria=[
                "Configurar reglas de firewall",
                "Implementar detección de intrusiones",
                "Analizar logs de seguridad",
                "Desarrollar planes de respuesta"
            ],
            resources=[
                "iptables Documentation",
                "Snort User Manual",
                "Suricata Documentation",
                "SIEM Implementation Guide"
            ]
        )
        
        # MÓDULO 5: EVALUACIÓN DE VULNERABILIDADES
        modules["evaluacion_vuln"] = CurriculumModule(
            id="evaluacion_vuln",
            title="Evaluación de Vulnerabilidades",
            description="Herramientas y técnicas para evaluar vulnerabilidades en sistemas",
            phase=LearningPhase.PRACTICA,
            topics=["vulnerability_assessment"],
            estimated_duration_hours=10,
            prerequisites=["defensas_red"],
            learning_objectives=[
                "Realizar evaluaciones de vulnerabilidades",
                "Interpretar resultados de scanners",
                "Priorizar vulnerabilidades",
                "Desarrollar planes de remediación"
            ],
            assessment_criteria=[
                "Ejecutar scans de vulnerabilidades",
                "Analizar reportes de seguridad",
                "Priorizar riesgos",
                "Implementar remediaciones"
            ],
            resources=[
                "NIST Vulnerability Database",
                "CVE Database",
                "OWASP Testing Guide",
                "Nessus Documentation"
            ]
        )
        
        # MÓDULO 6: CUMPLIMIENTO Y LEGISLACIÓN
        modules["cumplimiento"] = CurriculumModule(
            id="cumplimiento",
            title="Cumplimiento y Legislación",
            description="Marco legal y regulatorio en ciberseguridad",
            phase=LearningPhase.APLICACION,
            topics=["gdpr"],
            estimated_duration_hours=8,
            prerequisites=["fundamentos"],
            learning_objectives=[
                "Entender requisitos del GDPR",
                "Implementar medidas de privacidad",
                "Desarrollar políticas de seguridad",
                "Gestionar cumplimiento regulatorio"
            ],
            assessment_criteria=[
                "Interpretar regulaciones de privacidad",
                "Implementar controles de privacidad",
                "Desarrollar políticas de seguridad",
                "Auditar cumplimiento"
            ],
            resources=[
                "GDPR Official Text",
                "ICO Guidance",
                "Privacy by Design",
                "ISO 27001"
            ]
        )
        
        return modules
    
    def _initialize_milestones(self) -> Dict[str, LearningMilestone]:
        """Inicializa los hitos de aprendizaje"""
        milestones = {}
        
        # HITO 1: FUNDAMENTOS COMPLETADOS
        milestones["fundamentos_completados"] = LearningMilestone(
            id="fundamentos_completados",
            title="Fundamentos de Ciberseguridad Dominados",
            description="Has completado exitosamente los conceptos básicos de ciberseguridad",
            phase=LearningPhase.CONCEPTOS,
            required_topics=["autenticacion", "encriptacion"],
            assessment_type="evaluacion_practica",
            passing_score=80.0,
            rewards=["Badge: Fundamentos de Ciberseguridad", "Acceso a módulos avanzados"]
        )
        
        # HITO 2: DESARROLLADOR SEGURO
        milestones["desarrollador_seguro"] = LearningMilestone(
            id="desarrollador_seguro",
            title="Desarrollador de Código Seguro",
            description="Has demostrado competencia en desarrollo de código seguro",
            phase=LearningPhase.PRACTICA,
            required_topics=["secure_coding", "web_security"],
            assessment_type="proyecto_practico",
            passing_score=85.0,
            rewards=["Certificado: Desarrollador Seguro", "Herramientas de desarrollo"]
        )
        
        # HITO 3: ANALISTA DE AMENAZAS
        milestones["analista_amenazas"] = LearningMilestone(
            id="analista_amenazas",
            title="Analista de Amenazas Cibernéticas",
            description="Has desarrollado habilidades avanzadas en análisis de amenazas",
            phase=LearningPhase.APLICACION,
            required_topics=["malware", "phishing", "ids_ips"],
            assessment_type="simulacion_ataque",
            passing_score=90.0,
            rewards=["Certificación: Analista de Amenazas", "Acceso a herramientas avanzadas"]
        )
        
        # HITO 4: ESPECIALISTA EN SEGURIDAD
        milestones["especialista_seguridad"] = LearningMilestone(
            id="especialista_seguridad",
            title="Especialista en Ciberseguridad",
            description="Has alcanzado un nivel experto en ciberseguridad",
            phase=LearningPhase.MAESTRIA,
            required_topics=["autenticacion", "encriptacion", "malware", "phishing", 
                           "firewall", "ids_ips", "vulnerability_assessment", "gdpr"],
            assessment_type="examen_completo",
            passing_score=95.0,
            rewards=["Certificación: Especialista en Ciberseguridad", "Mentoría disponible"]
        )
        
        return milestones
    
    def _initialize_learning_paths(self) -> Dict[LearningPath, List[str]]:
        """Inicializa las rutas de aprendizaje"""
        return {
            LearningPath.FUNDAMENTOS: ["fundamentos"],
            LearningPath.DESARROLLADOR: ["fundamentos", "desarrollo_seguro", "web_security"],
            LearningPath.ADMINISTRADOR: ["fundamentos", "defensas_red", "evaluacion_vuln"],
            LearningPath.ANALISTA: ["fundamentos", "amenazas", "ids_ips", "vulnerability_assessment"],
            LearningPath.AUDITOR: ["fundamentos", "evaluacion_vuln", "cumplimiento"],
            LearningPath.COMPLETO: ["fundamentos", "desarrollo_seguro", "amenazas", 
                                  "defensas_red", "evaluacion_vuln", "cumplimiento"]
        }
    
    def get_learning_path(self, path: LearningPath) -> List[CurriculumModule]:
        """Obtiene los módulos de una ruta de aprendizaje"""
        module_ids = self.learning_paths.get(path, [])
        return [self.curriculum_modules[module_id] for module_id in module_ids 
                if module_id in self.curriculum_modules]
    
    def get_module(self, module_id: str) -> Optional[CurriculumModule]:
        """Obtiene un módulo específico"""
        return self.curriculum_modules.get(module_id)
    
    def get_available_modules(self, completed_modules: List[str] = None) -> List[CurriculumModule]:
        """Obtiene módulos disponibles basados en prerequisitos"""
        if completed_modules is None:
            completed_modules = []
        
        available = []
        for module in self.curriculum_modules.values():
            if all(prereq in completed_modules for prereq in module.prerequisites):
                available.append(module)
        
        return available
    
    def generate_learning_plan(self, path: LearningPath, user_level: SecurityLevel = SecurityLevel.BASICO) -> Dict[str, Any]:
        """Genera un plan de aprendizaje personalizado"""
        try:
            modules = self.get_learning_path(path)
            
            # Filtrar módulos según el nivel del usuario
            if user_level == SecurityLevel.BASICO:
                filtered_modules = [m for m in modules if m.phase in [LearningPhase.INTRODUCCION, LearningPhase.CONCEPTOS]]
            elif user_level == SecurityLevel.INTERMEDIO:
                filtered_modules = [m for m in modules if m.phase in [LearningPhase.CONCEPTOS, LearningPhase.PRACTICA]]
            elif user_level == SecurityLevel.AVANZADO:
                filtered_modules = [m for m in modules if m.phase in [LearningPhase.PRACTICA, LearningPhase.APLICACION]]
            else:  # EXPERTO
                filtered_modules = modules
            
            # Calcular duración total
            total_duration = sum(module.estimated_duration_hours for module in filtered_modules)
            
            # Generar cronograma
            schedule = []
            current_date = datetime.now()
            
            for i, module in enumerate(filtered_modules):
                start_date = current_date + timedelta(days=i * 7)  # 1 semana por módulo
                end_date = start_date + timedelta(days=6)
                
                schedule.append({
                    "week": i + 1,
                    "module": module,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_hours": module.estimated_duration_hours
                })
            
            # Generar prompts para cada módulo
            learning_prompts = []
            for module in filtered_modules:
                for topic_id in module.topics:
                    # Generar diferentes tipos de prompts
                    prompt_types = [PromptType.CONCEPTUAL, PromptType.PRACTICO, PromptType.CODIGO]
                    
                    for prompt_type in prompt_types:
                        try:
                            prompt = self.prompt_generator.generate_prompt(
                                topic_id, prompt_type, DifficultyLevel.MEDIO
                            )
                            learning_prompts.append(prompt)
                        except Exception as e:
                            print(f"Error generando prompt para {topic_id}: {e}")
            
            return {
                "learning_path": path.value,
                "user_level": user_level.value,
                "total_modules": len(filtered_modules),
                "total_duration_hours": total_duration,
                "estimated_weeks": len(filtered_modules),
                "schedule": schedule,
                "learning_prompts": len(learning_prompts),
                "milestones": [milestone for milestone in self.learning_milestones.values() 
                              if any(topic in [t for m in filtered_modules for t in m.topics] 
                                   for topic in milestone.required_topics)]
            }
            
        except Exception as e:
            raise Exception(f"Error generando plan de aprendizaje: {e}")
    
    def start_learning_session(self, module_id: str, user_id: str = None) -> LearningSession:
        """Inicia una sesión de aprendizaje para un módulo"""
        try:
            module = self.get_module(module_id)
            if not module:
                raise ValueError(f"Módulo no encontrado: {module_id}")
            
            # Crear sesión para el primer tema del módulo
            if module.topics:
                topic_id = module.topics[0]
                session = self.knowledge_base.create_learning_session(topic_id, user_id)
                
                # Generar prompts para la sesión
                for topic_id in module.topics:
                    for prompt_type in [PromptType.CONCEPTUAL, PromptType.PRACTICO]:
                        try:
                            prompt = self.prompt_generator.generate_prompt(
                                topic_id, prompt_type, DifficultyLevel.MEDIO
                            )
                            self.knowledge_base.add_prompt_to_session(session.id, prompt)
                        except Exception as e:
                            print(f"Error generando prompt: {e}")
                
                return session
            else:
                raise ValueError("El módulo no tiene temas asignados")
                
        except Exception as e:
            raise Exception(f"Error iniciando sesión de aprendizaje: {e}")
    
    def check_milestone_progress(self, user_id: str) -> List[Dict[str, Any]]:
        """Verifica el progreso hacia los hitos"""
        try:
            milestone_progress = []
            
            for milestone in self.learning_milestones.values():
                # Obtener progreso de cada tema requerido
                topic_progress = []
                for topic_id in milestone.required_topics:
                    progress = self.knowledge_base.get_learning_progress(topic_id, user_id)
                    topic_progress.append(progress)
                
                # Calcular progreso general del hito
                if topic_progress:
                    avg_completion = sum(p["completion_percentage"] for p in topic_progress) / len(topic_progress)
                    avg_accuracy = sum(p["accuracy_rate"] for p in topic_progress) / len(topic_progress)
                    
                    milestone_progress.append({
                        "milestone": milestone,
                        "completion_percentage": avg_completion,
                        "accuracy_rate": avg_accuracy,
                        "topics_completed": sum(1 for p in topic_progress if p["completion_percentage"] >= 80),
                        "total_topics": len(milestone.required_topics),
                        "is_achieved": avg_completion >= milestone.passing_score,
                        "topics_progress": topic_progress
                    })
            
            return milestone_progress
            
        except Exception as e:
            raise Exception(f"Error verificando progreso de hitos: {e}")
    
    def get_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones personalizadas de aprendizaje"""
        try:
            recommendations = []
            
            # Obtener progreso actual
            milestone_progress = self.check_milestone_progress(user_id)
            
            # Recomendaciones basadas en hitos no alcanzados
            for progress in milestone_progress:
                if not progress["is_achieved"]:
                    recommendations.append({
                        "type": "milestone_focus",
                        "title": f"Enfócate en {progress['milestone'].title}",
                        "description": f"Completa {progress['topics_completed']}/{progress['total_topics']} temas requeridos",
                        "priority": "high" if progress["completion_percentage"] > 50 else "medium",
                        "action": f"Continuar con los temas de {progress['milestone'].title}",
                        "estimated_time": "2-4 horas"
                    })
            
            # Recomendaciones basadas en temas débiles
            weak_topics = []
            for progress in milestone_progress:
                for topic_progress in progress["topics_progress"]:
                    if topic_progress["accuracy_rate"] < 70:
                        weak_topics.append(topic_progress["topic_id"])
            
            if weak_topics:
                recommendations.append({
                    "type": "skill_improvement",
                    "title": "Mejora tus habilidades en temas débiles",
                    "description": f"Refuerza tu conocimiento en: {', '.join(weak_topics[:3])}",
                    "priority": "high",
                    "action": "Realizar ejercicios adicionales en estos temas",
                    "estimated_time": "1-2 horas"
                })
            
            # Recomendaciones de nuevos módulos
            available_modules = self.get_available_modules()
            if available_modules:
                recommendations.append({
                    "type": "new_content",
                    "title": "Explora nuevos módulos",
                    "description": f"Tienes {len(available_modules)} módulos disponibles",
                    "priority": "low",
                    "action": "Revisar módulos disponibles en tu ruta de aprendizaje",
                    "estimated_time": "Variable"
                })
            
            return recommendations
            
        except Exception as e:
            raise Exception(f"Error obteniendo recomendaciones: {e}")
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del currículum"""
        try:
            total_modules = len(self.curriculum_modules)
            total_milestones = len(self.learning_milestones)
            total_topics = len(self.security_topics.get_all_topics())
            
            # Estadísticas por fase
            phase_stats = {}
            for phase in LearningPhase:
                phase_modules = [m for m in self.curriculum_modules.values() if m.phase == phase]
                phase_stats[phase.value] = {
                    "modules": len(phase_modules),
                    "total_hours": sum(m.estimated_duration_hours for m in phase_modules)
                }
            
            # Estadísticas por ruta
            path_stats = {}
            for path in LearningPath:
                modules = self.get_learning_path(path)
                path_stats[path.value] = {
                    "modules": len(modules),
                    "total_hours": sum(m.estimated_duration_hours for m in modules),
                    "topics": len(set(topic for m in modules for topic in m.topics))
                }
            
            return {
                "total_modules": total_modules,
                "total_milestones": total_milestones,
                "total_topics": total_topics,
                "phase_statistics": phase_stats,
                "path_statistics": path_stats,
                "learning_prompts_generated": len(self.prompt_generator.generated_prompts)
            }
            
        except Exception as e:
            raise Exception(f"Error obteniendo estadísticas del currículum: {e}")
