"""
Generador de Prompts para Aprendizaje Profundo
Versión: 0.6.0
Sistema de generación de prompts educativos para entrenamiento de IA en ciberseguridad
"""

import random
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel, TopicCategory

class PromptType(Enum):
    """Tipos de prompts educativos"""
    CONCEPTUAL = "conceptual"
    PRACTICO = "practico"
    CODIGO = "codigo"
    CASO_ESTUDIO = "caso_estudio"
    EVALUACION = "evaluacion"
    SIMULACION = "simulacion"

class DifficultyLevel(Enum):
    """Niveles de dificultad de prompts"""
    FACIL = "fácil"
    MEDIO = "medio"
    DIFICIL = "difícil"
    EXPERTO = "experto"

@dataclass
class LearningPrompt:
    """Estructura de un prompt de aprendizaje"""
    id: str
    topic_id: str
    prompt_type: PromptType
    difficulty: DifficultyLevel
    title: str
    content: str
    expected_response: str
    learning_objectives: List[str]
    keywords: List[str]
    code_examples: List[str]
    resources: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

class PromptGenerator:
    """Generador de prompts para aprendizaje profundo en ciberseguridad"""
    
    def __init__(self):
        self.security_topics = SecurityTopics()
        self.prompt_templates = self._initialize_templates()
        self.generated_prompts = []
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Inicializa plantillas de prompts por tipo"""
        return {
            "conceptual": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Categoría: {category}

Pregunta Conceptual:
{question}

Objetivos de Aprendizaje:
{objectives}

Contexto:
{context}

Instrucciones:
1. Explica el concepto de manera clara y detallada
2. Proporciona ejemplos prácticos
3. Menciona las mejores prácticas
4. Incluye consideraciones de seguridad

Recursos Adicionales:
{resources}
""",
                "questions": [
                    "¿Qué es {concept} y por qué es importante en ciberseguridad?",
                    "Explica los principios fundamentales de {concept}",
                    "¿Cuáles son los componentes principales de {concept}?",
                    "Describe el funcionamiento de {concept} en sistemas informáticos",
                    "¿Qué problemas resuelve {concept} en el ámbito de la seguridad?"
                ]
            },
            
            "practico": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Tipo: Ejercicio Práctico

Escenario:
{scenario}

Objetivo:
{objective}

Tareas a Realizar:
{tasks}

Criterios de Evaluación:
{evaluation_criteria}

Herramientas Sugeridas:
{tools}

Instrucciones:
1. Analiza el escenario proporcionado
2. Identifica los problemas de seguridad
3. Propón soluciones prácticas
4. Implementa las medidas de seguridad
5. Documenta el proceso

Recursos:
{resources}
""",
                "scenarios": [
                    "Una empresa ha sufrido un ataque de ransomware. Como especialista en ciberseguridad, debes:",
                    "Se ha detectado una vulnerabilidad crítica en una aplicación web. Tu tarea es:",
                    "Una organización necesita implementar un sistema de autenticación multifactor. Diseña:",
                    "Se requiere realizar una auditoría de seguridad de una red corporativa. Planifica:",
                    "Un sistema de e-commerce ha sido comprometido. Desarrolla un plan de respuesta:"
                ]
            },
            
            "codigo": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Tipo: Ejercicio de Código

Desafío de Programación:
{challenge}

Requisitos Técnicos:
{requirements}

Lenguaje de Programación: {language}

Funcionalidades a Implementar:
{features}

Criterios de Seguridad:
{security_criteria}

Código de Ejemplo:
{code_example}

Instrucciones:
1. Implementa la solución siguiendo las mejores prácticas de seguridad
2. Incluye validación de entrada y manejo de errores
3. Documenta el código con comentarios explicativos
4. Realiza pruebas de seguridad
5. Optimiza el rendimiento

Recursos:
{resources}
""",
                "challenges": [
                    "Implementa un sistema de hash seguro para contraseñas",
                    "Desarrolla un validador de entrada para prevenir inyecciones SQL",
                    "Crea un generador de tokens JWT con expiración",
                    "Programa un scanner de puertos con detección de servicios",
                    "Construye un sistema de logging de seguridad"
                ]
            },
            
            "caso_estudio": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Tipo: Caso de Estudio

Caso Real:
{case_description}

Contexto de la Organización:
{organization_context}

Incidente de Seguridad:
{security_incident}

Datos Disponibles:
{available_data}

Preguntas de Análisis:
{analysis_questions}

Tareas de Investigación:
{investigation_tasks}

Entregables Esperados:
{deliverables}

Instrucciones:
1. Analiza el caso de manera sistemática
2. Identifica las causas raíz del incidente
3. Evalúa el impacto y las consecuencias
4. Propón medidas de mitigación
5. Desarrolla un plan de prevención

Recursos:
{resources}
""",
                "cases": [
                    "Caso Equifax: Análisis del breach de datos de 2017",
                    "Caso WannaCry: Estudio del ataque de ransomware global",
                    "Caso SolarWinds: Investigación del ataque de cadena de suministro",
                    "Caso Colonial Pipeline: Análisis del ataque a infraestructura crítica",
                    "Caso Log4j: Estudio de la vulnerabilidad de día cero"
                ]
            },
            
            "evaluacion": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Tipo: Evaluación de Conocimientos

Preguntas de Evaluación:
{questions}

Formato de Respuesta:
{answer_format}

Criterios de Calificación:
{grading_criteria}

Tiempo Límite: {time_limit}

Instrucciones:
1. Responde todas las preguntas de manera completa
2. Proporciona ejemplos cuando sea apropiado
3. Justifica tus respuestas con fundamentos técnicos
4. Incluye consideraciones de seguridad
5. Menciona mejores prácticas

Recursos Permitidos:
{allowed_resources}
""",
                "question_types": [
                    "Preguntas de opción múltiple sobre conceptos",
                    "Preguntas de desarrollo sobre implementación",
                    "Ejercicios de análisis de código",
                    "Casos prácticos de resolución de problemas",
                    "Preguntas de comparación y contraste"
                ]
            },
            
            "simulacion": {
                "template": """
Tema: {topic_title}
Nivel: {level}
Tipo: Simulación de Ataque/Defensa

Escenario de Simulación:
{simulation_scenario}

Roles Asignados:
{assigned_roles}

Objetivos del Ejercicio:
{exercise_objectives}

Herramientas Disponibles:
{available_tools}

Reglas del Juego:
{game_rules}

Métricas de Evaluación:
{evaluation_metrics}

Instrucciones:
1. Asume tu rol asignado
2. Desarrolla tu estrategia
3. Implementa tus tácticas
4. Adapta tu enfoque según la situación
5. Documenta tus acciones y resultados

Recursos:
{resources}
""",
                "scenarios": [
                    "Simulación de ataque a una red corporativa",
                    "Ejercicio de respuesta a incidentes de seguridad",
                    "Simulación de penetración a una aplicación web",
                    "Ejercicio de defensa contra malware",
                    "Simulación de gestión de crisis de seguridad"
                ]
            }
        }
    
    def generate_prompt(self, topic_id: str, prompt_type: PromptType, 
                       difficulty: DifficultyLevel = DifficultyLevel.MEDIO,
                       custom_requirements: Dict[str, Any] = None) -> LearningPrompt:
        """Genera un prompt educativo personalizado"""
        try:
            # Obtener tema de seguridad
            topic = self.security_topics.get_topic(topic_id)
            if not topic:
                raise ValueError(f"Tema no encontrado: {topic_id}")
            
            # Obtener plantilla
            template_data = self.prompt_templates.get(prompt_type.value)
            if not template_data:
                raise ValueError(f"Tipo de prompt no soportado: {prompt_type}")
            
            # Generar contenido del prompt
            prompt_content = self._generate_prompt_content(
                topic, prompt_type, difficulty, template_data, custom_requirements
            )
            
            # Crear prompt
            prompt_id = f"{topic_id}_{prompt_type.value}_{difficulty.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            prompt = LearningPrompt(
                id=prompt_id,
                topic_id=topic_id,
                prompt_type=prompt_type,
                difficulty=difficulty,
                title=prompt_content["title"],
                content=prompt_content["content"],
                expected_response=prompt_content["expected_response"],
                learning_objectives=topic.learning_objectives,
                keywords=topic.keywords,
                code_examples=topic.code_examples,
                resources=topic.resources,
                created_at=datetime.now(),
                metadata={
                    "topic_category": topic.category.value,
                    "topic_level": topic.level.value,
                    "custom_requirements": custom_requirements or {}
                }
            )
            
            # Guardar prompt generado
            self.generated_prompts.append(prompt)
            
            return prompt
            
        except Exception as e:
            raise Exception(f"Error generando prompt: {e}")
    
    def _generate_prompt_content(self, topic: SecurityTopic, prompt_type: PromptType, 
                                difficulty: DifficultyLevel, template_data: Dict[str, str],
                                custom_requirements: Dict[str, Any] = None) -> Dict[str, str]:
        """Genera el contenido específico del prompt"""
        
        if prompt_type == PromptType.CONCEPTUAL:
            return self._generate_conceptual_prompt(topic, difficulty, template_data)
        elif prompt_type == PromptType.PRACTICO:
            return self._generate_practical_prompt(topic, difficulty, template_data)
        elif prompt_type == PromptType.CODIGO:
            return self._generate_code_prompt(topic, difficulty, template_data)
        elif prompt_type == PromptType.CASO_ESTUDIO:
            return self._generate_case_study_prompt(topic, difficulty, template_data)
        elif prompt_type == PromptType.EVALUACION:
            return self._generate_evaluation_prompt(topic, difficulty, template_data)
        elif prompt_type == PromptType.SIMULACION:
            return self._generate_simulation_prompt(topic, difficulty, template_data)
        else:
            raise ValueError(f"Tipo de prompt no implementado: {prompt_type}")
    
    def _generate_conceptual_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                                   template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt conceptual"""
        question = random.choice(template_data["questions"]).format(concept=topic.title)
        
        context = f"""
En el contexto de {topic.category.value}, {topic.description}.
Este tema es fundamental para entender {', '.join(topic.keywords[:3])} y su aplicación práctica.
"""
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            category=topic.category.value,
            question=question,
            objectives='\n'.join([f"- {obj}" for obj in topic.learning_objectives]),
            context=context,
            resources='\n'.join([f"- {res}" for res in topic.resources])
        )
        
        expected_response = f"""
Respuesta esperada para {topic.title}:

1. Definición y conceptos clave
2. Importancia en ciberseguridad
3. Componentes principales
4. Ejemplos prácticos de implementación
5. Mejores prácticas y consideraciones de seguridad
6. Casos de uso reales
"""
        
        return {
            "title": f"Conceptos de {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def _generate_practical_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                                  template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt práctico"""
        scenario = random.choice(template_data["scenarios"])
        
        objective = f"Implementar medidas de seguridad relacionadas con {topic.title}"
        
        tasks = [
            f"Analizar el escenario desde la perspectiva de {topic.title}",
            f"Aplicar los principios de {topic.title} para resolver el problema",
            f"Implementar soluciones prácticas basadas en {topic.title}",
            f"Evaluar la efectividad de las medidas implementadas"
        ]
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            scenario=scenario,
            objective=objective,
            tasks='\n'.join([f"- {task}" for task in tasks]),
            evaluation_criteria="Implementación correcta, seguridad, eficiencia, documentación",
            tools="Herramientas de seguridad, lenguajes de programación, frameworks",
            resources='\n'.join([f"- {res}" for res in topic.resources])
        )
        
        expected_response = f"""
Solución práctica para {topic.title}:

1. Análisis del escenario
2. Identificación de problemas de seguridad
3. Diseño de la solución
4. Implementación paso a paso
5. Pruebas y validación
6. Documentación y mantenimiento
"""
        
        return {
            "title": f"Ejercicio Práctico: {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def _generate_code_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                             template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt de código"""
        challenge = random.choice(template_data["challenges"])
        
        requirements = [
            f"Implementar funcionalidad relacionada con {topic.title}",
            "Seguir principios de código seguro",
            "Incluir validación de entrada",
            "Manejar errores de forma segura",
            "Documentar el código"
        ]
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            challenge=challenge,
            requirements='\n'.join([f"- {req}" for req in requirements]),
            language="Python/JavaScript/Java (según preferencia)",
            features='\n'.join([f"- {example}" for example in topic.code_examples[:3]]),
            security_criteria="Validación de entrada, encriptación, autenticación, logging",
            code_example="# Código de ejemplo aquí...",
            resources='\n'.join([f"- {res}" for res in topic.resources])
        )
        
        expected_response = f"""
Código esperado para {topic.title}:

1. Estructura del proyecto
2. Implementación de funcionalidades principales
3. Medidas de seguridad implementadas
4. Manejo de errores y excepciones
5. Pruebas unitarias
6. Documentación del código
"""
        
        return {
            "title": f"Desafío de Código: {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def _generate_case_study_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                                   template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt de caso de estudio"""
        case = random.choice(template_data["cases"])
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            case_description=case,
            organization_context="Organización de tamaño medio con infraestructura híbrida",
            security_incident=f"Incidente relacionado con {topic.title}",
            available_data="Logs de sistema, registros de red, reportes de usuarios",
            analysis_questions=[
                f"¿Cómo se relaciona este caso con {topic.title}?",
                "¿Cuáles fueron las causas raíz del incidente?",
                "¿Qué medidas preventivas se podrían haber implementado?",
                "¿Cómo se puede mejorar la respuesta a incidentes?"
            ],
            investigation_tasks=[
                f"Analizar el incidente desde la perspectiva de {topic.title}",
                "Identificar vulnerabilidades y puntos de falla",
                "Evaluar la efectividad de las medidas de seguridad",
                "Proponer mejoras y recomendaciones"
            ],
            deliverables="Reporte de análisis, plan de remediación, recomendaciones",
            resources='\n'.join([f"- {res}" for res in topic.resources])
        )
        
        expected_response = f"""
Análisis de caso para {topic.title}:

1. Resumen ejecutivo del incidente
2. Análisis técnico detallado
3. Identificación de causas raíz
4. Evaluación de impacto
5. Recomendaciones de mejora
6. Plan de implementación
"""
        
        return {
            "title": f"Caso de Estudio: {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def _generate_evaluation_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                                   template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt de evaluación"""
        questions = [
            f"Explica los conceptos fundamentales de {topic.title}",
            f"¿Cuáles son las mejores prácticas para implementar {topic.title}?",
            f"Describe un escenario real donde {topic.title} sea crítico",
            f"¿Qué herramientas y técnicas se utilizan en {topic.title}?",
            f"Evalúa los riesgos asociados con {topic.title}"
        ]
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            questions='\n'.join([f"{i+1}. {q}" for i, q in enumerate(questions)]),
            answer_format="Respuestas detalladas con ejemplos y justificaciones",
            grading_criteria="Comprensión conceptual, aplicación práctica, análisis crítico",
            time_limit="60 minutos",
            allowed_resources="Documentación técnica, ejemplos de código"
        )
        
        expected_response = f"""
Respuestas esperadas para {topic.title}:

1. Comprensión profunda de conceptos
2. Aplicación práctica de conocimientos
3. Análisis crítico de situaciones
4. Propuestas de soluciones innovadoras
5. Consideración de aspectos de seguridad
"""
        
        return {
            "title": f"Evaluación: {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def _generate_simulation_prompt(self, topic: SecurityTopic, difficulty: DifficultyLevel, 
                                   template_data: Dict[str, str]) -> Dict[str, str]:
        """Genera prompt de simulación"""
        scenario = random.choice(template_data["scenarios"])
        
        content = template_data["template"].format(
            topic_title=topic.title,
            level=topic.level.value,
            simulation_scenario=scenario,
            assigned_roles="Analista de seguridad, Administrador de sistemas, Desarrollador",
            exercise_objectives=f"Practicar {topic.title} en un entorno controlado",
            available_tools="Herramientas de seguridad, plataformas de simulación",
            game_rules="Tiempo limitado, objetivos específicos, restricciones técnicas",
            evaluation_metrics="Efectividad, creatividad, trabajo en equipo, documentación",
            resources='\n'.join([f"- {res}" for res in topic.resources])
        )
        
        expected_response = f"""
Resultados esperados para {topic.title}:

1. Estrategia desarrollada
2. Tácticas implementadas
3. Adaptación a situaciones cambiantes
4. Colaboración efectiva
5. Documentación de procesos
"""
        
        return {
            "title": f"Simulación: {topic.title}",
            "content": content,
            "expected_response": expected_response
        }
    
    def generate_learning_curriculum(self, topics: List[str], 
                                   prompt_types: List[PromptType] = None,
                                   difficulty_progression: bool = True) -> List[LearningPrompt]:
        """Genera un currículum completo de aprendizaje"""
        if prompt_types is None:
            prompt_types = [PromptType.CONCEPTUAL, PromptType.PRACTICO, PromptType.CODIGO]
        
        curriculum = []
        
        for topic_id in topics:
            topic = self.security_topics.get_topic(topic_id)
            if not topic:
                continue
            
            # Determinar dificultad basada en el nivel del tema
            if difficulty_progression:
                if topic.level == SecurityLevel.BASICO:
                    difficulties = [DifficultyLevel.FACIL, DifficultyLevel.MEDIO]
                elif topic.level == SecurityLevel.INTERMEDIO:
                    difficulties = [DifficultyLevel.MEDIO, DifficultyLevel.DIFICIL]
                elif topic.level == SecurityLevel.AVANZADO:
                    difficulties = [DifficultyLevel.DIFICIL, DifficultyLevel.EXPERTO]
                else:
                    difficulties = [DifficultyLevel.EXPERTO]
            else:
                difficulties = [DifficultyLevel.MEDIO]
            
            # Generar prompts para cada tipo y dificultad
            for prompt_type in prompt_types:
                for difficulty in difficulties:
                    try:
                        prompt = self.generate_prompt(topic_id, prompt_type, difficulty)
                        curriculum.append(prompt)
                    except Exception as e:
                        print(f"Error generando prompt para {topic_id}: {e}")
        
        return curriculum
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de prompts generados"""
        if not self.generated_prompts:
            return {"total_prompts": 0}
        
        by_type = {}
        by_difficulty = {}
        by_topic = {}
        
        for prompt in self.generated_prompts:
            # Por tipo
            prompt_type = prompt.prompt_type.value
            by_type[prompt_type] = by_type.get(prompt_type, 0) + 1
            
            # Por dificultad
            difficulty = prompt.difficulty.value
            by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
            
            # Por tema
            topic = prompt.topic_id
            by_topic[topic] = by_topic.get(topic, 0) + 1
        
        return {
            "total_prompts": len(self.generated_prompts),
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_topic": by_topic,
            "date_range": {
                "first": min(p.created_at for p in self.generated_prompts).isoformat(),
                "last": max(p.created_at for p in self.generated_prompts).isoformat()
            }
        }
    
    def export_prompts(self, filepath: str, format: str = "json") -> None:
        """Exporta prompts generados a archivo"""
        try:
            if format.lower() == "json":
                prompts_data = []
                for prompt in self.generated_prompts:
                    prompt_dict = {
                        "id": prompt.id,
                        "topic_id": prompt.topic_id,
                        "prompt_type": prompt.prompt_type.value,
                        "difficulty": prompt.difficulty.value,
                        "title": prompt.title,
                        "content": prompt.content,
                        "expected_response": prompt.expected_response,
                        "learning_objectives": prompt.learning_objectives,
                        "keywords": prompt.keywords,
                        "code_examples": prompt.code_examples,
                        "resources": prompt.resources,
                        "created_at": prompt.created_at.isoformat(),
                        "metadata": prompt.metadata
                    }
                    prompts_data.append(prompt_dict)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(prompts_data, f, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            print(f"Prompts exportados exitosamente a: {filepath}")
            
        except Exception as e:
            raise Exception(f"Error exportando prompts: {e}")
    
    def load_prompts(self, filepath: str) -> None:
        """Carga prompts desde archivo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
            
            self.generated_prompts = []
            for prompt_dict in prompts_data:
                prompt = LearningPrompt(
                    id=prompt_dict["id"],
                    topic_id=prompt_dict["topic_id"],
                    prompt_type=PromptType(prompt_dict["prompt_type"]),
                    difficulty=DifficultyLevel(prompt_dict["difficulty"]),
                    title=prompt_dict["title"],
                    content=prompt_dict["content"],
                    expected_response=prompt_dict["expected_response"],
                    learning_objectives=prompt_dict["learning_objectives"],
                    keywords=prompt_dict["keywords"],
                    code_examples=prompt_dict["code_examples"],
                    resources=prompt_dict["resources"],
                    created_at=datetime.fromisoformat(prompt_dict["created_at"]),
                    metadata=prompt_dict["metadata"]
                )
                self.generated_prompts.append(prompt)
            
            print(f"Prompts cargados exitosamente desde: {filepath}")
            
        except Exception as e:
            raise Exception(f"Error cargando prompts: {e}")
