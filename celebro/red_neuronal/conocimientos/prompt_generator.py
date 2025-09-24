"""
Generador de Prompts para Aprendizaje Profundo
Versión: 0.7.0
Sistema de generación de prompts educativos para entrenamiento de IA en ciberseguridad

Cambios principales en v0.7.0:
- Optimización de importaciones con lazy loading
- Mejora en la gestión de memoria con generators
- Implementación de cache para templates
- Extensión de funcionalidades de generación batch
- Mejora en la serialización y deserialización
- Implementación de validaciones robustas
"""

# Importaciones estándar optimizadas
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum, auto
from functools import lru_cache, cached_property
from pathlib import Path
from typing import (
    Dict, List, Any, Optional, Tuple, Generator, 
    Union, Iterator, Protocol, TypeVar, Generic
)
import warnings

# Importaciones locales con manejo de errores
try:
    from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel, TopicCategory
except ImportError as e:
    warnings.warn(f"Error importing security_topics: {e}")
    # Fallback para desarrollo independiente
    SecurityTopics = SecurityTopic = SecurityLevel = TopicCategory = None

# Type hints mejorados
T = TypeVar('T')
PromptData = Dict[str, Any]
TemplateData = Dict[str, Dict[str, Any]]

class PromptType(Enum):
    """
    Tipos de prompts educativos expandidos
    
    Cambios:
    - Añadidos nuevos tipos para mayor cobertura educativa
    - Implementación con auto() para mejor mantenimiento
    """
    CONCEPTUAL = auto()
    PRACTICO = auto()
    CODIGO = auto()
    CASO_ESTUDIO = auto()
    EVALUACION = auto()
    SIMULACION = auto()
    # Nuevos tipos añadidos
    INVESTIGACION = auto()
    LABORATORIO = auto()
    PROYECTO = auto()
    REVISION_PARES = auto()

    @property
    def display_name(self) -> str:
        """Nombres legibles para UI"""
        names = {
            self.CONCEPTUAL: "Conceptual",
            self.PRACTICO: "Práctico",
            self.CODIGO: "Código",
            self.CASO_ESTUDIO: "Caso de Estudio",
            self.EVALUACION: "Evaluación",
            self.SIMULACION: "Simulación",
            self.INVESTIGACION: "Investigación",
            self.LABORATORIO: "Laboratorio",
            self.PROYECTO: "Proyecto",
            self.REVISION_PARES: "Revisión por Pares"
        }
        return names.get(self, self.name.replace('_', ' ').title())

class DifficultyLevel(Enum):
    """
    Niveles de dificultad expandidos con métricas cuantitativas
    
    Mejoras:
    - Añadido sistema de puntuación numérica
    - Implementación de métodos de comparación
    """
    PRINCIPIANTE = (1, "principiante")
    FACIL = (2, "fácil")
    MEDIO = (3, "medio")
    DIFICIL = (4, "difícil")
    EXPERTO = (5, "experto")
    MAESTRO = (6, "maestro")

    def __init__(self, level: int, name: str):
        self.level = level
        self.display_name = name

    def __lt__(self, other) -> bool:
        return self.level < other.level

    def __le__(self, other) -> bool:
        return self.level <= other.level

    @classmethod
    def from_string(cls, difficulty_str: str) -> 'DifficultyLevel':
        """Conversión robusta desde string"""
        mapping = {d.display_name: d for d in cls}
        return mapping.get(difficulty_str.lower(), cls.MEDIO)

class PromptGenerationError(Exception):
    """Excepción personalizada para errores de generación de prompts"""
    pass

class ValidationError(Exception):
    """Excepción para errores de validación de datos"""
    pass

@dataclass(frozen=True)  # Inmutable para mejor performance en cache
class LearningPrompt:
    """
    Estructura mejorada de un prompt de aprendizaje
    
    Cambios:
    - Campos opcionales con valores por defecto
    - Validación automática en post_init
    - Métodos de utilidad integrados
    """
    id: str
    topic_id: str
    prompt_type: PromptType
    difficulty: DifficultyLevel
    title: str
    content: str
    expected_response: str
    learning_objectives: Tuple[str, ...] = ()  # Tuple para inmutabilidad
    keywords: Tuple[str, ...] = ()
    code_examples: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    # Nuevos campos
    estimated_duration: int = 30  # minutos
    prerequisites: Tuple[str, ...] = ()
    success_criteria: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()

    def __post_init__(self):
        """Validación automática y inicialización de campos"""
        if self.created_at is None:
            object.__setattr__(self, 'created_at', datetime.now())
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        
        # Validaciones
        if not self.id or not self.topic_id:
            raise ValidationError("ID y topic_id son requeridos")
        if len(self.title) < 5:
            raise ValidationError("El título debe tener al menos 5 caracteres")

    @cached_property
    def word_count(self) -> int:
        """Conteo de palabras del contenido"""
        return len(self.content.split())

    @cached_property
    def complexity_score(self) -> float:
        """Puntuación de complejidad basada en múltiples factores"""
        base_score = self.difficulty.level
        content_factor = min(self.word_count / 1000, 2.0)
        objectives_factor = len(self.learning_objectives) * 0.1
        return round(base_score + content_factor + objectives_factor, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Conversión optimizada a diccionario"""
        data = asdict(self)
        # Conversión de enums a valores serializables
        data['prompt_type'] = self.prompt_type.name
        data['difficulty'] = self.difficulty.display_name
        data['created_at'] = self.created_at.isoformat()
        return data

    def matches_criteria(self, **criteria) -> bool:
        """Verificación flexible de criterios de búsqueda"""
        for key, value in criteria.items():
            if hasattr(self, key):
                attr_value = getattr(self, key)
                if isinstance(value, (list, tuple)):
                    if attr_value not in value:
                        return False
                elif attr_value != value:
                    return False
        return True

class PromptGenerator:
    """
    Generador avanzado de prompts para aprendizaje profundo en ciberseguridad
    
    Mejoras principales:
    - Cache inteligente para templates y prompts generados
    - Generación batch para mejor performance
    - Sistema de plugins para tipos de prompts personalizados
    - Métricas avanzadas y analytics
    - Serialización optimizada
    """
    
    def __init__(self, cache_size: int = 128, enable_analytics: bool = True):
        """
        Inicialización optimizada con configuración flexible
        
        Args:
            cache_size: Tamaño del cache LRU
            enable_analytics: Habilitar recolección de métricas
        """
        # Lazy loading de SecurityTopics para mejor startup
        self._security_topics = None
        self.cache_size = cache_size
        self.enable_analytics = enable_analytics
        
        # Collections optimizadas
        self.generated_prompts: List[LearningPrompt] = []
        self._generation_stats = {
            'total_generated': 0,
            'generation_times': [],
            'error_count': 0,
            'cache_hits': 0
        }
        
        # Cache de templates inicializado bajo demanda
        self._template_cache: Optional[TemplateData] = None

    @property
    def security_topics(self) -> 'SecurityTopics':
        """Lazy loading de SecurityTopics para mejor performance"""
        if self._security_topics is None:
            if SecurityTopics is None:
                raise ImportError("SecurityTopics no disponible")
            self._security_topics = SecurityTopics()
        return self._security_topics

    @lru_cache(maxsize=64)  # Cache para templates compilados
    def _get_template_data(self, prompt_type: str) -> Dict[str, Any]:
        """
        Obtención cacheada de datos de template
        
        Mejora: Cache LRU para evitar recompilación de templates
        """
        if self._template_cache is None:
            self._template_cache = self._initialize_enhanced_templates()
        return self._template_cache.get(prompt_type, {})

    def _initialize_enhanced_templates(self) -> TemplateData:
        """
        Inicialización mejorada de templates con mayor flexibilidad
        
        Cambios:
        - Templates más modulares y reutilizables
        - Soporte para variables dinámicas
        - Plantillas específicas por nivel de dificultad
        """
        base_instructions = {
            'basic': [
                "Proporciona explicaciones claras y concisas",
                "Utiliza ejemplos simples y relevantes",
                "Evita jerga técnica compleja",
                "Incluye referencias para profundizar"
            ],
            'intermediate': [
                "Desarrolla explicaciones técnicas detalladas",
                "Incluye análisis comparativo",
                "Proporciona múltiples enfoques de solución",
                "Relaciona con casos de uso reales"
            ],
            'advanced': [
                "Presenta análisis crítico profundo",
                "Evalúa trade-offs y limitaciones",
                "Propone optimizaciones innovadoras",
                "Considera implicaciones futuras"
            ]
        }

        return {
            "conceptual": {
                "template": self._build_conceptual_template(),
                "questions": self._get_conceptual_questions(),
                "instructions": base_instructions,
                "evaluation_rubric": {
                    "understanding": "Comprensión de conceptos fundamentales",
                    "application": "Aplicación práctica del conocimiento",
                    "analysis": "Capacidad de análisis crítico",
                    "synthesis": "Síntesis e integración de ideas"
                }
            },
            
            "practico": {
                "template": self._build_practical_template(),
                "scenarios": self._get_practical_scenarios(),
                "instructions": base_instructions,
                "tools": {
                    "security": ["Nmap", "Wireshark", "Metasploit", "Burp Suite"],
                    "development": ["VS Code", "Git", "Docker", "Jenkins"],
                    "analysis": ["Splunk", "ELK Stack", "SIEM tools"]
                }
            },
            
            "codigo": {
                "template": self._build_code_template(),
                "challenges": self._get_code_challenges(),
                "languages": {
                    "python": {"framework": "Flask/Django", "security": "cryptography, hashlib"},
                    "javascript": {"framework": "Node.js/React", "security": "crypto, bcrypt"},
                    "java": {"framework": "Spring Boot", "security": "Spring Security"},
                    "go": {"framework": "Gin/Echo", "security": "crypto/tls"}
                },
                "security_patterns": [
                    "Input validation",
                    "Output encoding",
                    "Authentication",
                    "Authorization",
                    "Session management",
                    "Cryptography",
                    "Error handling",
                    "Logging"
                ]
            },
            
            # Templates expandidos para nuevos tipos
            "investigacion": {
                "template": self._build_research_template(),
                "methodologies": ["Systematic review", "Case study", "Experimental", "Survey"],
                "databases": ["ACM Digital Library", "IEEE Xplore", "Google Scholar", "ArXiv"]
            },
            
            "laboratorio": {
                "template": self._build_lab_template(),
                "environments": ["Virtual machines", "Containers", "Cloud sandboxes"],
                "tools": ["VirtualBox", "VMware", "Docker", "AWS/Azure labs"]
            }
        }

    def _build_conceptual_template(self) -> str:
        """Template mejorado para prompts conceptuales"""
        return """
# {topic_title} - Análisis Conceptual
**Nivel:** {level} | **Categoría:** {category} | **Duración:** {duration} min

## Pregunta Central
{question}

## Contexto Educativo
{context}

## Objetivos de Aprendizaje
{objectives}

## Instrucciones Específicas
{level_instructions}

## Criterios de Evaluación
{evaluation_criteria}

## Recursos Recomendados
{resources}

## Extensiones Opcionales
{optional_extensions}
"""

    def _build_practical_template(self) -> str:
        """Template mejorado para ejercicios prácticos"""
        return """
# {topic_title} - Ejercicio Práctico
**Nivel:** {level} | **Duración:** {duration} min | **Tipo:** Hands-on

## Escenario
{scenario}

## Objetivo Principal
{objective}

## Entorno de Trabajo
- **Herramientas:** {tools}
- **Plataformas:** {platforms}
- **Requisitos:** {requirements}

## Fases del Ejercicio
{phases}

## Criterios de Éxito
{success_criteria}

## Documentación Requerida
{documentation_requirements}

## Recursos y Referencias
{resources}
"""

    def _build_code_template(self) -> str:
        """Template mejorado para desafíos de código"""
        return """
# {topic_title} - Desafío de Programación
**Lenguaje:** {language} | **Nivel:** {level} | **Duración:** {duration} min

## Descripción del Desafío
{challenge_description}

## Especificaciones Técnicas
{technical_specs}

## Requisitos de Seguridad
{security_requirements}

## Estructura del Proyecto
{project_structure}

## Casos de Prueba
{test_cases}

## Código Base (Opcional)