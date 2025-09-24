"""
Temas de Seguridad en Internet
Versión: 0.7.0
Base de conocimientos sobre ciberseguridad para entrenamiento de IA
Arquitectura modular mejorada con mejor gestión de datos y funcionalidades extendidas
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Iterator, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, Counter
from functools import lru_cache, wraps
from datetime import datetime
import hashlib
import logging
import threading
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de dificultad en seguridad con valores numéricos para ordenamiento"""
    BASICO = ("básico", 1)
    INTERMEDIO = ("intermedio", 2)
    AVANZADO = ("avanzado", 3)
    EXPERTO = ("experto", 4)
    
    def __init__(self, label: str, value: int):
        self.label = label
        self.numeric_value = value
    
    def __lt__(self, other):
        return self.numeric_value < other.numeric_value
    
    @classmethod
    def from_string(cls, level_str: str) -> 'SecurityLevel':
        """Convierte string a SecurityLevel"""
        for level in cls:
            if level.label.lower() == level_str.lower():
                return level
        raise ValueError(f"Nivel de seguridad no válido: {level_str}")

class TopicCategory(Enum):
    """Categorías de temas de seguridad con metadatos adicionales"""
    CONCEPTOS_BASICOS = ("conceptos_basicos", "Conceptos Básicos", "🔐")
    AMENAZAS = ("amenazas", "Amenazas y Vulnerabilidades", "⚠️")
    DEFENSAS = ("defensas", "Sistemas de Defensa", "🛡️")
    HERRAMIENTAS = ("herramientas", "Herramientas de Seguridad", "🔧")
    LEGISLACION = ("legislacion", "Legislación y Compliance", "⚖️")
    MEJORES_PRACTICAS = ("mejores_practicas", "Mejores Prácticas", "✅")
    CODIGO_SEGURO = ("codigo_seguro", "Desarrollo Seguro", "💻")
    FORENSE = ("forense", "Análisis Forense", "🔍")
    REDES = ("redes", "Seguridad de Redes", "🌐")
    CLOUD = ("cloud", "Seguridad en la Nube", "☁️")
    
    def __init__(self, value: str, display_name: str, icon: str):
        self.value = value
        self.display_name = display_name
        self.icon = icon

class SecurityFramework(Enum):
    """Frameworks de seguridad reconocidos"""
    OWASP = "OWASP"
    NIST = "NIST"
    ISO27001 = "ISO 27001"
    CIS = "CIS Controls"
    MITRE = "MITRE ATT&CK"
    SANS = "SANS"

@dataclass
class SecurityResource:
    """Recurso de seguridad con metadatos mejorados"""
    title: str
    url: str
    resource_type: str  # "documentation", "tool", "course", "article", "video"
    framework: Optional[SecurityFramework] = None
    language: str = "es"
    difficulty: Optional[SecurityLevel] = None
    last_updated: Optional[datetime] = None
    reliability_score: float = 0.0
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class CodeExample:
    """Ejemplo de código con metadatos extendidos"""
    title: str
    code: str
    language: str
    description: str
    tags: List[str] = field(default_factory=list)
    complexity: SecurityLevel = SecurityLevel.BASICO
    dependencies: List[str] = field(default_factory=list)
    security_notes: List[str] = field(default_factory=list)
    
    def get_hash(self) -> str:
        """Genera hash único para el ejemplo de código"""
        content = f"{self.title}{self.code}{self.language}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class LearningPath:
    """Ruta de aprendizaje estructurada"""
    name: str
    description: str
    topics: List[str]  # IDs de temas
    estimated_hours: int
    prerequisites: List[str] = field(default_factory=list)
    target_audience: str = "general"
    
    def get_difficulty_progression(self) -> List[SecurityLevel]:
        """Obtiene la progresión de dificultad"""
        # Se implementará con acceso a SecurityTopics
        return []

@dataclass
class SecurityTopic:
    """Estructura mejorada de un tema de seguridad"""
    id: str
    title: str
    category: TopicCategory
    level: SecurityLevel
    description: str
    keywords: List[str]
    learning_objectives: List[str]
    practical_examples: List[str]
    code_examples: List[CodeExample]
    resources: List[SecurityResource]
    
    # Nuevos campos
    frameworks: List[SecurityFramework] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # IDs de otros temas
    related_topics: List[str] = field(default_factory=list)  # IDs de temas relacionados
    estimated_time: int = 60  # minutos
    last_updated: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    difficulty_score: float = 0.0
    popularity_score: float = 0.0
    
    def __post_init__(self):
        # Convertir listas a objetos apropiados si es necesario
        if self.code_examples and isinstance(self.code_examples[0], str):
            self.code_examples = [
                CodeExample(
                    title=f"Ejemplo {i+1}",
                    code=code,
                    language="python",
                    description=f"Ejemplo práctico para {self.title}"
                ) for i, code in enumerate(self.code_examples)
            ]
        
        if self.resources and isinstance(self.resources[0], str):
            self.resources = [
                SecurityResource(
                    title=resource,
                    url="",
                    resource_type="documentation"
                ) for resource in self.resources
            ]
        
        # Calcular score de dificultad basado en el nivel
        self.difficulty_score = self.level.numeric_value
        
        # Generar tags automáticamente
        self.tags.update(self.keywords)
        self.tags.add(self.category.value)
        self.tags.add(self.level.label)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el tema a diccionario"""
        return asdict(self)
    
    def get_complexity_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de complejidad del tema"""
        return {
            "level": self.level.label,
            "difficulty_score": self.difficulty_score,
            "num_objectives": len(self.learning_objectives),
            "num_examples": len(self.practical_examples),
            "num_code_examples": len(self.code_examples),
            "estimated_time": self.estimated_time,
            "has_prerequisites": len(self.prerequisites) > 0
        }

class SecurityTopicsInterface(ABC):
    """Interface para sistemas de temas de seguridad"""
    
    @abstractmethod
    def get_topic(self, topic_id: str) -> Optional[SecurityTopic]:
        pass
    
    @abstractmethod
    def search_topics(self, query: str) -> List[SecurityTopic]:
        pass
    
    @abstractmethod
    def get_learning_path(self, start_level: SecurityLevel) -> List[SecurityTopic]:
        pass

class TopicFactory:
    """Factory para crear temas de seguridad de manera consistente"""
    
    @staticmethod
    def create_topic(
        id: str,
        title: str,
        category: TopicCategory,
        level: SecurityLevel,
        description: str,
        **kwargs
    ) -> SecurityTopic:
        """Crea un tema con validaciones"""
        if not id or not title:
            raise ValueError("ID y título son obligatorios")
        
        defaults = {
            'keywords': [],
            'learning_objectives': [],
            'practical_examples': [],
            'code_examples': [],
            'resources': []
        }
        defaults.update(kwargs)
        
        return SecurityTopic(
            id=id,
            title=title,
            category=category,
            level=level,
            description=description,
            **defaults
        )

class SecurityTopicsCache:
    """Sistema de caché para mejorar rendimiento"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_count: Counter = Counter()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._access_count[key] += 1
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remover el elemento menos accedido
                least_used = self._access_count.most_common()[-1][0]
                del self._cache[least_used]
                del self._access_count[least_used]
            
            self._cache[key] = value
            self._access_count[key] = 1
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_count.clear()

class SecurityTopics(SecurityTopicsInterface):
    """Base de conocimientos mejorada sobre seguridad en internet"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("security_data")
        self.data_path.mkdir(exist_ok=True)
        
        self.topics: Dict[str, SecurityTopic] = {}
        self.learning_paths: Dict[str, LearningPath] = {}
        self.cache = SecurityTopicsCache()
        
        # Índices para búsqueda eficiente
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[TopicCategory, Set[str]] = defaultdict(set)
        self._level_index: Dict[SecurityLevel, Set[str]] = defaultdict(set)
        self._framework_index: Dict[SecurityFramework, Set[str]] = defaultdict(set)
        
        self._initialize_topics()
        self._build_indices()
        self._create_learning_paths()
    
    def _initialize_topics(self) -> None:
        """Inicializa todos los temas de seguridad con contenido extendido"""
        
        # Cargar desde archivo si existe
        topics_file = self.data_path / "topics.json"
        if topics_file.exists():
            try:
                self._load_from_file(topics_file)
                return
            except Exception as e:
                logger.warning(f"Error cargando topics desde archivo: {e}")
        
        # Inicializar temas por defecto (código existente mejorado)
        self._create_default_topics()
        
        # Guardar en archivo
        self._save_to_file(topics_file)
    
    def _create_default_topics(self) -> None:
        """Crea los temas por defecto con contenido mejorado"""
        
        # CONCEPTOS BÁSICOS - Ampliados
        self.topics["autenticacion"] = TopicFactory.create_topic(
            id="autenticacion",
            title="Autenticación y Autorización",
            category=TopicCategory.CONCEPTOS_BASICOS,
            level=SecurityLevel.BASICO,
            description="Fundamentos completos de autenticación y autorización en sistemas informáticos modernos",
            keywords=["autenticación", "autorización", "login", "password", "2FA", "MFA", "SSO", "OAuth", "SAML"],
            learning_objectives=[
                "Distinguir entre autenticación, autorización y accounting",
                "Implementar sistemas de autenticación multifactor robustos",
                "Diseñar esquemas de autorización basados en roles y atributos",
                "Integrar sistemas de single sign-on (SSO)",
                "Aplicar principios de least privilege y defense in depth"
            ],
            practical_examples=[
                "Sistema de login con hash bcrypt y salt",
                "Implementación completa de 2FA con TOTP y backup codes",
                "RBAC con herencia de roles y permisos granulares",
                "SSO con OAuth 2.0 y OpenID Connect",
                "Autenticación biométrica en aplicaciones móviles"
            ],
            code_examples=[
                CodeExample(
                    title="Sistema de Hash Seguro",
                    code="""
import bcrypt
import secrets

class SecurePasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        return secrets.token_urlsafe(length)
                    """,
                    language="python",
                    description="Implementación segura de hashing de contraseñas",
                    tags=["bcrypt", "hashing", "security"],
                    complexity=SecurityLevel.INTERMEDIO,
                    dependencies=["bcrypt"],
                    security_notes=[
                        "Usar siempre salt aleatorio",
                        "Configurar rounds apropiados para el hardware",
                        "Nunca almacenar contraseñas en texto plano"
                    ]
                )
            ],
            resources=[
                SecurityResource(
                    title="OWASP Authentication Cheat Sheet",
                    url="https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
                    resource_type="documentation",
                    framework=SecurityFramework.OWASP,
                    difficulty=SecurityLevel.INTERMEDIO
                )
            ],
            frameworks=[SecurityFramework.OWASP, SecurityFramework.NIST],
            estimated_time=120,
            tags={"authentication", "authorization", "security basics"}
        )
        
        # Continuar con todos los demás temas mejorados...
        self._create_advanced_topics()
    
    def _create_advanced_topics(self) -> None:
        """Crea temas avanzados adicionales"""
        
        # NUEVO: Seguridad en DevSecOps
        self.topics["devsecops"] = TopicFactory.create_topic(
            id="devsecops",
            title="DevSecOps - Seguridad en CI/CD",
            category=TopicCategory.MEJORES_PRACTICAS,
            level=SecurityLevel.AVANZADO,
            description="Integración de seguridad en pipelines de desarrollo y despliegue continuo",
            keywords=["DevSecOps", "CI/CD", "pipeline security", "SAST", "DAST", "container security"],
            learning_objectives=[
                "Implementar security gates en pipelines CI/CD",
                "Automatizar análisis de seguridad de código",
                "Gestionar secretos en entornos de desarrollo",
                "Monitorear vulnerabilidades en tiempo real"
            ],
            frameworks=[SecurityFramework.OWASP, SecurityFramework.NIST],
            estimated_time=180
        )
        
        # NUEVO: Threat Intelligence
        self.topics["threat_intelligence"] = TopicFactory.create_topic(
            id="threat_intelligence",
            title="Threat Intelligence y CTI",
            category=TopicCategory.AMENAZAS,
            level=SecurityLevel.EXPERTO,
            description="Recopilación, análisis y aplicación de inteligencia de amenazas",
            keywords=["CTI", "IOC", "TTP", "STIX", "TAXII", "threat hunting"],
            learning_objectives=[
                "Desarrollar programas de threat intelligence",
                "Analizar indicadores de compromiso (IOCs)",
                "Implementar sistemas STIX/TAXII",
                "Realizar threat hunting proactivo"
            ],
            frameworks=[SecurityFramework.MITRE],
            estimated_time=240
        )
    
    def _build_indices(self) -> None:
        """Construye índices para búsqueda eficiente"""
        logger.info("Construyendo índices de búsqueda...")
        
        for topic_id, topic in self.topics.items():
            # Índice de palabras clave
            for keyword in topic.keywords:
                self._keyword_index[keyword.lower()].add(topic_id)
            
            # Índice de categorías
            self._category_index[topic.category].add(topic_id)
            
            # Índice de niveles
            self._level_index[topic.level].add(topic_id)
            
            # Índice de frameworks
            for framework in topic.frameworks:
                self._framework_index[framework].add(topic_id)
        
        logger.info(f"Índices construidos para {len(self.topics)} temas")
    
    def _create_learning_paths(self) -> None:
        """Crea rutas de aprendizaje estructuradas"""
        
        # Ruta básica de seguridad
        self.learning_paths["security_fundamentals"] = LearningPath(
            name="Fundamentos de Seguridad",
            description="Ruta básica para iniciarse en ciberseguridad",
            topics=["autenticacion", "encriptacion", "phishing"],
            estimated_hours=8,
            target_audience="principiantes"
        )
        
        # Ruta avanzada
        self.learning_paths["advanced_security"] = LearningPath(
            name="Seguridad Avanzada",
            description="Ruta para profesionales de seguridad",
            topics=["ids_ips", "vulnerability_assessment", "devsecops", "threat_intelligence"],
            estimated_hours=20,
            prerequisites=["security_fundamentals"],
            target_audience="profesionales"
        )
    
    @lru_cache(maxsize=128)
    def get_topic(self, topic_id: str) -> Optional[SecurityTopic]:
        """Obtiene un tema específico (con caché)"""
        return self.topics.get(topic_id)
    
    def get_topics_by_category(self, category: TopicCategory) -> List[SecurityTopic]:
        """Obtiene temas por categoría usando índice"""
        topic_ids = self._category_index.get(category, set())
        return [self.topics[tid] for tid in topic_ids]
    
    def get_topics_by_level(self, level: SecurityLevel) -> List[SecurityTopic]:
        """Obtiene temas por nivel usando índice"""
        topic_ids = self._level_index.get(level, set())
        return [self.topics[tid] for tid in topic_ids]
    
    def get_topics_by_framework(self, framework: SecurityFramework) -> List[SecurityTopic]:
        """Obtiene temas por framework"""
        topic_ids = self._framework_index.get(framework, set())
        return [self.topics[tid] for tid in topic_ids]
    
    def search_topics(self, 
                     query: str,
                     category: Optional[TopicCategory] = None,
                     level: Optional[SecurityLevel] = None,
                     framework: Optional[SecurityFramework] = None) -> List[SecurityTopic]:
        """Búsqueda avanzada de temas con filtros"""
        
        cache_key = f"search_{query}_{category}_{level}_{framework}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        query_lower = query.lower()
        matching_topics = set()
        
        # Búsqueda en índice de palabras clave
        for keyword, topic_ids in self._keyword_index.items():
            if query_lower in keyword:
                matching_topics.update(topic_ids)
        
        # Búsqueda en títulos y descripciones
        for topic_id, topic in self.topics.items():
            if (query_lower in topic.title.lower() or 
                query_lower in topic.description.lower()):
                matching_topics.add(topic_id)
        
        # Aplicar filtros
        filtered_topics = []
        for topic_id in matching_topics:
            topic = self.topics[topic_id]
            
            if category and topic.category != category:
                continue
            if level and topic.level != level:
                continue
            if framework and framework not in topic.frameworks:
                continue
            
            filtered_topics.append(topic)
        
        # Ordenar por relevancia
        filtered_topics.sort(key=lambda t: t.popularity_score, reverse=True)
        
        self.cache.set(cache_key, filtered_topics)
        return filtered_topics
    
    def get_learning_path(self, 
                         path_name: Optional[str] = None,
                         start_level: SecurityLevel = SecurityLevel.BASICO) -> List[SecurityTopic]:
        """Obtiene una ruta de aprendizaje"""
        
        if path_name and path_name in self.learning_paths:
            path = self.learning_paths[path_name]
            return [self.topics[tid] for tid in path.topics if tid in self.topics]
        
        # Ruta automática por nivel
        levels = [SecurityLevel.BASICO, SecurityLevel.INTERMEDIO, 
                 SecurityLevel.AVANZADO, SecurityLevel.EXPERTO]
        start_index = levels.index(start_level)
        
        learning_path = []
        for level in levels[start_index:]:
            level_topics = self.get_topics_by_level(level)
            # Ordenar por popularidad y prerequisitos
            level_topics.sort(key=lambda t: (len(t.prerequisites), -t.popularity_score))
            learning_path.extend(level_topics[:3])  # Máximo 3 por nivel
        
        return learning_path
    
    def get_topic_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas de los temas"""
        total_topics = len(self.topics)
        
        by_category = {cat.value: len(self._category_index[cat]) 
                      for cat in TopicCategory}
        
        by_level = {level.label: len(self._level_index[level]) 
                   for level in SecurityLevel}
        
        by_framework = {fw.value: len(self._framework_index[fw]) 
                       for fw in SecurityFramework}
        
        avg_time = sum(t.estimated_time for t in self.topics.values()) / total_topics
        
        return {
            "total_topics": total_topics,
            "by_category": by_category,
            "by_level": by_level,
            "by_framework": by_framework,
            "average_estimated_time": avg_time,
            "total_code_examples": sum(len(t.code_examples) for t in self.topics.values()),
            "total_resources": sum(len(t.resources) for t in self.topics.values()),
            "learning_paths": len(self.learning_paths)
        }
    
    def add_topic(self, topic: SecurityTopic) -> None:
        """Añade un nuevo tema al sistema"""
        if topic.id in self.topics:
            logger.warning(f"Tema {topic.id} ya existe, será reemplazado")
        
        self.topics[topic.id] = topic
        self._update_indices_for_topic(topic)
        self.cache.clear()  # Limpiar caché
        
        logger.info(f"Tema {topic.id} añadido exitosamente")
    
    def update_topic(self, topic_id: str, updates: Dict[str, Any]) -> bool:
        """Actualiza un tema existente"""
        if topic_id not in self.topics:
            return False
        
        topic = self.topics[topic_id]
        for key, value in updates.items():
            if hasattr(topic, key):
                setattr(topic, key, value)
        
        topic.last_updated = datetime.now()
        self._update_indices_for_topic(topic)
        self.cache.clear()
        
        return True
    
    def _update_indices_for_topic(self, topic: SecurityTopic) -> None:
        """Actualiza índices para un tema específico"""
        topic_id = topic.id
        
        # Limpiar índices existentes para este tema
        for keyword_set in self._keyword_index.values():
            keyword_set.discard(topic_id)
        for category_set in self._category_index.values():
            category_set.discard(topic_id)
        for level_set in self._level_index.values():
            level_set.discard(topic_id)
        for framework_set in self._framework_index.values():
            framework_set.discard(topic_id)
        
        # Rebuilder índices para este tema
        for keyword in topic.keywords:
            self._keyword_index[keyword.lower()].add(topic_id)
        
        self._category_index[topic.category].add(topic_id)
        self._level_index[topic.level].add(topic_id)
        
        for framework in topic.frameworks:
            self._framework_index[framework].add(topic_id)
    
    def export_topics(self, format: str = "json") -> str:
        """Exporta temas en diferentes formatos"""
        if format.lower() == "json":
            return json.dumps([topic.to_dict() for topic in self.topics.values()], 
                            indent=2, ensure_ascii=False, default=str)
        else:
            raise ValueError(f"Formato no soportado: {format}")
    
    def _save_to_file(self, file_path: Path) -> None:
        """Guarda temas en archivo"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([topic.to_dict() for topic in self.topics.values()], 
                         f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Temas guardados en {file_path}")
        except Exception as e:
            logger.error(f"Error guardando temas: {e}")
    
    def _load_from_file(self, file_path: Path) -> None:
        """Carga temas desde archivo"""
        with open(file_path, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        
        for topic_data in topics_data:
            # Convertir de dict a objetos
            topic = self._dict_to_topic(topic_data)
            self.topics[topic.id] = topic
        
        logger.info(f"Cargados {len(self.topics)} temas desde {file_path}")
    
    def _dict_to_topic(self, topic_data: Dict[str, Any]) -> SecurityTopic:
        """Convierte diccionario a SecurityTopic"""
        # Implementar conversión completa desde dict
        # Esta es una versión simplificada
        return SecurityTopic(**topic_data)
    
    def get_topic_recommendations(self, topic_id: str, limit: int = 5) -> List[SecurityTopic]:
        """Obtiene recomendaciones de temas relacionados"""
        if topic_id not in self.topics:
            return []
        
        current_topic = self.topics[topic_id]
        recommendations = []
        
        # Recomendar temas relacionados
        for related_id in current_topic.related_topics:
            if related_id in self.topics:
                recommendations.append(self.topics[related_id])
        
        # Recomendar por categoría similar
        if len(recommendations) < limit:
            category_topics = self.get_topics_by_category(current_topic.category)
            for topic in category_topics:
                if topic.id != topic_id and topic not in recommendations:
                    recommendations.append(topic)
                    if len(recommendations) >= limit:
                        break
        
        return recommendations[:limit]

# Funciones de utilidad para importación
def create_security_topics_instance(data_path: Optional[str] = None) -> SecurityTopics:
    """Factory function para crear instancia de SecurityTopics"""
    path = Path(data_path) if data_path else None
    return SecurityTopics(path)

def get_all_categories() -> List[TopicCategory]:
    """Obtiene todas las categorías disponibles"""
    return list(TopicCategory)

def get_all_levels() -> List[SecurityLevel]:
    """Obtiene todos los niveles disponibles"""
    return list(SecurityLevel)

def get_all_frameworks() -> List[SecurityFramework]:
    """Obtiene todos los frameworks disponibles"""
    return list(SecurityFramework)

# Singleton instance para uso global
_security_topics_instance = None

def get_security_topics_instance() -> SecurityTopics:
    """Obtiene instancia singleton de SecurityTopics"""
    global _security_topics_instance
    if _security_topics_instance is None:
        _security_topics_instance = SecurityTopics()
    return _security_topics_instance

if __name__ == "__main__":
    # Ejemplo de uso
    security_topics = SecurityTopics()
    
    # Obtener estadísticas
    stats = security_topics.get_topic_statistics()
    print(f"Total de temas: {stats['total_topics']}")
    
    # Buscar temas
    results = security_topics.search_topics("autenticación")
    print(f"Encontrados {len(results)} temas sobre autenticación")
    
    # Obtener ruta de aprendizaje
    path = security_topics.get_learning_path("security_fundamentals")
    print(f"Ruta de aprendizaje con {len(path)} temas")