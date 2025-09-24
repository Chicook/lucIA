"""
Sistema de Conocimientos para @red_neuronal
Versión: 0.6.0
Sistema de creación de prompts para aprendizaje profundo en ciberseguridad

Este módulo proporciona las herramientas necesarias para la generación de conocimientos,
entrenamiento y gestión de contenido educativo en el ámbito de la ciberseguridad.
"""

import logging
from typing import List

# Configuración de logging para el módulo
logger = logging.getLogger(__name__)

try:
    from .prompt_generator import PromptGenerator
    from .knowledge_base import KnowledgeBase
    from .security_topics import SecurityTopics
    from .learning_curriculum import LearningCurriculum
    from .deep_learning_trainer import DeepLearningTrainer
except ImportError as e:
    logger.error(f"Error al importar módulos: {e}")
    raise

# Metadatos del paquete
__version__ = "0.6.0"
__author__ = "LucIA Development Team"
__description__ = "Sistema de conocimientos para red neuronal en ciberseguridad"
__license__ = "MIT"

# Lista de exportaciones públicas
__all__: List[str] = [
    "PromptGenerator",
    "KnowledgeBase", 
    "SecurityTopics",
    "LearningCurriculum",
    "DeepLearningTrainer"
]

def get_version() -> str:
    """Retorna la versión actual del módulo."""
    return __version__

def get_available_modules() -> List[str]:
    """Retorna una lista de módulos disponibles."""
    return __all__.copy()
