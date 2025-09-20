"""
Sistema de Conocimientos para @red_neuronal
Versión: 0.6.0
Sistema de creación de prompts para aprendizaje profundo en ciberseguridad
"""

from .prompt_generator import PromptGenerator
from .knowledge_base import KnowledgeBase
from .security_topics import SecurityTopics
from .learning_curriculum import LearningCurriculum
from .deep_learning_trainer import DeepLearningTrainer

__version__ = "0.6.0"
__author__ = "LucIA Development Team"

__all__ = [
    "PromptGenerator",
    "KnowledgeBase", 
    "SecurityTopics",
    "LearningCurriculum",
    "DeepLearningTrainer"
]
