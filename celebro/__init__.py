"""
@celebro - Sistema de Interpretación de Respuestas de IAs
Versión: 0.6.0
Cerebro interpretativo de LucIA para análisis y transformación de respuestas
"""

from .response_analyzer import ResponseAnalyzer
from .context_processor import ContextProcessor
from .response_generator import ResponseGenerator
from .knowledge_synthesizer import KnowledgeSynthesizer
from .celebro_core import CelebroCore

__version__ = "0.6.0"
__author__ = "LucIA Development Team"

__all__ = [
    "ResponseAnalyzer",
    "ContextProcessor", 
    "ResponseGenerator",
    "KnowledgeSynthesizer",
    "CelebroCore"
]
