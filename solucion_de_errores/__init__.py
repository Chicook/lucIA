"""
Sistema de Solución de Errores Críticos - LucIA v0.6.0
Módulo que soluciona errores críticos en sincronización asíncrona,
optimización de TensorFlow, predicciones y validación de sistemas.
"""

__version__ = "0.6.0"
__author__ = "LucIA Error Resolution System"

# Importar todos los solucionadores
from .async_sync_fixer import AsyncSyncFixer
from .tensorflow_optimizer import TensorFlowOptimizer
from .prediction_enhancer import PredictionEnhancer
from .system_validator import SystemValidator
from .error_monitor import ErrorMonitor
from .error_resolution_system import ErrorResolutionSystem
from .integration import ErrorResolutionIntegration

__all__ = [
    'AsyncSyncFixer',
    'TensorFlowOptimizer', 
    'PredictionEnhancer',
    'SystemValidator',
    'ErrorMonitor',
    'ErrorResolutionSystem',
    'ErrorResolutionIntegration'
]
