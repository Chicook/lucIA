"""
Paquete Red Neuronal Profunda (DNN) - LucIA Project
==================================================

Este paquete implementa una Red Neuronal Profunda completamente conectada
usando TensorFlow/Keras para el proyecto LucIA. Incluye módulos para la
definición de arquitectura, generación de datos simulados y entrenamiento.

Módulos principales:
- modelo: Definición de la arquitectura DNN
- datos: Generación de datos simulados
- entrenar: Script principal de entrenamiento

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

__version__ = "1.0.0"
__author__ = "LucIA Development Team"
__email__ = "lucia@metaverso.com"

# Imports principales
try:
    from .modelo import crear_modelo, get_model_info, validate_model_architecture
    from .datos import cargar_datos_simulados, get_data_statistics
    from .entrenar import DNNTrainer
    
    __all__ = [
        'crear_modelo',
        'get_model_info', 
        'validate_model_architecture',
        'cargar_datos_simulados',
        'get_data_statistics',
        'DNNTrainer'
    ]
    
except ImportError as e:
    # En caso de que falten dependencias, solo registrar el error
    import warnings
    warnings.warn(f"No se pudieron importar todos los módulos: {e}")
    __all__ = []

# Información del paquete
__package_info__ = {
    'name': 'RD_neuronal_LC',
    'version': __version__,
    'description': 'Red Neuronal Profunda para LucIA',
    'author': __author__,
    'email': __email__,
    'modules': ['modelo', 'datos', 'entrenar'],
    'requirements': [
        'tensorflow>=2.13.0',
        'keras>=2.13.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0'
    ]
}
