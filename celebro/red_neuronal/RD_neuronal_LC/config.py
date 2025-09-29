"""
Archivo de Configuración para Red Neuronal Profunda (DNN)
========================================================

Este archivo contiene todas las configuraciones centralizadas para el módulo DNN,
incluyendo parámetros de arquitectura, entrenamiento y datos. Permite modificar
fácilmente el comportamiento del sistema sin cambiar el código principal.

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

from typing import Dict, Any, List
import os

# ============================================================================
# CONFIGURACIÓN DE ARQUITECTURA DE LA RED NEURONAL
# ============================================================================

ARCHITECTURE_CONFIG = {
    'input_shape': 8,                    # Número de características de entrada
    'num_classes': 4,                    # Número de clases de salida
    'hidden_layers': [32, 16, 8],        # Tamaños de capas ocultas (decrecientes)
    'activation_hidden': 'relu',         # Función de activación para capas ocultas
    'activation_output': 'softmax',      # Función de activación para capa de salida
    'dropout_rate': 0.0,                 # Tasa de dropout (0 = sin dropout)
    'use_batch_normalization': False,    # Usar normalización por lotes
    'use_regularization': False,         # Usar regularización L1/L2
    'regularization_factor': 0.001       # Factor de regularización
}

# ============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# ============================================================================

TRAINING_CONFIG = {
    'epochs': 10,                        # Número de épocas de entrenamiento
    'batch_size': 32,                    # Tamaño del lote
    'validation_split': 0.2,             # Proporción de datos para validación
    'learning_rate': 0.001,              # Tasa de aprendizaje inicial
    'optimizer': 'adam',                 # Optimizador (adam, sgd, rmsprop)
    'loss_function': 'categorical_crossentropy',  # Función de pérdida
    'metrics': ['accuracy', 'precision', 'recall'],  # Métricas a monitorear
    'shuffle': True,                     # Mezclar datos en cada época
    'verbose': 1                         # Nivel de verbosidad (0, 1, 2)
}

# ============================================================================
# CONFIGURACIÓN DE CALLBACKS
# ============================================================================

CALLBACKS_CONFIG = {
    'use_early_stopping': True,          # Usar early stopping
    'early_stopping_patience': 3,        # Paciencia para early stopping
    'early_stopping_monitor': 'val_loss', # Métrica a monitorear
    'restore_best_weights': True,        # Restaurar mejores pesos
    
    'use_reduce_lr': True,               # Reducir tasa de aprendizaje
    'reduce_lr_patience': 2,             # Paciencia para reducir LR
    'reduce_lr_factor': 0.5,             # Factor de reducción de LR
    'reduce_lr_min': 1e-7,               # LR mínimo
    
    'use_model_checkpoint': True,        # Guardar mejor modelo
    'checkpoint_monitor': 'val_accuracy', # Métrica para checkpoint
    'checkpoint_save_best_only': True,   # Solo guardar el mejor
    'checkpoint_filename': 'best_model.h5'  # Nombre del archivo
}

# ============================================================================
# CONFIGURACIÓN DE DATOS
# ============================================================================

DATA_CONFIG = {
    'n_samples': 1000,                   # Número de muestras a generar
    'n_features': 8,                     # Número de características
    'n_classes': 4,                      # Número de clases
    'random_state': 42,                  # Semilla para reproducibilidad
    'normalize_features': True,          # Normalizar características
    'balance_classes': True,             # Balancear clases
    'add_noise': False,                  # Agregar ruido a los datos
    'noise_level': 0.05,                 # Nivel de ruido (si se activa)
    'correlation_threshold': 0.7         # Umbral de correlación entre características
}

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

VISUALIZATION_CONFIG = {
    'save_plots': True,                  # Guardar gráficos en archivos
    'show_plots': True,                  # Mostrar gráficos en pantalla
    'plot_format': 'png',                # Formato de gráficos (png, pdf, svg)
    'plot_dpi': 300,                     # Resolución de gráficos
    'figure_size': (15, 10),             # Tamaño de figuras
    'style': 'seaborn-v0_8',             # Estilo de matplotlib
    'color_palette': 'husl',             # Paleta de colores
    'font_size': 12,                     # Tamaño de fuente
    'line_width': 2                      # Grosor de líneas
}

# ============================================================================
# CONFIGURACIÓN DE ARCHIVOS Y DIRECTORIOS
# ============================================================================

PATHS_CONFIG = {
    'base_dir': os.path.dirname(__file__),
    'models_dir': 'models',
    'plots_dir': 'plots', 
    'reports_dir': 'reports',
    'logs_dir': 'logs',
    'data_dir': 'data',
    'temp_dir': 'temp'
}

# Nombres de archivos
FILE_NAMES = {
    'model_checkpoint': 'best_model.h5',
    'dataset_file': 'dataset_entrenamiento.npz',
    'training_history': 'training_history.png',
    'training_log': 'training.log',
    'report_template': 'training_report_{timestamp}.txt'
}

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',                     # Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,                 # Guardar logs en archivo
    'log_to_console': True,              # Mostrar logs en consola
    'log_file': 'training.log',          # Archivo de log
    'max_log_size': 10 * 1024 * 1024,   # Tamaño máximo de log (10MB)
    'backup_count': 5                    # Número de archivos de backup
}

# ============================================================================
# CONFIGURACIÓN DE RENDIMIENTO
# ============================================================================

PERFORMANCE_CONFIG = {
    'use_gpu': True,                     # Usar GPU si está disponible
    'gpu_memory_growth': True,           # Crecimiento dinámico de memoria GPU
    'mixed_precision': False,            # Usar precisión mixta
    'num_parallel_calls': 4,             # Llamadas paralelas para data loading
    'cache_dataset': True,               # Cachear dataset en memoria
    'prefetch_buffer_size': 2            # Tamaño del buffer de prefetch
}

# ============================================================================
# CONFIGURACIÓN DE VALIDACIÓN Y TESTING
# ============================================================================

TESTING_CONFIG = {
    'test_size': 0.1,                    # Proporción de datos para testing
    'cross_validation_folds': 5,         # Número de folds para CV
    'random_state_test': 123,            # Semilla para testing
    'confidence_threshold': 0.95,        # Umbral de confianza para predicciones
    'evaluation_metrics': [              # Métricas de evaluación
        'accuracy',
        'precision_macro',
        'recall_macro', 
        'f1_macro',
        'confusion_matrix'
    ]
}

# ============================================================================
# CONFIGURACIÓN COMPLETA
# ============================================================================

DNN_CONFIG = {
    'architecture': ARCHITECTURE_CONFIG,
    'training': TRAINING_CONFIG,
    'callbacks': CALLBACKS_CONFIG,
    'data': DATA_CONFIG,
    'visualization': VISUALIZATION_CONFIG,
    'paths': PATHS_CONFIG,
    'files': FILE_NAMES,
    'logging': LOGGING_CONFIG,
    'performance': PERFORMANCE_CONFIG,
    'testing': TESTING_CONFIG
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_config(section: str = None) -> Dict[str, Any]:
    """
    Obtiene la configuración completa o de una sección específica.
    
    Args:
        section (str): Sección específica a obtener (opcional)
        
    Returns:
        Dict[str, Any]: Configuración solicitada
    """
    if section is None:
        return DNN_CONFIG
    elif section in DNN_CONFIG:
        return DNN_CONFIG[section]
    else:
        raise ValueError(f"Sección '{section}' no encontrada. Secciones disponibles: {list(DNN_CONFIG.keys())}")

def update_config(section: str, updates: Dict[str, Any]) -> None:
    """
    Actualiza la configuración de una sección específica.
    
    Args:
        section (str): Sección a actualizar
        updates (Dict[str, Any]): Valores a actualizar
    """
    if section not in DNN_CONFIG:
        raise ValueError(f"Sección '{section}' no encontrada")
    
    DNN_CONFIG[section].update(updates)

def create_directories() -> None:
    """Crea todos los directorios necesarios según la configuración."""
    for dir_name in PATHS_CONFIG.values():
        if isinstance(dir_name, str) and not dir_name.startswith('/'):
            os.makedirs(dir_name, exist_ok=True)

def validate_config() -> List[str]:
    """
    Valida que la configuración sea correcta.
    
    Returns:
        List[str]: Lista de errores encontrados (vacía si no hay errores)
    """
    errors = []
    
    # Validar arquitectura
    arch = ARCHITECTURE_CONFIG
    if arch['input_shape'] <= 0:
        errors.append("input_shape debe ser positivo")
    if arch['num_classes'] <= 0:
        errors.append("num_classes debe ser positivo")
    if len(arch['hidden_layers']) == 0:
        errors.append("hidden_layers no puede estar vacío")
    if not all(size > 0 for size in arch['hidden_layers']):
        errors.append("Todos los tamaños de hidden_layers deben ser positivos")
    
    # Validar entrenamiento
    train = TRAINING_CONFIG
    if train['epochs'] <= 0:
        errors.append("epochs debe ser positivo")
    if train['batch_size'] <= 0:
        errors.append("batch_size debe ser positivo")
    if not (0 < train['validation_split'] < 1):
        errors.append("validation_split debe estar entre 0 y 1")
    if train['learning_rate'] <= 0:
        errors.append("learning_rate debe ser positivo")
    
    # Validar datos
    data = DATA_CONFIG
    if data['n_samples'] <= 0:
        errors.append("n_samples debe ser positivo")
    if data['n_features'] <= 0:
        errors.append("n_features debe ser positivo")
    if data['n_classes'] <= 0:
        errors.append("n_classes debe ser positivo")
    
    return errors

# ============================================================================
# CONFIGURACIONES PREDEFINIDAS
# ============================================================================

# Configuración para desarrollo rápido
QUICK_CONFIG = {
    'training': {'epochs': 2, 'batch_size': 16},
    'data': {'n_samples': 100},
    'callbacks': {'use_early_stopping': False, 'use_reduce_lr': False}
}

# Configuración para entrenamiento intensivo
INTENSIVE_CONFIG = {
    'training': {'epochs': 100, 'batch_size': 64},
    'callbacks': {'early_stopping_patience': 10, 'reduce_lr_patience': 5},
    'data': {'n_samples': 10000}
}

# Configuración para producción
PRODUCTION_CONFIG = {
    'training': {'epochs': 50, 'batch_size': 32, 'verbose': 0},
    'callbacks': {'use_early_stopping': True, 'use_reduce_lr': True},
    'visualization': {'show_plots': False, 'save_plots': True},
    'logging': {'level': 'WARNING', 'log_to_console': False}
}

def apply_preset_config(preset: str) -> None:
    """
    Aplica una configuración predefinida.
    
    Args:
        preset (str): Nombre del preset ('quick', 'intensive', 'production')
    """
    presets = {
        'quick': QUICK_CONFIG,
        'intensive': INTENSIVE_CONFIG,
        'production': PRODUCTION_CONFIG
    }
    
    if preset not in presets:
        raise ValueError(f"Preset '{preset}' no encontrado. Disponibles: {list(presets.keys())}")
    
    config_updates = presets[preset]
    for section, updates in config_updates.items():
        update_config(section, updates)

if __name__ == "__main__":
    """
    Script de prueba para el archivo de configuración.
    """
    print("=" * 60)
    print("CONFIGURACIÓN DNN - LucIA Project")
    print("=" * 60)
    
    # Mostrar configuración actual
    print("\nArquitectura:")
    for key, value in ARCHITECTURE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nEntrenamiento:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Validar configuración
    errors = validate_config()
    if errors:
        print(f"\n⚠️  Errores en configuración: {errors}")
    else:
        print("\n✅ Configuración válida")
    
    # Crear directorios
    create_directories()
    print("\n✅ Directorios creados")
    
    print("\n" + "=" * 60)
