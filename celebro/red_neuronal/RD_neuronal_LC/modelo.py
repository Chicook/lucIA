"""
Módulo 1: Definición de la Arquitectura de Red Neuronal Profunda (DNN)
=====================================================================

Este módulo contiene la función para crear un modelo de red neuronal profunda
completamente conectada usando TensorFlow/Keras con la siguiente arquitectura:
- Capa de entrada: 8 características
- 3 capas ocultas densas con tamaños decrecientes (32, 16, 8) y activación ReLU
- Capa de salida: 4 neuronas con activación Softmax para clasificación multiclase

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import numpy as np
import logging
from typing import Tuple, Optional
import warnings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings de TensorFlow
warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')


class DNNArchitecture:
    """
    Clase para definir y gestionar la arquitectura de la Red Neuronal Profunda.
    
    Esta clase encapsula toda la configuración necesaria para crear una DNN
    con arquitectura específica y proporciona métodos para su construcción
    y configuración.
    """
    
    def __init__(self, input_shape: int = 8, num_classes: int = 4):
        """
        Inicializa la configuración de la arquitectura DNN.
        
        Args:
            input_shape (int): Número de características de entrada (default: 8)
            num_classes (int): Número de clases de salida (default: 4)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layers_config = [32, 16, 8]  # Tamaños decrecientes
        self.activation_hidden = 'relu'
        self.activation_output = 'softmax'
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
        
        logger.info(f"Arquitectura DNN inicializada: {input_shape} -> {self.hidden_layers_config} -> {num_classes}")
    
    def _create_sequential_model(self) -> Sequential:
        """
        Crea el modelo secuencial de Keras con la arquitectura definida.
        
        Returns:
            Sequential: Modelo de Keras compilado
        """
        try:
            model = Sequential()
            
            # Capa de entrada
            model.add(layers.Dense(
                self.hidden_layers_config[0], 
                activation=self.activation_hidden,
                input_shape=(self.input_shape,),
                name='input_layer'
            ))
            
            # Capas ocultas con tamaños decrecientes
            for i, neurons in enumerate(self.hidden_layers_config[1:], 1):
                model.add(layers.Dense(
                    neurons,
                    activation=self.activation_hidden,
                    name=f'hidden_layer_{i}'
                ))
            
            # Capa de salida
            model.add(layers.Dense(
                self.num_classes,
                activation=self.activation_output,
                name='output_layer'
            ))
            
            logger.info("Modelo secuencial creado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error al crear el modelo secuencial: {str(e)}")
            raise
    
    def _compile_model(self, model: Sequential) -> Sequential:
        """
        Compila el modelo con optimizador, función de pérdida y métricas.
        
        Args:
            model (Sequential): Modelo de Keras a compilar
            
        Returns:
            Sequential: Modelo compilado
        """
        try:
            # Configurar optimizador Adam con parámetros personalizados
            optimizer = Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )
            
            # Configurar función de pérdida
            loss_function = CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=0.0
            )
            
            # Configurar métricas de evaluación
            metrics = [
                Accuracy(name='accuracy'),
                Precision(name='precision', average='macro'),
                Recall(name='recall', average='macro')
            ]
            
            # Compilar el modelo
            model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=metrics
            )
            
            logger.info("Modelo compilado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error al compilar el modelo: {str(e)}")
            raise
    
    def create_model(self) -> Sequential:
        """
        Crea y compila el modelo completo de la DNN.
        
        Returns:
            Sequential: Modelo de Keras completamente configurado
        """
        try:
            logger.info("Iniciando creación del modelo DNN...")
            
            # Crear modelo secuencial
            model = self._create_sequential_model()
            
            # Compilar modelo
            model = self._compile_model(model)
            
            logger.info("Modelo DNN creado y compilado exitosamente")
            return model
            
        except Exception as e:
            logger.error(f"Error en la creación del modelo: {str(e)}")
            raise


def crear_modelo(input_shape: int = 8, num_classes: int = 4) -> Sequential:
    """
    Función principal para crear un modelo de red neuronal profunda.
    
    Esta función crea una DNN completamente conectada con:
    - Capa de entrada: input_shape características
    - 3 capas ocultas: 32, 16, 8 neuronas respectivamente (con activación ReLU)
    - Capa de salida: num_classes neuronas (con activación Softmax)
    
    Args:
        input_shape (int): Número de características de entrada (default: 8)
        num_classes (int): Número de clases de salida (default: 4)
    
    Returns:
        Sequential: Modelo de Keras compilado listo para entrenamiento
        
    Raises:
        ValueError: Si los parámetros de entrada son inválidos
        RuntimeError: Si hay error en la creación del modelo
        
    Example:
        >>> model = crear_modelo(input_shape=8, num_classes=4)
        >>> print(model.summary())
    """
    try:
        # Validar parámetros de entrada
        if input_shape <= 0:
            raise ValueError("input_shape debe ser un número positivo")
        if num_classes <= 0:
            raise ValueError("num_classes debe ser un número positivo")
        
        logger.info(f"Creando modelo DNN: {input_shape} entradas -> {num_classes} salidas")
        
        # Crear instancia de la arquitectura
        dnn_arch = DNNArchitecture(input_shape, num_classes)
        
        # Crear y retornar el modelo
        model = dnn_arch.create_model()
        
        logger.info("Modelo DNN creado exitosamente")
        return model
        
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al crear modelo: {str(e)}")
        raise RuntimeError(f"Error al crear modelo DNN: {str(e)}")


def get_model_info(model: Sequential) -> dict:
    """
    Obtiene información detallada sobre el modelo creado.
    
    Args:
        model (Sequential): Modelo de Keras
        
    Returns:
        dict: Diccionario con información del modelo
    """
    try:
        info = {
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            'layers_count': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
        
        logger.info(f"Información del modelo obtenida: {info['total_params']} parámetros totales")
        return info
        
    except Exception as e:
        logger.error(f"Error al obtener información del modelo: {str(e)}")
        return {}


def validate_model_architecture(model: Sequential, expected_input: int = 8, expected_output: int = 4) -> bool:
    """
    Valida que la arquitectura del modelo coincida con las especificaciones.
    
    Args:
        model (Sequential): Modelo a validar
        expected_input (int): Número esperado de entradas (default: 8)
        expected_output (int): Número esperado de salidas (default: 4)
        
    Returns:
        bool: True si la arquitectura es válida, False en caso contrario
    """
    try:
        # Validar forma de entrada
        if model.input_shape[1] != expected_input:
            logger.warning(f"Forma de entrada incorrecta: esperado {expected_input}, obtenido {model.input_shape[1]}")
            return False
        
        # Validar forma de salida
        if model.output_shape[1] != expected_output:
            logger.warning(f"Forma de salida incorrecta: esperado {expected_output}, obtenido {model.output_shape[1]}")
            return False
        
        # Validar número de capas (1 entrada + 3 ocultas + 1 salida = 5)
        if len(model.layers) != 5:
            logger.warning(f"Número de capas incorrecto: esperado 5, obtenido {len(model.layers)}")
            return False
        
        logger.info("Arquitectura del modelo validada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error al validar arquitectura: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Script de prueba para el módulo modelo.py
    """
    try:
        print("=" * 60)
        print("PRUEBA DEL MÓDULO MODELO.PY")
        print("=" * 60)
        
        # Crear modelo de prueba
        model = crear_modelo(input_shape=8, num_classes=4)
        
        # Mostrar resumen
        print("\nRESUMEN DEL MODELO:")
        print("-" * 30)
        model.summary()
        
        # Obtener información detallada
        info = get_model_info(model)
        print(f"\nINFORMACIÓN DETALLADA:")
        print(f"Total de parámetros: {info.get('total_params', 'N/A')}")
        print(f"Parámetros entrenables: {info.get('trainable_params', 'N/A')}")
        print(f"Número de capas: {info.get('layers_count', 'N/A')}")
        
        # Validar arquitectura
        is_valid = validate_model_architecture(model)
        print(f"\nArquitectura válida: {'SÍ' if is_valid else 'NO'}")
        
        print("\n" + "=" * 60)
        print("PRUEBA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error en la prueba: {str(e)}")
        logger.error(f"Error en prueba del módulo: {str(e)}")
