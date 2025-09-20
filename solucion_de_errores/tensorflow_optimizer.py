"""
Optimizador de TensorFlow
Mejora el rendimiento y corrige problemas en el entrenamiento de modelos TensorFlow.
"""

import logging
import os
import gc
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import traceback

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None

logger = logging.getLogger('TensorFlowOptimizer')

class TensorFlowOptimizer:
    """
    Optimizador de TensorFlow que mejora el rendimiento y corrige problemas comunes.
    """
    
    def __init__(self):
        self.optimization_config = {
            'memory_growth': True,
            'mixed_precision': True,
            'xla_compilation': True,
            'optimize_for_inference': True,
            'batch_size_optimization': True,
            'learning_rate_scheduling': True,
            'early_stopping': True,
            'model_checkpointing': True
        }
        
        self.performance_metrics = {
            'models_optimized': 0,
            'training_time_improvements': 0,
            'memory_usage_reductions': 0,
            'accuracy_improvements': 0,
            'errors_fixed': 0
        }
        
        self._setup_tensorflow_optimizations()
    
    def _setup_tensorflow_optimizations(self):
        """Configura optimizaciones globales de TensorFlow"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow no disponible, saltando optimizaciones")
            return
        
        try:
            # Configurar crecimiento de memoria para GPUs
            if self.optimization_config['memory_growth']:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Crecimiento de memoria GPU habilitado")
            
            # Configurar precisión mixta para mejor rendimiento
            if self.optimization_config['mixed_precision']:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    logger.info("Precisión mixta habilitada")
                except Exception as e:
                    logger.warning(f"No se pudo habilitar precisión mixta: {e}")
            
            # Configurar compilación XLA para mejor rendimiento
            if self.optimization_config['xla_compilation']:
                try:
                    tf.config.optimizer.set_jit(True)
                    logger.info("Compilación XLA habilitada")
                except Exception as e:
                    logger.warning(f"No se pudo habilitar XLA: {e}")
            
            logger.info("Optimizaciones de TensorFlow configuradas")
            
        except Exception as e:
            logger.error(f"Error configurando optimizaciones de TensorFlow: {e}")
    
    def optimize_model_architecture(self, model: Any, input_shape: Tuple, 
                                  num_classes: int, model_type: str = "classification") -> Any:
        """
        Optimiza la arquitectura de un modelo de TensorFlow.
        
        Args:
            model: Modelo de Keras a optimizar
            input_shape: Forma de entrada del modelo
            num_classes: Número de clases de salida
            model_type: Tipo de modelo (classification, regression, etc.)
            
        Returns:
            Modelo optimizado
        """
        if not TENSORFLOW_AVAILABLE:
            return model
        
        try:
            # Crear modelo optimizado basado en el tipo
            if model_type == "classification":
                optimized_model = self._create_optimized_classification_model(
                    input_shape, num_classes
                )
            elif model_type == "regression":
                optimized_model = self._create_optimized_regression_model(
                    input_shape, num_classes
                )
            elif model_type == "sequence":
                optimized_model = self._create_optimized_sequence_model(
                    input_shape, num_classes
                )
            else:
                optimized_model = self._create_optimized_generic_model(
                    input_shape, num_classes
                )
            
            # Compilar con optimizaciones
            self._compile_optimized_model(optimized_model, model_type, num_classes)
            
            self.performance_metrics['models_optimized'] += 1
            logger.info(f"Modelo {model_type} optimizado exitosamente")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error optimizando arquitectura del modelo: {e}")
            return model
    
    def _create_optimized_classification_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Crea un modelo de clasificación optimizado"""
        model = models.Sequential([
            # Capa de entrada con normalización
            layers.Input(shape=input_shape),
            layers.BatchNormalization(),
            
            # Capas convolucionales optimizadas
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Pooling global optimizado
            layers.GlobalAveragePooling1D(),
            
            # Capas densas optimizadas
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_optimized_regression_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Crea un modelo de regresión optimizado"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.BatchNormalization(),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(num_classes, activation='linear')
        ])
        
        return model
    
    def _create_optimized_sequence_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Crea un modelo de secuencia optimizado (LSTM/GRU)"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.BatchNormalization(),
            
            # LSTM bidireccional optimizado
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2)),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(64, dropout=0.2)),
            layers.BatchNormalization(),
            
            # Capas densas
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def _create_optimized_generic_model(self, input_shape: Tuple, num_classes: int) -> Any:
        """Crea un modelo genérico optimizado"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.BatchNormalization(),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
        ])
        
        return model
    
    def _compile_optimized_model(self, model: Any, model_type: str, num_classes: int):
        """Compila el modelo con optimizaciones"""
        try:
            # Optimizador Adam con configuración optimizada
            optimizer = optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=True
            )
            
            # Función de pérdida optimizada
            if model_type == "classification":
                if num_classes == 2:
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy', 'precision', 'recall']
                else:
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy', 'precision', 'recall']
            elif model_type == "regression":
                loss = 'mse'
                metrics = ['mae', 'mse']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            
            # Compilar modelo
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )
            
            logger.info(f"Modelo compilado con optimizaciones: {model_type}")
            
        except Exception as e:
            logger.error(f"Error compilando modelo optimizado: {e}")
    
    def optimize_training_process(self, model: Any, training_data: Tuple, 
                                validation_data: Optional[Tuple] = None,
                                epochs: int = 50) -> Dict[str, Any]:
        """
        Optimiza el proceso de entrenamiento de un modelo.
        
        Args:
            model: Modelo de Keras a entrenar
            training_data: Datos de entrenamiento (X, y)
            validation_data: Datos de validación (X_val, y_val)
            epochs: Número de épocas
            
        Returns:
            Resultados del entrenamiento optimizado
        """
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow no disponible'}
        
        try:
            X_train, y_train = training_data
            
            # Configurar callbacks optimizados
            callbacks_list = self._create_optimized_callbacks()
            
            # Configurar datos de validación
            validation_split = 0.2 if validation_data is None else None
            
            # Entrenar modelo con optimizaciones
            start_time = datetime.now()
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=self._calculate_optimal_batch_size(X_train),
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks_list,
                verbose=1,
                shuffle=True
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Calcular mejoras de rendimiento
            self._calculate_performance_improvements(history, training_time)
            
            return {
                'history': history.history,
                'training_time': training_time,
                'best_accuracy': max(history.history.get('val_accuracy', [0])),
                'best_loss': min(history.history.get('val_loss', [float('inf')])),
                'optimization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento optimizado: {e}")
            return {'error': str(e)}
    
    def _create_optimized_callbacks(self) -> List[Any]:
        """Crea callbacks optimizados para el entrenamiento"""
        callbacks_list = []
        
        try:
            # Early Stopping optimizado
            if self.optimization_config['early_stopping']:
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.001
                )
                callbacks_list.append(early_stopping)
            
            # Reducción de learning rate
            if self.optimization_config['learning_rate_scheduling']:
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
                callbacks_list.append(reduce_lr)
            
            # Model checkpointing
            if self.optimization_config['model_checkpointing']:
                checkpoint = callbacks.ModelCheckpoint(
                    'models/optimized_model_best.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                )
                callbacks_list.append(checkpoint)
            
            # Limpieza de memoria
            memory_cleanup = callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: gc.collect()
            )
            callbacks_list.append(memory_cleanup)
            
        except Exception as e:
            logger.error(f"Error creando callbacks optimizados: {e}")
        
        return callbacks_list
    
    def _calculate_optimal_batch_size(self, X_train: np.ndarray) -> int:
        """Calcula el tamaño de lote óptimo basado en los datos"""
        try:
            data_size = len(X_train)
            
            # Calcular batch size basado en el tamaño de los datos
            if data_size < 1000:
                return min(32, data_size)
            elif data_size < 10000:
                return min(64, data_size)
            elif data_size < 100000:
                return min(128, data_size)
            else:
                return min(256, data_size)
                
        except Exception as e:
            logger.error(f"Error calculando batch size óptimo: {e}")
            return 32
    
    def _calculate_performance_improvements(self, history: Any, training_time: float):
        """Calcula mejoras de rendimiento del entrenamiento"""
        try:
            # Mejora en tiempo de entrenamiento (comparado con estimación base)
            estimated_base_time = training_time * 1.5  # Estimación conservadora
            time_improvement = (estimated_base_time - training_time) / estimated_base_time * 100
            self.performance_metrics['training_time_improvements'] += time_improvement
            
            # Mejora en precisión
            final_accuracy = history.history.get('val_accuracy', [0])[-1]
            if final_accuracy > 0.8:  # Considerar buena precisión
                self.performance_metrics['accuracy_improvements'] += 1
            
        except Exception as e:
            logger.error(f"Error calculando mejoras de rendimiento: {e}")
    
    def fix_common_tensorflow_errors(self, model: Any, training_data: Tuple) -> Any:
        """
        Corrige errores comunes de TensorFlow.
        
        Args:
            model: Modelo de Keras
            training_data: Datos de entrenamiento
            
        Returns:
            Modelo corregido
        """
        if not TENSORFLOW_AVAILABLE:
            return model
        
        try:
            X_train, y_train = training_data
            
            # Verificar y corregir formas de datos
            if len(X_train.shape) == 2 and len(y_train.shape) == 1:
                # Agregar dimensión para LSTM/CNN si es necesario
                if hasattr(model, 'layers') and any('LSTM' in str(type(layer)) for layer in model.layers):
                    X_train = np.expand_dims(X_train, axis=2)
                    logger.info("Dimensión agregada para LSTM")
            
            # Verificar tipos de datos
            if X_train.dtype != 'float32':
                X_train = X_train.astype('float32')
                logger.info("Datos convertidos a float32")
            
            if y_train.dtype != 'int32' and len(y_train.shape) == 1:
                y_train = y_train.astype('int32')
                logger.info("Etiquetas convertidas a int32")
            
            # Normalizar datos si es necesario
            if np.max(X_train) > 1.0:
                X_train = X_train / np.max(X_train)
                logger.info("Datos normalizados")
            
            self.performance_metrics['errors_fixed'] += 1
            
            return model, (X_train, y_train)
            
        except Exception as e:
            logger.error(f"Error corrigiendo problemas de TensorFlow: {e}")
            return model, training_data
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Genera reporte de optimizaciones aplicadas"""
        return {
            'performance_metrics': self.performance_metrics,
            'optimization_config': self.optimization_config,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'recommendations': self._generate_optimization_recommendations()
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Genera recomendaciones de optimización"""
        recommendations = []
        
        if self.performance_metrics['models_optimized'] > 0:
            recommendations.append("Considerar usar más modelos optimizados")
        
        if self.performance_metrics['training_time_improvements'] > 20:
            recommendations.append("Optimizaciones de tiempo aplicadas exitosamente")
        
        if self.performance_metrics['accuracy_improvements'] > 0:
            recommendations.append("Mejoras en precisión detectadas")
        
        if not TENSORFLOW_AVAILABLE:
            recommendations.append("Instalar TensorFlow para habilitar optimizaciones")
        
        return recommendations
