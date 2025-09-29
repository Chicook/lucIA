#!/usr/bin/env python3
"""
Integración de TensorFlow con @celebro - LucIA
Versión: 0.6.0
Sistema avanzado de aprendizaje profundo integrado con @celebro
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pickle

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger('Celebro_TensorFlow')

class ModelType(Enum):
    """Tipos de modelos de TensorFlow"""
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RESPONSE_GENERATION = "response_generation"
    CONTEXT_UNDERSTANDING = "context_understanding"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    SECURITY_ANALYSIS = "security_analysis"

class TrainingStatus(Enum):
    """Estados de entrenamiento"""
    IDLE = "idle"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    max_sequence_length: int = 1000
    vocabulary_size: int = 10000
    embedding_dim: int = 128

@dataclass
class ModelMetrics:
    """Métricas del modelo"""
    accuracy: float = 0.0
    loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time: float = 0.0
    validation_accuracy: float = 0.0
    validation_loss: float = 0.0

class TensorFlowCelebroIntegration:
    """
    Integración de TensorFlow con el sistema @celebro
    """
    
    def __init__(self, models_dir: str = "celebro/models"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Tokenizer] = {}
        self.training_config = TrainingConfig()
        self.training_status = TrainingStatus.IDLE
        
        # Crear directorio de modelos
        os.makedirs(models_dir, exist_ok=True)
        
        # Configurar TensorFlow
        self._configure_tensorflow()
        
        logger.info("Integración de TensorFlow con @celebro inicializada")
    
    def _configure_tensorflow(self):
        """Configura TensorFlow para optimizar rendimiento"""
        try:
            # Configurar GPU si está disponible
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU configurada: {len(gpus)} dispositivos disponibles")
                except RuntimeError as e:
                    logger.warning(f"Error configurando GPU: {e}")
            else:
                logger.info("No se detectó GPU, usando CPU")
            
            # Configurar threading
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
        except Exception as e:
            logger.error(f"Error configurando TensorFlow: {e}")
    
    def create_text_classification_model(self, model_name: str, 
                                       num_classes: int,
                                       vocabulary_size: int = None,
                                       max_sequence_length: int = None) -> str:
        """
        Crea un modelo de clasificación de texto
        
        Args:
            model_name: Nombre del modelo
            num_classes: Número de clases
            vocabulary_size: Tamaño del vocabulario
            max_sequence_length: Longitud máxima de secuencia
            
        Returns:
            ID del modelo creado
        """
        try:
            vocab_size = vocabulary_size or self.training_config.vocabulary_size
            max_len = max_sequence_length or self.training_config.max_sequence_length
            
            # Crear modelo
            model = models.Sequential([
                layers.Embedding(vocab_size, self.training_config.embedding_dim, 
                               input_length=max_len),
                layers.Conv1D(128, 5, activation='relu'),
                layers.GlobalMaxPooling1D(),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compilar modelo
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.training_config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Guardar modelo
            model_id = self._save_model(model, model_name, ModelType.TEXT_CLASSIFICATION)
            
            logger.info(f"Modelo de clasificación de texto creado: {model_name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creando modelo de clasificación: {e}")
            raise
    
    def create_sentiment_analysis_model(self, model_name: str) -> str:
        """
        Crea un modelo de análisis de sentimientos
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            ID del modelo creado
        """
        try:
            # Modelo especializado para análisis de sentimientos
            model = models.Sequential([
                layers.Embedding(self.training_config.vocabulary_size, 
                               self.training_config.embedding_dim,
                               input_length=self.training_config.max_sequence_length),
                layers.LSTM(64, return_sequences=True, dropout=0.2),
                layers.LSTM(32, dropout=0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(3, activation='softmax')  # Positivo, Negativo, Neutral
            ])
            
            # Compilar modelo
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.training_config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Guardar modelo
            model_id = self._save_model(model, model_name, ModelType.SENTIMENT_ANALYSIS)
            
            logger.info(f"Modelo de análisis de sentimientos creado: {model_name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creando modelo de análisis de sentimientos: {e}")
            raise
    
    def create_response_generation_model(self, model_name: str) -> str:
        """
        Crea un modelo de generación de respuestas
        
        Args:
            model_name: Nombre del modelo
            
        Returns:
            ID del modelo creado
        """
        try:
            # Modelo de generación de respuestas usando LSTM
            model = models.Sequential([
                layers.Embedding(self.training_config.vocabulary_size, 
                               self.training_config.embedding_dim,
                               input_length=self.training_config.max_sequence_length),
                layers.LSTM(256, return_sequences=True, dropout=0.2),
                layers.LSTM(128, return_sequences=True, dropout=0.2),
                layers.LSTM(64, dropout=0.2),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.training_config.vocabulary_size, activation='softmax')
            ])
            
            # Compilar modelo
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.training_config.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Guardar modelo
            model_id = self._save_model(model, model_name, ModelType.RESPONSE_GENERATION)
            
            logger.info(f"Modelo de generación de respuestas creado: {model_name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creando modelo de generación: {e}")
            raise
    
    def create_security_analysis_model(self, model_name: str, 
                                     security_categories: List[str]) -> str:
        """
        Crea un modelo especializado en análisis de seguridad
        
        Args:
            model_name: Nombre del modelo
            security_categories: Categorías de seguridad
            
        Returns:
            ID del modelo creado
        """
        try:
            num_categories = len(security_categories)
            
            # Modelo especializado para análisis de seguridad
            model = models.Sequential([
                layers.Embedding(self.training_config.vocabulary_size, 
                               self.training_config.embedding_dim,
                               input_length=self.training_config.max_sequence_length),
                layers.Conv1D(128, 3, activation='relu'),
                layers.Conv1D(64, 3, activation='relu'),
                layers.GlobalMaxPooling1D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_categories, activation='softmax')
            ])
            
            # Compilar modelo
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.training_config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Guardar modelo
            model_id = self._save_model(model, model_name, ModelType.SECURITY_ANALYSIS)
            
            # Guardar categorías
            self._save_model_metadata(model_id, {
                'security_categories': security_categories,
                'model_type': 'security_analysis'
            })
            
            logger.info(f"Modelo de análisis de seguridad creado: {model_name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creando modelo de seguridad: {e}")
            raise
    
    def train_model(self, model_id: str, training_data: List[str], 
                   labels: List[Union[int, str]], 
                   validation_data: Tuple[List[str], List[Union[int, str]]] = None) -> ModelMetrics:
        """
        Entrena un modelo con datos de texto
        
        Args:
            model_id: ID del modelo
            training_data: Datos de entrenamiento
            labels: Etiquetas de entrenamiento
            validation_data: Datos de validación (opcional)
            
        Returns:
            Métricas del entrenamiento
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo no encontrado: {model_id}")
            
            self.training_status = TrainingStatus.TRAINING
            start_time = datetime.now()
            
            model = self.models[model_id]
            model_type = self._get_model_type(model_id)
            
            # Preparar datos
            if model_type in [ModelType.TEXT_CLASSIFICATION, ModelType.SECURITY_ANALYSIS]:
                X_train, y_train = self._prepare_classification_data(training_data, labels)
                
                if validation_data:
                    X_val, y_val = self._prepare_classification_data(
                        validation_data[0], validation_data[1]
                    )
                    validation_data = (X_val, y_val)
                else:
                    validation_split = self.training_config.validation_split
            else:
                X_train, y_train = self._prepare_sequence_data(training_data, labels)
                validation_split = self.training_config.validation_split
                validation_data = None
            
            # Guardar tokenizer para uso posterior
            if model_type in [ModelType.TEXT_CLASSIFICATION, ModelType.SECURITY_ANALYSIS, ModelType.SENTIMENT_ANALYSIS]:
                tokenizer = Tokenizer(num_words=self.training_config.vocabulary_size)
                tokenizer.fit_on_texts(training_data)
                self.tokenizers[model_id] = tokenizer
            
            # Configurar callbacks
            callbacks_list = self._create_callbacks(model_id)
            
            # Entrenar modelo
            history = model.fit(
                X_train, y_train,
                epochs=self.training_config.epochs,
                batch_size=self.training_config.batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Calcular métricas
            training_time = (datetime.now() - start_time).total_seconds()
            metrics = self._calculate_metrics(model, X_train, y_train, history, training_time)
            
            # Guardar modelo entrenado
            self._save_trained_model(model_id, model)
            
            self.training_status = TrainingStatus.COMPLETED
            
            logger.info(f"Modelo {model_id} entrenado exitosamente")
            return metrics
            
        except Exception as e:
            self.training_status = TrainingStatus.ERROR
            logger.error(f"Error entrenando modelo: {e}")
            raise
    
    def predict(self, model_id: str, text: str) -> Dict[str, Any]:
        """
        Realiza predicción con un modelo
        
        Args:
            model_id: ID del modelo
            text: Texto a analizar
            
        Returns:
            Resultado de la predicción
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo no encontrado: {model_id}")
            
            model = self.models[model_id]
            model_type = self._get_model_type(model_id)
            
            # Preparar texto
            if model_type == ModelType.RESPONSE_GENERATION:
                # Para generación, usar el tokenizer correspondiente
                tokenizer = self.tokenizers.get(model_id)
                if not tokenizer:
                    raise ValueError("Tokenizer no encontrado para el modelo")
                
                sequence = tokenizer.texts_to_sequences([text])
                padded_sequence = pad_sequences(sequence, maxlen=self.training_config.max_sequence_length)
            else:
                # Para clasificación
                tokenizer = self.tokenizers.get(model_id)
                if not tokenizer:
                    # Crear tokenizer básico
                    tokenizer = Tokenizer(num_words=self.training_config.vocabulary_size)
                    tokenizer.fit_on_texts([text])
                    self.tokenizers[model_id] = tokenizer
                
                sequence = tokenizer.texts_to_sequences([text])
                padded_sequence = pad_sequences(sequence, maxlen=self.training_config.max_sequence_length)
            
            # Realizar predicción
            prediction = model.predict(padded_sequence, verbose=0)
            
            # Procesar resultado según el tipo de modelo
            if model_type == ModelType.SENTIMENT_ANALYSIS:
                sentiment_labels = ['Negativo', 'Neutral', 'Positivo']
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                return {
                    'sentiment': sentiment_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': {
                        label: float(prob) for label, prob in zip(sentiment_labels, prediction[0])
                    }
                }
            
            elif model_type == ModelType.SECURITY_ANALYSIS:
                metadata = self._load_model_metadata(model_id)
                categories = metadata.get('security_categories', [])
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                return {
                    'security_category': categories[predicted_class] if predicted_class < len(categories) else 'Unknown',
                    'confidence': confidence,
                    'probabilities': {
                        cat: float(prob) for cat, prob in zip(categories, prediction[0])
                    }
                }
            
            else:
                # Clasificación general
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                return {
                    'predicted_class': int(predicted_class),
                    'confidence': confidence,
                    'probabilities': [float(p) for p in prediction[0]]
                }
                
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def _prepare_classification_data(self, texts: List[str], labels: List[Union[int, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para clasificación"""
        # Crear o usar tokenizer existente
        tokenizer = Tokenizer(num_words=self.training_config.vocabulary_size)
        tokenizer.fit_on_texts(texts)
        
        # Convertir textos a secuencias
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.training_config.max_sequence_length)
        
        # Convertir etiquetas
        if isinstance(labels[0], str):
            # Convertir strings a números
            unique_labels = list(set(labels))
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_to_int[label] for label in labels])
            y = to_categorical(y, num_classes=len(unique_labels))
        else:
            y = to_categorical(labels, num_classes=len(set(labels)))
        
        return X, y
    
    def _prepare_sequence_data(self, texts: List[str], labels: List[Union[int, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para generación de secuencias"""
        # Similar a clasificación pero para generación
        tokenizer = Tokenizer(num_words=self.training_config.vocabulary_size)
        tokenizer.fit_on_texts(texts)
        
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.training_config.max_sequence_length)
        
        # Para generación, las etiquetas son las secuencias desplazadas
        y = sequences
        y = pad_sequences(y, maxlen=self.training_config.max_sequence_length)
        
        return X, y
    
    def _create_callbacks(self, model_id: str) -> List[callbacks.Callback]:
        """Crea callbacks para el entrenamiento"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config.early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.training_config.reduce_lr_patience,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.models_dir, f"{model_id}_best.weights.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]
        
        return callbacks_list
    
    def _calculate_metrics(self, model, X_test, y_test, history, training_time) -> ModelMetrics:
        """Calcula métricas del modelo"""
        try:
            # Evaluar modelo
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            # Obtener métricas de validación
            val_accuracy = history.history.get('val_accuracy', [0])[-1]
            val_loss = history.history.get('val_loss', [0])[-1]
            
            # Calcular F1 score (simplificado)
            predictions = model.predict(X_test, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # F1 score macro
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(true_classes, predicted_classes, average='macro')
            precision = precision_score(true_classes, predicted_classes, average='macro')
            recall = recall_score(true_classes, predicted_classes, average='macro')
            
            return ModelMetrics(
                accuracy=float(accuracy),
                loss=float(loss),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                training_time=training_time,
                validation_accuracy=float(val_accuracy),
                validation_loss=float(val_loss)
            )
            
        except Exception as e:
            logger.error(f"Error calculando métricas: {e}")
            return ModelMetrics()
    
    def _save_model(self, model, model_name: str, model_type: ModelType) -> str:
        """Guarda un modelo"""
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Guardar modelo
        model_path = os.path.join(self.models_dir, f"{model_id}.h5")
        model.save(model_path)
        
        # Guardar metadatos
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type.value,
            'created_at': datetime.now().isoformat(),
            'file_path': model_path
        }
        
        self._save_model_metadata(model_id, metadata)
        self.models[model_id] = model
        
        return model_id
    
    def _save_trained_model(self, model_id: str, model):
        """Guarda un modelo entrenado"""
        model_path = os.path.join(self.models_dir, f"{model_id}_trained.h5")
        model.save(model_path)
        
        # Actualizar metadatos
        metadata = self._load_model_metadata(model_id)
        metadata['trained_at'] = datetime.now().isoformat()
        metadata['trained_model_path'] = model_path
        self._save_model_metadata(model_id, metadata)
    
    def _save_model_metadata(self, model_id: str, metadata: Dict[str, Any]):
        """Guarda metadatos del modelo"""
        metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Carga metadatos del modelo"""
        metadata_path = os.path.join(self.models_dir, f"{model_id}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_model_type(self, model_id: str) -> ModelType:
        """Obtiene el tipo de modelo"""
        metadata = self._load_model_metadata(model_id)
        model_type_str = metadata.get('model_type', 'text_classification')
        return ModelType(model_type_str)
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Obtiene información de un modelo"""
        if model_id not in self.models:
            return {'error': 'Modelo no encontrado'}
        
        metadata = self._load_model_metadata(model_id)
        model = self.models[model_id]
        
        return {
            'model_id': model_id,
            'model_name': metadata.get('model_name', 'Unknown'),
            'model_type': metadata.get('model_type', 'Unknown'),
            'created_at': metadata.get('created_at', 'Unknown'),
            'trained_at': metadata.get('trained_at', 'Not trained'),
            'total_params': model.count_params(),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos los modelos disponibles"""
        models_info = []
        for model_id in self.models.keys():
            models_info.append(self.get_model_info(model_id))
        return models_info
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo"""
        try:
            if model_id in self.models:
                del self.models[model_id]
            
            # Eliminar archivos
            model_files = [
                f"{model_id}.h5",
                f"{model_id}_trained.h5",
                f"{model_id}_best.h5",
                f"{model_id}_metadata.json"
            ]
            
            for file_name in model_files:
                file_path = os.path.join(self.models_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info(f"Modelo {model_id} eliminado")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando modelo: {e}")
            return False

# Instancia global de la integración de TensorFlow
tensorflow_celebro = TensorFlowCelebroIntegration()
