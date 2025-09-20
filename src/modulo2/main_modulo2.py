"""
Módulo 2: Sistema de Aprendizaje y Adaptación
Versión: 0.6.0
Funcionalidad: Motor de aprendizaje automático, adaptación y mejora continua
"""

import numpy as np
import json
import os
import pickle
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

logger = logging.getLogger('LucIA_Learning')

class LearningEngine:
    """
    Motor de aprendizaje automático para LucIA.
    Gestiona modelos de ML, aprendizaje supervisado y no supervisado.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.models_dir = "models/neural"
        self.training_data_dir = "data/learning"
        self.learning_rate = 0.01
        self.learning_cycles = 0
        self.models = {}
        self.scalers = {}
        self.training_history = []
        self.performance_metrics = {}
        
        # Configuración de modelos
        self.model_configs = {
            'decision_tree': {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42}
            },
            'neural_network': {
                'class': MLPRegressor,
                'params': {'hidden_layer_sizes': (100, 50), 'max_iter': 1000, 'random_state': 42}
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'random_state': 42}
            }
        }
        
        # Inicializar directorios
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Cargar modelos existentes
        self._load_existing_models()
        
        logger.info("Motor de aprendizaje inicializado")
    
    def _load_existing_models(self):
        """Carga modelos previamente entrenados desde disco"""
        try:
            for model_name in self.model_configs.keys():
                model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
                
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Modelo {model_name} cargado desde disco")
                
                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)
                    logger.info(f"Scaler {model_name} cargado desde disco")
                    
        except Exception as e:
            logger.error(f"Error cargando modelos existentes: {e}")
    
    async def create_model(self, model_name: str, model_type: str = 'decision_tree') -> bool:
        """
        Crea un nuevo modelo de aprendizaje
        
        Args:
            model_name: Nombre único del modelo
            model_type: Tipo de modelo a crear
        
        Returns:
            True si se creó exitosamente
        """
        try:
            if model_type not in self.model_configs:
                logger.error(f"Tipo de modelo no soportado: {model_type}")
                return False
            
            config = self.model_configs[model_type]
            model = config['class'](**config['params'])
            
            self.models[model_name] = model
            self.scalers[model_name] = StandardScaler()
            
            logger.info(f"Modelo {model_name} de tipo {model_type} creado")
            return True
            
        except Exception as e:
            logger.error(f"Error creando modelo {model_name}: {e}")
            return False
    
    async def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                         validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena un modelo con datos proporcionados
        
        Args:
            model_name: Nombre del modelo a entrenar
            X: Características de entrada
            y: Valores objetivo
            validation_split: Proporción de datos para validación
        
        Returns:
            Métricas de entrenamiento
        """
        try:
            if model_name not in self.models:
                logger.error(f"Modelo {model_name} no encontrado")
                return {}
            
            # Dividir datos en entrenamiento y validación
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Escalar características
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_val_scaled = self.scalers[model_name].transform(X_val)
            
            # Entrenar modelo
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            # Calcular métricas
            if len(np.unique(y)) > 10:  # Regresión
                train_mse = mean_squared_error(y_train, train_pred)
                val_mse = mean_squared_error(y_val, val_pred)
                metrics = {
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'train_rmse': np.sqrt(train_mse),
                    'val_rmse': np.sqrt(val_mse)
                }
            else:  # Clasificación
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                metrics = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc
                }
            
            # Guardar modelo entrenado
            await self._save_model(model_name)
            
            # Registrar entrenamiento
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'samples': len(X),
                'metrics': metrics,
                'learning_cycle': self.learning_cycles
            }
            self.training_history.append(training_record)
            
            self.learning_cycles += 1
            logger.info(f"Modelo {model_name} entrenado exitosamente")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_name}: {e}")
            return {}
    
    async def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando un modelo entrenado
        
        Args:
            model_name: Nombre del modelo
            X: Características de entrada
        
        Returns:
            Predicciones del modelo
        """
        try:
            if model_name not in self.models:
                logger.error(f"Modelo {model_name} no encontrado")
                return np.array([])
            
            # Escalar características
            X_scaled = self.scalers[model_name].transform(X)
            
            # Realizar predicción
            predictions = self.models[model_name].predict(X_scaled)
            
            logger.debug(f"Predicciones generadas para {len(X)} muestras")
            return predictions
            
        except Exception as e:
            logger.error(f"Error realizando predicción con {model_name}: {e}")
            return np.array([])
    
    async def update_model(self, model_name: str, X_new: np.ndarray, y_new: np.ndarray) -> bool:
        """
        Actualiza un modelo existente con nuevos datos (aprendizaje incremental)
        
        Args:
            model_name: Nombre del modelo
            X_new: Nuevas características
            y_new: Nuevos valores objetivo
        
        Returns:
            True si se actualizó exitosamente
        """
        try:
            if model_name not in self.models:
                logger.error(f"Modelo {model_name} no encontrado")
                return False
            
            # Escalar nuevos datos
            X_new_scaled = self.scalers[model_name].transform(X_new)
            
            # Actualizar modelo (si soporta partial_fit)
            model = self.models[model_name]
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X_new_scaled, y_new)
                logger.info(f"Modelo {model_name} actualizado incrementalmente")
            else:
                # Para modelos que no soportan aprendizaje incremental,
                # re-entrenar con datos combinados
                logger.warning(f"Modelo {model_name} no soporta aprendizaje incremental")
                return False
            
            # Guardar modelo actualizado
            await self._save_model(model_name)
            
            self.learning_cycles += 1
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando modelo {model_name}: {e}")
            return False
    
    async def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el rendimiento de un modelo
        
        Args:
            model_name: Nombre del modelo
            X_test: Características de prueba
            y_test: Valores objetivo de prueba
        
        Returns:
            Métricas de evaluación
        """
        try:
            if model_name not in self.models:
                logger.error(f"Modelo {model_name} no encontrado")
                return {}
            
            # Realizar predicciones
            predictions = await self.predict(model_name, X_test)
            
            if len(predictions) == 0:
                return {}
            
            # Calcular métricas
            if len(np.unique(y_test)) > 10:  # Regresión
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - predictions))
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': 1 - (mse / np.var(y_test))
                }
            else:  # Clasificación
                accuracy = accuracy_score(y_test, predictions)
                metrics = {
                    'accuracy': accuracy,
                    'error_rate': 1 - accuracy
                }
            
            # Guardar métricas
            self.performance_metrics[model_name] = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            logger.info(f"Modelo {model_name} evaluado: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_name}: {e}")
            return {}
    
    async def process_learning_cycle(self):
        """Procesa un ciclo de aprendizaje automático"""
        try:
            # Buscar datos de entrenamiento nuevos
            training_files = [f for f in os.listdir(self.training_data_dir) 
                            if f.endswith('.json')]
            
            for file_name in training_files:
                file_path = os.path.join(self.training_data_dir, file_name)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Procesar datos de entrenamiento
                    await self._process_training_data(data)
                    
                    # Mover archivo procesado
                    processed_dir = os.path.join(self.training_data_dir, 'processed')
                    os.makedirs(processed_dir, exist_ok=True)
                    os.rename(file_path, os.path.join(processed_dir, file_name))
                    
                except Exception as e:
                    logger.error(f"Error procesando archivo {file_name}: {e}")
            
            # Limpiar modelos antiguos si es necesario
            await self._cleanup_old_models()
            
            logger.debug("Ciclo de aprendizaje completado")
            
        except Exception as e:
            logger.error(f"Error en ciclo de aprendizaje: {e}")
    
    async def _process_training_data(self, data: Dict[str, Any]):
        """Procesa datos de entrenamiento específicos"""
        try:
            model_name = data.get('model_name')
            if not model_name:
                return
            
            # Crear modelo si no existe
            if model_name not in self.models:
                model_type = data.get('model_type', 'decision_tree')
                await self.create_model(model_name, model_type)
            
            # Extraer características y objetivos
            X = np.array(data.get('features', []))
            y = np.array(data.get('targets', []))
            
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"Datos vacíos para modelo {model_name}")
                return
            
            # Entrenar modelo
            metrics = await self.train_model(model_name, X, y)
            
            logger.info(f"Modelo {model_name} entrenado con {len(X)} muestras")
            
        except Exception as e:
            logger.error(f"Error procesando datos de entrenamiento: {e}")
    
    async def _save_model(self, model_name: str):
        """Guarda un modelo entrenado en disco"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            
            joblib.dump(self.models[model_name], model_path)
            joblib.dump(self.scalers[model_name], scaler_path)
            
            logger.debug(f"Modelo {model_name} guardado en disco")
            
        except Exception as e:
            logger.error(f"Error guardando modelo {model_name}: {e}")
    
    async def _cleanup_old_models(self):
        """Limpia modelos antiguos o con bajo rendimiento"""
        try:
            # Implementar lógica de limpieza basada en rendimiento
            # Por ahora, solo limpiar archivos temporales
            temp_files = [f for f in os.listdir(self.training_data_dir) 
                         if f.startswith('temp_')]
            
            for temp_file in temp_files:
                file_path = os.path.join(self.training_data_dir, temp_file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Archivo temporal eliminado: {temp_file}")
                    
        except Exception as e:
            logger.error(f"Error en limpieza de modelos: {e}")
    
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de aprendizaje"""
        return {
            'total_models': len(self.models),
            'learning_cycles': self.learning_cycles,
            'training_history_count': len(self.training_history),
            'performance_metrics': self.performance_metrics,
            'available_models': list(self.models.keys())
        }
    
    async def save_state(self):
        """Guarda el estado actual del sistema de aprendizaje"""
        try:
            # Guardar historial de entrenamiento
            history_path = os.path.join(self.training_data_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Guardar métricas de rendimiento
            metrics_path = os.path.join(self.training_data_dir, 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            logger.info("Estado del sistema de aprendizaje guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")

# Instancia global del motor de aprendizaje
learning_engine = LearningEngine()

async def initialize_module(core_engine):
    """Inicializa el módulo de aprendizaje"""
    global learning_engine
    learning_engine.core_engine = core_engine
    core_engine.learning_engine = learning_engine
    logger.info("Módulo de aprendizaje inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de aprendizaje"""
    if isinstance(input_data, dict) and 'train' in input_data:
        # Entrenar modelo
        model_name = input_data.get('model_name', 'default')
        features = np.array(input_data.get('features', []))
        targets = np.array(input_data.get('targets', []))
        
        if len(features) > 0 and len(targets) > 0:
            metrics = await learning_engine.train_model(model_name, features, targets)
            return {'training_completed': True, 'metrics': metrics}
    
    elif isinstance(input_data, dict) and 'predict' in input_data:
        # Realizar predicción
        model_name = input_data.get('model_name', 'default')
        features = np.array(input_data.get('features', []))
        
        if len(features) > 0 and model_name in learning_engine.models:
            predictions = await learning_engine.predict(model_name, features)
            return {'predictions': predictions.tolist()}
    
    return input_data

async def process_learning_cycle():
    """Procesa un ciclo de aprendizaje automático"""
    try:
        logger.info("Iniciando ciclo de aprendizaje automático")
        
        # Incrementar contador de ciclos
        learning_engine.learning_cycles += 1
        
        # Verificar si hay datos de entrenamiento disponibles
        if not learning_engine.training_history:
            logger.info("No hay datos de entrenamiento disponibles")
            return
        
        # Procesar aprendizaje incremental
        await _incremental_learning()
        
        # Actualizar métricas de rendimiento
        await _update_performance_metrics()
        
        logger.info(f"Ciclo de aprendizaje {learning_engine.learning_cycles} completado")
        
    except Exception as e:
        logger.error(f"Error en ciclo de aprendizaje: {e}")

async def _incremental_learning():
    """Realiza aprendizaje incremental en los modelos existentes"""
    try:
        # Obtener datos recientes
        recent_data = learning_engine.training_history[-100:] if len(learning_engine.training_history) > 100 else learning_engine.training_history
        
        if not recent_data:
            return
        
        # Procesar cada modelo
        for model_name, model in learning_engine.models.items():
            try:
                # Preparar datos para el modelo
                X, y = learning_engine._prepare_training_data(recent_data, model_name)
                
                if X is not None and y is not None and len(X) > 0:
                    # Entrenamiento incremental
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X, y)
                    else:
                        # Re-entrenar con datos recientes
                        model.fit(X, y)
                    
                    logger.info(f"Modelo {model_name} actualizado con aprendizaje incremental")
                
            except Exception as e:
                logger.warning(f"Error en aprendizaje incremental para {model_name}: {e}")
        
    except Exception as e:
        logger.error(f"Error en aprendizaje incremental: {e}")

async def _update_performance_metrics():
    """Actualiza las métricas de rendimiento"""
    try:
        learning_engine.performance_metrics = {
            "learning_cycles": learning_engine.learning_cycles,
            "total_models": len(learning_engine.models),
            "last_update": datetime.now().isoformat(),
            "training_data_points": len(learning_engine.training_history)
        }
        
        # Calcular métricas de rendimiento por modelo
        for model_name, model in learning_engine.models.items():
            try:
                if hasattr(model, 'score'):
                    # Obtener datos de prueba
                    test_data = learning_engine.training_history[-20:] if len(learning_engine.training_history) > 20 else learning_engine.training_history
                    if test_data:
                        X_test, y_test = learning_engine._prepare_training_data(test_data, model_name)
                        if X_test is not None and y_test is not None and len(X_test) > 0:
                            score = model.score(X_test, y_test)
                            learning_engine.performance_metrics[f"{model_name}_score"] = float(score)
            
            except Exception as e:
                logger.warning(f"Error calculando métricas para {model_name}: {e}")
        
    except Exception as e:
        logger.error(f"Error actualizando métricas de rendimiento: {e}")

def run_modulo2():
    """Función de compatibilidad con el sistema anterior"""
    print("🧠 Módulo 2: Sistema de Aprendizaje y Adaptación")
    print("   - Motor de aprendizaje automático")
    print("   - Modelos de machine learning")
    print("   - Aprendizaje incremental")
    print("   - Evaluación de rendimiento")
    print("   ✅ Módulo inicializado correctamente")