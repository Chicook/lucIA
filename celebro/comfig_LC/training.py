"""
Sistema de Entrenamiento para Redes Neuronales
Versión: 0.6.0
Implementación del sistema de entrenamiento completo
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json

logger = logging.getLogger('Neural_Training')

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    verbose: int = 1
    shuffle: bool = True
    callbacks: List[Callable] = None

@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento"""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    timestamp: datetime

class Callback:
    """Clase base para callbacks de entrenamiento"""
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        """Llamado al inicio del entrenamiento"""
        pass
    
    def on_train_end(self, logs: Dict[str, Any] = None):
        """Llamado al final del entrenamiento"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        """Llamado al inicio de cada época"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        """Llamado al final de cada época"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any] = None):
        """Llamado al inicio de cada lote"""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any] = None):
        """Llamado al final de cada lote"""
        pass

class EarlyStopping(Callback):
    """Callback para early stopping"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, 
                 min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_value = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_value = logs.get(self.monitor, float('inf'))
        
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                # Guardar mejores pesos (implementar según necesidad)
                pass
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True  # Detener entrenamiento
        
        return False

class LearningRateScheduler(Callback):
    """Callback para programar la tasa de aprendizaje"""
    
    def __init__(self, schedule: Callable[[int, float], float]):
        self.schedule = schedule
        self.learning_rates = []
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None):
        # Obtener tasa de aprendizaje actual
        current_lr = logs.get('learning_rate', 0.001)
        
        # Calcular nueva tasa
        new_lr = self.schedule(epoch, current_lr)
        
        # Actualizar (implementar según necesidad)
        logs['learning_rate'] = new_lr
        self.learning_rates.append(new_lr)

class ModelCheckpoint(Callback):
    """Callback para guardar el modelo"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', 
                 save_best_only: bool = True, save_weights_only: bool = False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_value = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_value = logs.get(self.monitor, float('inf'))
        
        if current_value < self.best_value:
            self.best_value = current_value
            # Guardar modelo (implementar según necesidad)
            pass

class ReduceLROnPlateau(Callback):
    """Callback para reducir la tasa de aprendizaje en meseta"""
    
    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5, 
                 patience: int = 10, min_lr: float = 1e-7):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_value = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] = None):
        current_value = logs.get(self.monitor, float('inf'))
        
        if current_value < self.best_value:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reducir tasa de aprendizaje
                current_lr = logs.get('learning_rate', 0.001)
                new_lr = max(current_lr * self.factor, self.min_lr)
                logs['learning_rate'] = new_lr
                self.wait = 0

class Trainer:
    """Entrenador de redes neuronales"""
    
    def __init__(self, network, config: TrainingConfig = None):
        self.network = network
        self.config = config or TrainingConfig()
        self.history = []
        self.callbacks = self.config.callbacks or []
        
        # Agregar callbacks por defecto
        if self.config.early_stopping:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                restore_best_weights=self.config.restore_best_weights
            )
            self.callbacks.append(early_stopping)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray = None, y_val: np.ndarray = None) -> List[TrainingMetrics]:
        """Entrena la red neuronal"""
        try:
            # Preparar datos de validación
            if x_val is None or y_val is None:
                x_val, y_val = self._split_validation_data(x_train, y_train)
            
            # Llamar callbacks de inicio
            self._call_callbacks('on_train_begin', {'x_train': x_train, 'y_train': y_train})
            
            # Entrenar por épocas
            for epoch in range(self.config.epochs):
                # Llamar callbacks de inicio de época
                epoch_logs = {'epoch': epoch}
                self._call_callbacks('on_epoch_begin', epoch_logs)
                
                # Entrenar una época
                train_metrics = self._train_epoch(x_train, y_train, epoch)
                
                # Evaluar en validación
                val_metrics = self._evaluate_epoch(x_val, y_val, epoch)
                
                # Crear métricas combinadas
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=train_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    val_loss=val_metrics['loss'],
                    val_accuracy=val_metrics['accuracy'],
                    learning_rate=epoch_logs.get('learning_rate', self.config.learning_rate),
                    timestamp=datetime.now()
                )
                
                self.history.append(metrics)
                
                # Preparar logs para callbacks
                epoch_logs.update({
                    'train_loss': metrics.train_loss,
                    'train_accuracy': metrics.train_accuracy,
                    'val_loss': metrics.val_loss,
                    'val_accuracy': metrics.val_accuracy,
                    'learning_rate': metrics.learning_rate
                })
                
                # Llamar callbacks de final de época
                should_stop = self._call_callbacks('on_epoch_end', epoch_logs)
                
                # Log de progreso
                if self.config.verbose > 0:
                    self._log_epoch(metrics)
                
                # Verificar early stopping
                if should_stop:
                    logger.info(f"Entrenamiento detenido por early stopping en época {epoch + 1}")
                    break
            
            # Llamar callbacks de final
            self._call_callbacks('on_train_end', {'history': self.history})
            
            logger.info("Entrenamiento completado")
            return self.history
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            raise
    
    def _split_validation_data(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Divide los datos de entrenamiento en entrenamiento y validación"""
        try:
            n_samples = x_train.shape[0]
            n_val = int(n_samples * self.config.validation_split)
            
            # Mezclar datos
            if self.config.shuffle:
                indices = np.random.permutation(n_samples)
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            # Dividir
            x_val = x_train[:n_val]
            y_val = y_train[:n_val]
            x_train = x_train[n_val:]
            y_train = y_train[n_val:]
            
            return x_val, y_val
            
        except Exception as e:
            logger.error(f"Error dividiendo datos de validación: {e}")
            raise
    
    def _train_epoch(self, x_train: np.ndarray, y_train: np.ndarray, epoch: int) -> Dict[str, float]:
        """Entrena una época"""
        try:
            n_samples = x_train.shape[0]
            batch_size = self.config.batch_size
            n_batches = n_samples // batch_size
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for batch in range(n_batches):
                # Obtener lote
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Llamar callbacks de inicio de lote
                batch_logs = {'batch': batch, 'epoch': epoch}
                self._call_callbacks('on_batch_begin', batch_logs)
                
                # Forward pass
                predictions = self.network.forward(x_batch)
                
                # Calcular pérdida
                loss = self.network.loss_function.compute(y_batch, predictions)
                
                # Calcular gradientes
                loss_gradients = self.network.loss_function.gradient(y_batch, predictions)
                
                # Backward pass
                self.network.backward(loss_gradients)
                
                # Actualizar parámetros
                self._update_parameters()
                
                # Acumular métricas
                epoch_loss += loss
                
                # Calcular precisión
                if self.network.config.output_activation == 'softmax':
                    predicted_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predicted_classes == true_classes)
                else:
                    accuracy = 1.0 - (loss / np.var(y_batch))
                
                epoch_accuracy += accuracy
                
                # Llamar callbacks de final de lote
                batch_logs.update({'loss': loss, 'accuracy': accuracy})
                self._call_callbacks('on_batch_end', batch_logs)
            
            return {
                'loss': epoch_loss / n_batches,
                'accuracy': epoch_accuracy / n_batches
            }
            
        except Exception as e:
            logger.error(f"Error entrenando época: {e}")
            raise
    
    def _evaluate_epoch(self, x_val: np.ndarray, y_val: np.ndarray, epoch: int) -> Dict[str, float]:
        """Evalúa una época en datos de validación"""
        try:
            # Establecer modo de inferencia
            for layer in self.network.layers:
                layer.set_training(False)
            
            # Evaluar
            metrics = self.network.evaluate(x_val, y_val)
            
            # Restaurar modo de entrenamiento
            for layer in self.network.layers:
                layer.set_training(True)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando época: {e}")
            raise
    
    def _update_parameters(self):
        """Actualiza los parámetros de la red"""
        try:
            # Recopilar gradientes de todas las capas
            all_gradients = {}
            all_parameters = {}
            
            for layer in self.network.layers:
                layer_gradients = layer.get_gradients()
                layer_parameters = layer.get_parameters()
                
                for key, grad in layer_gradients.items():
                    full_key = f"{layer.name}_{key}"
                    all_gradients[full_key] = grad
                    all_parameters[full_key] = layer_parameters[key]
            
            # Actualizar parámetros
            updated_parameters = self.network.optimizer.update(all_parameters, all_gradients)
            
            # Distribuir parámetros actualizados a las capas
            for layer in self.network.layers:
                layer_parameters = {}
                for key, param in layer.get_parameters().items():
                    full_key = f"{layer.name}_{key}"
                    if full_key in updated_parameters:
                        layer_parameters[key] = updated_parameters[full_key]
                    else:
                        layer_parameters[key] = param
                
                layer.set_parameters(layer_parameters)
            
        except Exception as e:
            logger.error(f"Error actualizando parámetros: {e}")
            raise
    
    def _call_callbacks(self, method_name: str, logs: Dict[str, Any] = None) -> bool:
        """Llama a un método de todos los callbacks"""
        should_stop = False
        
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                method = getattr(callback, method_name)
                result = method(logs or {})
                if result is True:
                    should_stop = True
        
        return should_stop
    
    def _log_epoch(self, metrics: TrainingMetrics):
        """Registra el progreso de una época"""
        print(f"Época {metrics.epoch:3d} - "
              f"Pérdida: {metrics.train_loss:.4f} - "
              f"Precisión: {metrics.train_accuracy:.4f} - "
              f"Val. Pérdida: {metrics.val_loss:.4f} - "
              f"Val. Precisión: {metrics.val_accuracy:.4f}")
    
    def get_history(self) -> List[TrainingMetrics]:
        """Obtiene el historial de entrenamiento"""
        return self.history
    
    def save_history(self, filepath: str):
        """Guarda el historial de entrenamiento"""
        try:
            history_data = []
            for metrics in self.history:
                history_data.append({
                    'epoch': metrics.epoch,
                    'train_loss': metrics.train_loss,
                    'train_accuracy': metrics.train_accuracy,
                    'val_loss': metrics.val_loss,
                    'val_accuracy': metrics.val_accuracy,
                    'learning_rate': metrics.learning_rate,
                    'timestamp': metrics.timestamp.isoformat()
                })
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Historial guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")
            raise
    
    def load_history(self, filepath: str):
        """Carga el historial de entrenamiento"""
        try:
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            self.history = []
            for data in history_data:
                metrics = TrainingMetrics(
                    epoch=data['epoch'],
                    train_loss=data['train_loss'],
                    train_accuracy=data['train_accuracy'],
                    val_loss=data['val_loss'],
                    val_accuracy=data['val_accuracy'],
                    learning_rate=data['learning_rate'],
                    timestamp=datetime.fromisoformat(data['timestamp'])
                )
                self.history.append(metrics)
            
            logger.info(f"Historial cargado desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando historial: {e}")
            raise
