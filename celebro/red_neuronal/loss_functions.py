"""
Funciones de Pérdida para Redes Neuronales
Versión: 0.6.0
Implementación de diferentes funciones de pérdida
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger('Neural_Loss_Functions')

class LossFunction(ABC):
    """Clase base abstracta para funciones de pérdida"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida"""
        pass
    
    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de la función de pérdida"""
        return {
            'name': self.name,
            'class': self.__class__.__name__
        }

class MSE(LossFunction):
    """Error Cuadrático Medio (Mean Squared Error)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula MSE"""
        return np.mean((y_true - y_pred) ** 2)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de MSE"""
        return 2 * (y_pred - y_true) / y_true.size

class MAE(LossFunction):
    """Error Absoluto Medio (Mean Absolute Error)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula MAE"""
        return np.mean(np.abs(y_true - y_pred))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de MAE"""
        return np.sign(y_pred - y_true) / y_true.size

class Huber(LossFunction):
    """Pérdida de Huber (robusta a outliers)"""
    
    def __init__(self, delta: float = 1.0, name: str = None):
        super().__init__(name)
        self.delta = delta
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida de Huber"""
        error = y_true - y_pred
        abs_error = np.abs(error)
        
        # Aplicar función de Huber
        loss = np.where(
            abs_error <= self.delta,
            0.5 * error ** 2,
            self.delta * (abs_error - 0.5 * self.delta)
        )
        
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida de Huber"""
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Aplicar gradiente de Huber
        gradient = np.where(
            abs_error <= self.delta,
            error,
            self.delta * np.sign(error)
        )
        
        return gradient / y_true.size

class CrossEntropy(LossFunction):
    """Entropía Cruzada Categórica"""
    
    def __init__(self, from_logits: bool = False, label_smoothing: float = 0.0, name: str = None):
        super().__init__(name)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la entropía cruzada categórica"""
        if self.from_logits:
            # Aplicar softmax si viene de logits
            y_pred = self._softmax(y_pred)
        
        # Aplicar label smoothing si está habilitado
        if self.label_smoothing > 0:
            y_true = self._apply_label_smoothing(y_true, self.label_smoothing)
        
        # Calcular pérdida
        loss = -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la entropía cruzada categórica"""
        if self.from_logits:
            # Si viene de logits, el gradiente es y_pred - y_true
            return (y_pred - y_true) / y_true.shape[0]
        else:
            # Si ya es probabilidades, el gradiente es -y_true / y_pred
            return -y_true / (y_pred + 1e-15) / y_true.shape[0]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Implementación de softmax numéricamente estable"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _apply_label_smoothing(self, y_true: np.ndarray, smoothing: float) -> np.ndarray:
        """Aplica label smoothing"""
        num_classes = y_true.shape[-1]
        return (1 - smoothing) * y_true + smoothing / num_classes

class BinaryCrossEntropy(LossFunction):
    """Entropía Cruzada Binaria"""
    
    def __init__(self, from_logits: bool = False, name: str = None):
        super().__init__(name)
        self.from_logits = from_logits
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la entropía cruzada binaria"""
        if self.from_logits:
            # Aplicar sigmoid si viene de logits
            y_pred = self._sigmoid(y_pred)
        
        # Calcular pérdida
        loss = -(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la entropía cruzada binaria"""
        if self.from_logits:
            # Si viene de logits, el gradiente es y_pred - y_true
            return (y_pred - y_true) / y_true.shape[0]
        else:
            # Si ya es probabilidades, el gradiente es (y_pred - y_true) / (y_pred * (1 - y_pred))
            return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15) / y_true.shape[0]
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Implementación de sigmoid numéricamente estable"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SparseCategoricalCrossEntropy(LossFunction):
    """Entropía Cruzada Categórica Esparsa"""
    
    def __init__(self, from_logits: bool = False, name: str = None):
        super().__init__(name)
        self.from_logits = from_logits
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la entropía cruzada categórica esparsa"""
        if self.from_logits:
            # Aplicar softmax si viene de logits
            y_pred = self._softmax(y_pred)
        
        # Convertir etiquetas esparsas a one-hot
        y_true_one_hot = self._to_one_hot(y_true, y_pred.shape[-1])
        
        # Calcular pérdida
        loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-15), axis=1)
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la entropía cruzada categórica esparsa"""
        if self.from_logits:
            # Convertir etiquetas esparsas a one-hot
            y_true_one_hot = self._to_one_hot(y_true, y_pred.shape[-1])
            return (y_pred - y_true_one_hot) / y_true.shape[0]
        else:
            # Convertir etiquetas esparsas a one-hot
            y_true_one_hot = self._to_one_hot(y_true, y_pred.shape[-1])
            return -y_true_one_hot / (y_pred + 1e-15) / y_true.shape[0]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Implementación de softmax numéricamente estable"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _to_one_hot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convierte etiquetas esparsas a one-hot"""
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        return one_hot

class KLDivergence(LossFunction):
    """Divergencia de Kullback-Leibler"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la divergencia de KL"""
        # Asegurar que las distribuciones sumen 1
        y_true_norm = y_true / (np.sum(y_true, axis=-1, keepdims=True) + 1e-15)
        y_pred_norm = y_pred / (np.sum(y_pred, axis=-1, keepdims=True) + 1e-15)
        
        # Calcular KL divergence
        kl_div = np.sum(y_true_norm * np.log(y_true_norm / (y_pred_norm + 1e-15)), axis=1)
        return np.mean(kl_div)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la divergencia de KL"""
        # Asegurar que las distribuciones sumen 1
        y_true_norm = y_true / (np.sum(y_true, axis=-1, keepdims=True) + 1e-15)
        y_pred_norm = y_pred / (np.sum(y_pred, axis=-1, keepdims=True) + 1e-15)
        
        # Gradiente de KL divergence
        gradient = -y_true_norm / (y_pred_norm + 1e-15)
        return gradient / y_true.shape[0]

class CosineSimilarity(LossFunction):
    """Pérdida de Similitud Coseno"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida de similitud coseno"""
        # Normalizar vectores
        y_true_norm = y_true / (np.linalg.norm(y_true, axis=-1, keepdims=True) + 1e-15)
        y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-15)
        
        # Calcular similitud coseno
        cosine_sim = np.sum(y_true_norm * y_pred_norm, axis=-1)
        
        # Convertir a pérdida (1 - similitud)
        loss = 1 - cosine_sim
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la similitud coseno"""
        # Normalizar vectores
        y_true_norm = y_true / (np.linalg.norm(y_true, axis=-1, keepdims=True) + 1e-15)
        y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-15)
        
        # Calcular similitud coseno
        cosine_sim = np.sum(y_true_norm * y_pred_norm, axis=-1, keepdims=True)
        
        # Gradiente de similitud coseno
        gradient = y_true_norm - cosine_sim * y_pred_norm
        return -gradient / y_true.shape[0]

class FocalLoss(LossFunction):
    """Pérdida Focal (para problemas de clasificación desbalanceados)"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, name: str = None):
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida focal"""
        # Aplicar softmax si es necesario
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = self._softmax(y_pred)
        
        # Calcular pérdida focal
        ce_loss = -y_true * np.log(y_pred + 1e-15)
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return np.mean(focal_loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida focal"""
        # Aplicar softmax si es necesario
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = self._softmax(y_pred)
        
        # Calcular gradiente de pérdida focal
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        gradient = alpha_t * (1 - p_t) ** self.gamma * (
            (1 - p_t) * np.log(p_t + 1e-15) - p_t
        ) * (y_pred - y_true)
        
        return gradient / y_true.shape[0]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Implementación de softmax numéricamente estable"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class DiceLoss(LossFunction):
    """Pérdida de Dice (para segmentación)"""
    
    def __init__(self, smooth: float = 1e-5, name: str = None):
        super().__init__(name)
        self.smooth = smooth
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida de Dice"""
        # Aplicar sigmoid si es necesario
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = self._sigmoid(y_pred)
        
        # Calcular intersección y unión
        intersection = np.sum(y_true * y_pred, axis=-1)
        union = np.sum(y_true, axis=-1) + np.sum(y_pred, axis=-1)
        
        # Calcular coeficiente de Dice
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Convertir a pérdida
        loss = 1 - dice
        return np.mean(loss)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida de Dice"""
        # Aplicar sigmoid si es necesario
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = self._sigmoid(y_pred)
        
        # Calcular intersección y unión
        intersection = np.sum(y_true * y_pred, axis=-1, keepdims=True)
        union = np.sum(y_true, axis=-1, keepdims=True) + np.sum(y_pred, axis=-1, keepdims=True)
        
        # Calcular gradiente
        gradient = 2 * (y_true * union - intersection) / (union + self.smooth) ** 2
        
        return gradient / y_true.shape[0]
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Implementación de sigmoid numéricamente estable"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class CombinedLoss(LossFunction):
    """Pérdida combinada (combinación de múltiples pérdidas)"""
    
    def __init__(self, losses: list, weights: list = None, name: str = None):
        super().__init__(name)
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)
        
        if len(self.weights) != len(self.losses):
            raise ValueError("El número de pesos debe coincidir con el número de pérdidas")
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula la pérdida combinada"""
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss.compute(y_true, y_pred)
        return total_loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calcula el gradiente de la pérdida combinada"""
        total_gradient = np.zeros_like(y_pred)
        for loss, weight in zip(self.losses, self.weights):
            total_gradient += weight * loss.gradient(y_true, y_pred)
        return total_gradient
