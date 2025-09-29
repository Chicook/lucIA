"""
Funciones de Activación para Redes Neuronales
Versión: 0.6.0
Implementación de diferentes funciones de activación
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger('Neural_Activations')

class ActivationFunction(ABC):
    """Clase base abstracta para funciones de activación"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica la función de activación"""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función de activación"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de la función"""
        return {
            'name': self.name,
            'class': self.__class__.__name__
        }

class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica ReLU: max(0, x)"""
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ReLU: 1 si x > 0, 0 si x <= 0"""
        return np.where(x > 0, 1, 0)

class LeakyReLU(ActivationFunction):
    """Leaky Rectified Linear Unit"""
    
    def __init__(self, alpha: float = 0.01, name: str = None):
        super().__init__(name)
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica LeakyReLU: x si x > 0, alpha * x si x <= 0"""
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de LeakyReLU: 1 si x > 0, alpha si x <= 0"""
        return np.where(x > 0, 1, self.alpha)

class ELU(ActivationFunction):
    """Exponential Linear Unit (ELU)"""
    
    def __init__(self, alpha: float = 1.0, name: str = None):
        super().__init__(name)
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica ELU: x si x > 0, alpha * (exp(x) - 1) si x <= 0"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de ELU: 1 si x > 0, alpha * exp(x) si x <= 0"""
        return np.where(x > 0, 1, self.alpha * np.exp(x))

class Sigmoid(ActivationFunction):
    """Función Sigmoid"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Sigmoid: 1 / (1 + exp(-x))"""
        # Clipping para estabilidad numérica
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)

class Tanh(ActivationFunction):
    """Función Tangente Hiperbólica"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Tanh: tanh(x)"""
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Tanh: 1 - tanh²(x)"""
        return 1 - np.tanh(x) ** 2

class Softmax(ActivationFunction):
    """Función Softmax"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Softmax: exp(x) / sum(exp(x))"""
        # Estabilidad numérica
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Softmax: softmax(x) * (1 - softmax(x))"""
        softmax_x = self.forward(x)
        return softmax_x * (1 - softmax_x)

class Swish(ActivationFunction):
    """Función Swish: x * sigmoid(x)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Swish: x * sigmoid(x)"""
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return x * sigmoid_x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
        sigmoid_x = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)

class GELU(ActivationFunction):
    """Gaussian Error Linear Unit (GELU)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de GELU"""
        # Implementación simplificada de la derivada de GELU
        tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
        return 0.5 * (1 + tanh_term) + 0.5 * x * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 0.134145 * x**2)

class Linear(ActivationFunction):
    """Función Lineal (sin activación)"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Linear: x"""
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Linear: 1"""
        return np.ones_like(x)

class Step(ActivationFunction):
    """Función Escalón"""
    
    def __init__(self, threshold: float = 0.0, name: str = None):
        super().__init__(name)
        self.threshold = threshold
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Step: 1 si x > threshold, 0 si x <= threshold"""
        return np.where(x > self.threshold, 1, 0)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Step: 0 (no diferenciable)"""
        return np.zeros_like(x)

class RReLU(ActivationFunction):
    """Randomized Leaky ReLU"""
    
    def __init__(self, lower: float = 1/8, upper: float = 1/3, name: str = None):
        super().__init__(name)
        self.lower = lower
        self.upper = upper
        self.alpha = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica RReLU con alpha aleatorio"""
        if self.alpha is None:
            self.alpha = np.random.uniform(self.lower, self.upper)
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de RReLU"""
        if self.alpha is None:
            self.alpha = np.random.uniform(self.lower, self.upper)
        return np.where(x > 0, 1, self.alpha)

class PReLU(ActivationFunction):
    """Parametric ReLU"""
    
    def __init__(self, alpha: float = 0.25, name: str = None):
        super().__init__(name)
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica PReLU: x si x > 0, alpha * x si x <= 0"""
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de PReLU: 1 si x > 0, alpha si x <= 0"""
        return np.where(x > 0, 1, self.alpha)

class Mish(ActivationFunction):
    """Función Mish: x * tanh(softplus(x))"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Mish: x * tanh(softplus(x))"""
        softplus_x = np.log(1 + np.exp(x))
        return x * np.tanh(softplus_x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Mish"""
        softplus_x = np.log(1 + np.exp(x))
        tanh_softplus = np.tanh(softplus_x)
        return tanh_softplus + x * (1 - tanh_softplus**2) * (1 / (1 + np.exp(-x)))

class HardSigmoid(ActivationFunction):
    """Hard Sigmoid"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Hard Sigmoid: clip(0.2 * x + 0.5, 0, 1)"""
        return np.clip(0.2 * x + 0.5, 0, 1)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Hard Sigmoid: 0.2 si -2.5 < x < 2.5, 0 si no"""
        return np.where((x > -2.5) & (x < 2.5), 0.2, 0)

class HardTanh(ActivationFunction):
    """Hard Tanh"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Aplica Hard Tanh: clip(x, -1, 1)"""
        return np.clip(x, -1, 1)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Hard Tanh: 1 si -1 < x < 1, 0 si no"""
        return np.where((x > -1) & (x < 1), 1, 0)

# Diccionario de funciones de activación disponibles
ACTIVATION_FUNCTIONS = {
    'linear': Linear,
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'swish': Swish,
    'gelu': GELU,
    'step': Step,
    'rrelu': RReLU,
    'prelu': PReLU,
    'mish': Mish,
    'hard_sigmoid': HardSigmoid,
    'hard_tanh': HardTanh
}

def get_activation(activation_name: str, **kwargs) -> ActivationFunction:
    """Obtiene una función de activación por nombre"""
    if activation_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Función de activación '{activation_name}' no encontrada. "
                        f"Disponibles: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    return ACTIVATION_FUNCTIONS[activation_name](**kwargs)
