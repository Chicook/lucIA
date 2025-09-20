"""
Optimizadores para Redes Neuronales
Versión: 0.6.0
Implementación de diferentes algoritmos de optimización
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger('Neural_Optimizers')

class Optimizer(ABC):
    """Clase base abstracta para optimizadores"""
    
    def __init__(self, learning_rate: float = 0.001, name: str = None):
        self.learning_rate = learning_rate
        self.name = name or self.__class__.__name__
        self.iterations = 0
    
    @abstractmethod
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando los gradientes"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del optimizador"""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'iterations': self.iterations
        }

class SGD(Optimizer):
    """Descenso de Gradiente Estocástico (SGD)"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, 
                 nesterov: bool = False, name: str = None):
        super().__init__(learning_rate, name)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = {}
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando SGD"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(param)
            
            if self.momentum > 0:
                # SGD con momentum
                self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * gradients[key]
                
                if self.nesterov:
                    # Nesterov momentum
                    updated_parameters[key] = param + self.momentum * self.velocity[key] - self.learning_rate * gradients[key]
                else:
                    updated_parameters[key] = param + self.velocity[key]
            else:
                # SGD básico
                updated_parameters[key] = param - self.learning_rate * gradients[key]
        
        self.iterations += 1
        return updated_parameters

class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam)"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Primer momento
        self.v = {}  # Segundo momento
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando Adam"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            # Actualizar momentos
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Corrección de sesgo
            m_hat = self.m[key] / (1 - self.beta1 ** (self.iterations + 1))
            v_hat = self.v[key] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Actualizar parámetros
            updated_parameters[key] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class RMSprop(Optimizer):
    """Root Mean Square Propagation (RMSprop)"""
    
    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9, 
                 epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando RMSprop"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.cache:
                self.cache[key] = np.zeros_like(param)
            
            # Actualizar cache (promedio móvil de gradientes al cuadrado)
            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (gradients[key] ** 2)
            
            # Actualizar parámetros
            updated_parameters[key] = param - self.learning_rate * gradients[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class Adagrad(Optimizer):
    """Adaptive Gradient Algorithm (Adagrad)"""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando Adagrad"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.cache:
                self.cache[key] = np.zeros_like(param)
            
            # Acumular gradientes al cuadrado
            self.cache[key] += gradients[key] ** 2
            
            # Actualizar parámetros
            updated_parameters[key] = param - self.learning_rate * gradients[key] / (np.sqrt(self.cache[key]) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class AdamW(Optimizer):
    """Adam with Weight Decay (AdamW)"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0.01, name: str = None):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}  # Primer momento
        self.v = {}  # Segundo momento
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando AdamW"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            # Aplicar weight decay
            param_with_decay = param - self.learning_rate * self.weight_decay * param
            
            # Actualizar momentos
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Corrección de sesgo
            m_hat = self.m[key] / (1 - self.beta1 ** (self.iterations + 1))
            v_hat = self.v[key] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Actualizar parámetros
            updated_parameters[key] = param_with_decay - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class AdaDelta(Optimizer):
    """AdaDelta - An Adaptive Learning Rate Method"""
    
    def __init__(self, learning_rate: float = 1.0, rho: float = 0.95, 
                 epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.rho = rho
        self.epsilon = epsilon
        self.accum_grad = {}  # Acumulador de gradientes
        self.accum_delta = {}  # Acumulador de deltas
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando AdaDelta"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.accum_grad:
                self.accum_grad[key] = np.zeros_like(param)
                self.accum_delta[key] = np.zeros_like(param)
            
            # Actualizar acumulador de gradientes
            self.accum_grad[key] = self.rho * self.accum_grad[key] + (1 - self.rho) * (gradients[key] ** 2)
            
            # Calcular delta
            delta = -np.sqrt(self.accum_delta[key] + self.epsilon) / np.sqrt(self.accum_grad[key] + self.epsilon) * gradients[key]
            
            # Actualizar acumulador de deltas
            self.accum_delta[key] = self.rho * self.accum_delta[key] + (1 - self.rho) * (delta ** 2)
            
            # Actualizar parámetros
            updated_parameters[key] = param + delta
        
        self.iterations += 1
        return updated_parameters

class Nadam(Optimizer):
    """Nesterov-accelerated Adaptive Moment Estimation (Nadam)"""
    
    def __init__(self, learning_rate: float = 0.002, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Primer momento
        self.v = {}  # Segundo momento
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando Nadam"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            # Actualizar momentos
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Corrección de sesgo
            m_hat = self.m[key] / (1 - self.beta1 ** (self.iterations + 1))
            v_hat = self.v[key] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Aplicar Nesterov momentum
            m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * gradients[key] / (1 - self.beta1 ** (self.iterations + 1))
            
            # Actualizar parámetros
            updated_parameters[key] = param - self.learning_rate * m_nesterov / (np.sqrt(v_hat) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class RAdam(Optimizer):
    """Rectified Adam (RAdam)"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, name: str = None):
        super().__init__(learning_rate, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Primer momento
        self.v = {}  # Segundo momento
    
    def update(self, parameters: Dict[str, np.ndarray], 
               gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Actualiza los parámetros usando RAdam"""
        updated_parameters = {}
        
        for key, param in parameters.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            
            # Actualizar momentos
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Corrección de sesgo
            m_hat = self.m[key] / (1 - self.beta1 ** (self.iterations + 1))
            v_hat = self.v[key] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Rectificación
            rho_inf = 2 / (1 - self.beta2) - 1
            rho_t = rho_inf - 2 * (self.iterations + 1) * self.beta2 ** (self.iterations + 1) / (1 - self.beta2 ** (self.iterations + 1))
            
            if rho_t > 4:  # Rectificación activa
                r_t = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                updated_parameters[key] = param - self.learning_rate * r_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:  # Sin rectificación
                updated_parameters[key] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.iterations += 1
        return updated_parameters

class LearningRateScheduler:
    """Programador de tasa de aprendizaje"""
    
    def __init__(self, optimizer: Optimizer, schedule_type: str = 'step', 
                 **kwargs):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.initial_lr = optimizer.learning_rate
    
    def step(self, epoch: int) -> float:
        """Actualiza la tasa de aprendizaje según el programa"""
        if self.schedule_type == 'step':
            # Reducción por pasos
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            new_lr = self.initial_lr * (gamma ** (epoch // step_size))
        
        elif self.schedule_type == 'exponential':
            # Decaimiento exponencial
            gamma = self.kwargs.get('gamma', 0.95)
            new_lr = self.initial_lr * (gamma ** epoch)
        
        elif self.schedule_type == 'cosine':
            # Decaimiento coseno
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            new_lr = eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
        
        elif self.schedule_type == 'plateau':
            # Reducción en meseta
            factor = self.kwargs.get('factor', 0.5)
            patience = self.kwargs.get('patience', 10)
            min_lr = self.kwargs.get('min_lr', 1e-6)
            
            # Esta implementación es simplificada
            # En una implementación real, necesitarías monitorear la pérdida
            new_lr = max(self.optimizer.learning_rate * factor, min_lr)
        
        else:
            new_lr = self.initial_lr
        
        self.optimizer.learning_rate = new_lr
        return new_lr

class GradientClipping:
    """Recorte de gradientes para estabilizar el entrenamiento"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: str = 'l2'):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Recorta los gradientes según la norma especificada"""
        if self.norm_type == 'l2':
            # Calcular norma L2 global
            total_norm = 0
            for grad in gradients.values():
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)
            
            # Recortar si es necesario
            if total_norm > self.max_norm:
                clip_coef = self.max_norm / (total_norm + 1e-8)
                clipped_gradients = {}
                for key, grad in gradients.items():
                    clipped_gradients[key] = grad * clip_coef
                return clipped_gradients
        
        elif self.norm_type == 'l1':
            # Calcular norma L1 global
            total_norm = sum(np.sum(np.abs(grad)) for grad in gradients.values())
            
            # Recortar si es necesario
            if total_norm > self.max_norm:
                clip_coef = self.max_norm / (total_norm + 1e-8)
                clipped_gradients = {}
                for key, grad in gradients.items():
                    clipped_gradients[key] = grad * clip_coef
                return clipped_gradients
        
        return gradients
