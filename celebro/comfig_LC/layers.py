"""
Capas de Red Neuronal
Versión: 0.6.0
Implementación de diferentes tipos de capas para redes neuronales
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger('Neural_Layers')

class Layer(ABC):
    """Clase base abstracta para capas de la red neuronal"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.input_shape = None
        self.output_shape = None
        self.parameters = {}
        self.gradients = {}
        self.cache = {}
        self.is_training = True
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        pass
    
    @abstractmethod
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        pass
    
    @abstractmethod
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        pass
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Obtiene los parámetros de la capa"""
        return self.parameters
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """Establece los parámetros de la capa"""
        self.parameters.update(parameters)
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Obtiene los gradientes de la capa"""
        return self.gradients
    
    def set_training(self, training: bool) -> None:
        """Establece el modo de entrenamiento"""
        self.is_training = training

class DenseLayer(Layer):
    """Capa densa (completamente conectada)"""
    
    def __init__(self, units: int, activation: str = 'linear', 
                 use_bias: bool = True, kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros', name: str = None):
        super().__init__(name)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
    
    def _get_activation_function(self, activation: str):
        """Obtiene la función de activación"""
        activations = {
            'linear': lambda x: x,
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': lambda x: np.tanh(x),
            'softmax': lambda x: self._softmax(x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
            'swish': lambda x: x * (1 / (1 + np.exp(-x))),
            'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        }
        return activations.get(activation, activations['linear'])
    
    def _get_activation_derivative(self, activation: str):
        """Obtiene la derivada de la función de activación"""
        derivatives = {
            'linear': lambda x: np.ones_like(x),
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: self._sigmoid_derivative(x),
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'softmax': lambda x: self._softmax_derivative(x),
            'leaky_relu': lambda x: np.where(x > 0, 1, 0.01),
            'elu': lambda x: np.where(x > 0, 1, np.exp(x)),
            'swish': lambda x: self._swish_derivative(x),
            'gelu': lambda x: self._gelu_derivative(x)
        }
        return derivatives.get(activation, derivatives['linear'])
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Implementación de softmax numéricamente estable"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _softmax_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de softmax"""
        softmax_output = self._softmax(x)
        return softmax_output * (1 - softmax_output)
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de sigmoid"""
        sigmoid_output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid_output * (1 - sigmoid_output)
    
    def _swish_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de swish"""
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    
    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de GELU"""
        tanh_term = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
        return 0.5 * (1 + tanh_term) + 0.5 * x * (1 - tanh_term**2) * np.sqrt(2/np.pi) * (1 + 0.134145 * x**2)
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        input_units = input_shape[-1]
        
        # Inicializar pesos
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6.0 / (input_units + self.units))
            self.parameters['kernel'] = np.random.uniform(-limit, limit, (input_units, self.units))
        elif self.kernel_initializer == 'glorot_normal':
            std = np.sqrt(2.0 / (input_units + self.units))
            self.parameters['kernel'] = np.random.normal(0, std, (input_units, self.units))
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6.0 / input_units)
            self.parameters['kernel'] = np.random.uniform(-limit, limit, (input_units, self.units))
        elif self.kernel_initializer == 'he_normal':
            std = np.sqrt(2.0 / input_units)
            self.parameters['kernel'] = np.random.normal(0, std, (input_units, self.units))
        elif self.kernel_initializer == 'xavier_uniform':
            limit = np.sqrt(6.0 / (input_units + self.units))
            self.parameters['kernel'] = np.random.uniform(-limit, limit, (input_units, self.units))
        elif self.kernel_initializer == 'xavier_normal':
            std = np.sqrt(2.0 / (input_units + self.units))
            self.parameters['kernel'] = np.random.normal(0, std, (input_units, self.units))
        else:  # zeros
            self.parameters['kernel'] = np.zeros((input_units, self.units))
        
        # Inicializar sesgos
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.parameters['bias'] = np.zeros(self.units)
            elif self.bias_initializer == 'ones':
                self.parameters['bias'] = np.ones(self.units)
            else:  # random
                self.parameters['bias'] = np.random.normal(0, 0.1, self.units)
        
        self.output_shape = (self.units,)
        self.gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        # Guardar entrada para backward
        self.cache['inputs'] = inputs
        
        # Computar salida lineal
        linear_output = np.dot(inputs, self.parameters['kernel'])
        
        # Agregar sesgo si está habilitado
        if self.use_bias:
            linear_output += self.parameters['bias']
        
        # Aplicar función de activación
        output = self.activation_function(linear_output)
        
        # Guardar salida para backward
        self.cache['linear_output'] = linear_output
        self.cache['output'] = output
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        inputs = self.cache['inputs']
        linear_output = self.cache['linear_output']
        
        # Gradiente de la función de activación
        activation_grad = self.activation_derivative(linear_output)
        
        # Gradiente de la salida
        linear_gradients = gradients * activation_grad
        
        # Gradientes de los parámetros
        self.gradients['kernel'] = np.dot(inputs.T, linear_gradients)
        if self.use_bias:
            self.gradients['bias'] = np.sum(linear_gradients, axis=0)
        
        # Gradiente de la entrada
        input_gradients = np.dot(linear_gradients, self.parameters['kernel'].T)
        
        return input_gradients

class ConvLayer(Layer):
    """Capa convolucional 2D"""
    
    def __init__(self, filters: int, kernel_size: Tuple[int, int], 
                 strides: Tuple[int, int] = (1, 1), padding: str = 'valid',
                 activation: str = 'relu', use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform', name: str = None):
        super().__init__(name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
    
    def _get_activation_function(self, activation: str):
        """Obtiene la función de activación"""
        activations = {
            'linear': lambda x: x,
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': lambda x: np.tanh(x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1)
        }
        return activations.get(activation, activations['relu'])
    
    def _get_activation_derivative(self, activation: str):
        """Obtiene la derivada de la función de activación"""
        derivatives = {
            'linear': lambda x: np.ones_like(x),
            'relu': lambda x: np.where(x > 0, 1, 0),
            'sigmoid': lambda x: self._sigmoid_derivative(x),
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'leaky_relu': lambda x: np.where(x > 0, 1, 0.01),
            'elu': lambda x: np.where(x > 0, 1, np.exp(x))
        }
        return derivatives.get(activation, derivatives['relu'])
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de sigmoid"""
        sigmoid_output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid_output * (1 - sigmoid_output)
    
    def _pad_input(self, inputs: np.ndarray) -> np.ndarray:
        """Aplica padding a la entrada"""
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            return np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        return inputs
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        height, width, channels = input_shape[1:]
        
        # Calcular tamaño de salida
        if self.padding == 'same':
            out_height = height // self.strides[0]
            out_width = width // self.strides[1]
        else:  # valid
            out_height = (height - self.kernel_size[0]) // self.strides[0] + 1
            out_width = (width - self.kernel_size[1]) // self.strides[1] + 1
        
        self.output_shape = (out_height, out_width, self.filters)
        
        # Inicializar pesos
        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6.0 / (self.kernel_size[0] * self.kernel_size[1] * channels + self.filters))
            self.parameters['kernel'] = np.random.uniform(
                -limit, limit, 
                (self.kernel_size[0], self.kernel_size[1], channels, self.filters)
            )
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6.0 / (self.kernel_size[0] * self.kernel_size[1] * channels))
            self.parameters['kernel'] = np.random.uniform(
                -limit, limit,
                (self.kernel_size[0], self.kernel_size[1], channels, self.filters)
            )
        else:  # zeros
            self.parameters['kernel'] = np.zeros(
                (self.kernel_size[0], self.kernel_size[1], channels, self.filters)
            )
        
        # Inicializar sesgos
        if self.use_bias:
            self.parameters['bias'] = np.zeros(self.filters)
        
        self.gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        batch_size = inputs.shape[0]
        height, width, channels = inputs.shape[1:]
        
        # Aplicar padding si es necesario
        padded_inputs = self._pad_input(inputs)
        
        # Calcular tamaño de salida
        if self.padding == 'same':
            out_height = height // self.strides[0]
            out_width = width // self.strides[1]
        else:
            out_height = (height - self.kernel_size[0]) // self.strides[0] + 1
            out_width = (width - self.kernel_size[1]) // self.strides[1] + 1
        
        # Inicializar salida
        output = np.zeros((batch_size, out_height, out_width, self.filters))
        
        # Convolución
        for b in range(batch_size):
            for f in range(self.filters):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.strides[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.strides[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extraer región
                        region = padded_inputs[b, h_start:h_end, w_start:w_end, :]
                        
                        # Aplicar convolución
                        conv_result = np.sum(region * self.parameters['kernel'][:, :, :, f])
                        
                        # Agregar sesgo
                        if self.use_bias:
                            conv_result += self.parameters['bias'][f]
                        
                        output[b, h, w, f] = conv_result
        
        # Aplicar función de activación
        output = self.activation_function(output)
        
        # Guardar para backward
        self.cache['inputs'] = inputs
        self.cache['padded_inputs'] = padded_inputs
        self.cache['output'] = output
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        inputs = self.cache['inputs']
        padded_inputs = self.cache['padded_inputs']
        
        batch_size = inputs.shape[0]
        height, width, channels = inputs.shape[1:]
        
        # Gradiente de la función de activación
        activation_grad = self.activation_derivative(self.cache['output'])
        linear_gradients = gradients * activation_grad
        
        # Inicializar gradientes
        self.gradients['kernel'] = np.zeros_like(self.parameters['kernel'])
        if self.use_bias:
            self.gradients['bias'] = np.zeros_like(self.parameters['bias'])
        
        input_gradients = np.zeros_like(padded_inputs)
        
        # Calcular gradientes
        for b in range(batch_size):
            for f in range(self.filters):
                for h in range(linear_gradients.shape[1]):
                    for w in range(linear_gradients.shape[2]):
                        h_start = h * self.strides[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.strides[1]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Gradiente del kernel
                        region = padded_inputs[b, h_start:h_end, w_start:w_end, :]
                        self.gradients['kernel'][:, :, :, f] += region * linear_gradients[b, h, w, f]
                        
                        # Gradiente de la entrada
                        input_gradients[b, h_start:h_end, w_start:w_end, :] += \
                            self.parameters['kernel'][:, :, :, f] * linear_gradients[b, h, w, f]
                
                # Gradiente del sesgo
                if self.use_bias:
                    self.gradients['bias'][f] += np.sum(linear_gradients[b, :, :, f])
        
        # Remover padding del gradiente de entrada
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            input_gradients = input_gradients[:, pad_h:-pad_h, pad_w:-pad_w, :]
        
        return input_gradients

class PoolingLayer(Layer):
    """Capa de pooling (MaxPool, AvgPool)"""
    
    def __init__(self, pool_size: Tuple[int, int], strides: Tuple[int, int] = None,
                 padding: str = 'valid', pool_type: str = 'max', name: str = None):
        super().__init__(name)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding
        self.pool_type = pool_type
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        height, width, channels = input_shape[1:]
        
        # Calcular tamaño de salida
        if self.padding == 'same':
            out_height = height // self.strides[0]
            out_width = width // self.strides[1]
        else:  # valid
            out_height = (height - self.pool_size[0]) // self.strides[0] + 1
            out_width = (width - self.pool_size[1]) // self.strides[1] + 1
        
        self.output_shape = (out_height, out_width, channels)
        self.parameters = {}
        self.gradients = {}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        batch_size = inputs.shape[0]
        height, width, channels = inputs.shape[1:]
        
        # Calcular tamaño de salida
        if self.padding == 'same':
            out_height = height // self.strides[0]
            out_width = width // self.strides[1]
        else:
            out_height = (height - self.pool_size[0]) // self.strides[0] + 1
            out_width = (width - self.pool_size[1]) // self.strides[1] + 1
        
        # Inicializar salida
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        # Pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.strides[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.strides[1]
                        w_end = w_start + self.pool_size[1]
                        
                        # Extraer región
                        region = inputs[b, h_start:h_end, w_start:w_end, c]
                        
                        # Aplicar pooling
                        if self.pool_type == 'max':
                            output[b, h, w, c] = np.max(region)
                        elif self.pool_type == 'avg':
                            output[b, h, w, c] = np.mean(region)
        
        # Guardar para backward
        self.cache['inputs'] = inputs
        self.cache['output'] = output
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        inputs = self.cache['inputs']
        output = self.cache['output']
        
        batch_size = inputs.shape[0]
        height, width, channels = inputs.shape[1:]
        
        input_gradients = np.zeros_like(inputs)
        
        # Calcular gradientes
        for b in range(batch_size):
            for c in range(channels):
                for h in range(gradients.shape[1]):
                    for w in range(gradients.shape[2]):
                        h_start = h * self.strides[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.strides[1]
                        w_end = w_start + self.pool_size[1]
                        
                        # Extraer región
                        region = inputs[b, h_start:h_end, w_start:w_end, c]
                        
                        # Calcular gradiente
                        if self.pool_type == 'max':
                            # Solo el elemento máximo recibe el gradiente
                            max_idx = np.unravel_index(np.argmax(region), region.shape)
                            input_gradients[b, h_start + max_idx[0], w_start + max_idx[1], c] += \
                                gradients[b, h, w, c]
                        elif self.pool_type == 'avg':
                            # Todos los elementos reciben el gradiente promedio
                            avg_grad = gradients[b, h, w, c] / (self.pool_size[0] * self.pool_size[1])
                            input_gradients[b, h_start:h_end, w_start:w_end, c] += avg_grad
        
        return input_gradients

class DropoutLayer(Layer):
    """Capa de dropout para regularización"""
    
    def __init__(self, rate: float = 0.5, name: str = None):
        super().__init__(name)
        self.rate = rate
        self.scale = 1.0 / (1.0 - rate) if rate < 1.0 else 1.0
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.parameters = {}
        self.gradients = {}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        if self.is_training and self.rate > 0:
            # Crear máscara de dropout
            mask = np.random.binomial(1, 1 - self.rate, inputs.shape)
            # Aplicar dropout y escalar
            output = inputs * mask * self.scale
            # Guardar máscara para backward
            self.cache['mask'] = mask
        else:
            # En modo inferencia, no aplicar dropout
            output = inputs
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        if self.is_training and self.rate > 0:
            # Aplicar la misma máscara que en forward
            mask = self.cache['mask']
            input_gradients = gradients * mask * self.scale
        else:
            input_gradients = gradients
        
        return input_gradients

class BatchNormLayer(Layer):
    """Capa de normalización por lotes"""
    
    def __init__(self, momentum: float = 0.9, epsilon: float = 1e-5, name: str = None):
        super().__init__(name)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        self.output_shape = input_shape
        
        # Parámetros aprendibles
        self.parameters['gamma'] = np.ones(input_shape[-1])
        self.parameters['beta'] = np.zeros(input_shape[-1])
        
        # Estadísticas móviles
        self.running_mean = np.zeros(input_shape[-1])
        self.running_var = np.ones(input_shape[-1])
        
        self.gradients = {key: np.zeros_like(value) for key, value in self.parameters.items()}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        if self.is_training:
            # Calcular estadísticas del lote actual
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            
            # Actualizar estadísticas móviles
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # Normalizar
            normalized = (inputs - mean) / np.sqrt(var + self.epsilon)
            
            # Guardar para backward
            self.cache['inputs'] = inputs
            self.cache['mean'] = mean
            self.cache['var'] = var
            self.cache['normalized'] = normalized
        else:
            # En modo inferencia, usar estadísticas móviles
            normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Aplicar escala y desplazamiento
        output = self.parameters['gamma'] * normalized + self.parameters['beta']
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        if not self.is_training:
            return gradients
        
        inputs = self.cache['inputs']
        mean = self.cache['mean']
        var = self.cache['var']
        normalized = self.cache['normalized']
        
        m = inputs.shape[0]
        
        # Gradientes de gamma y beta
        self.gradients['gamma'] = np.sum(gradients * normalized, axis=0)
        self.gradients['beta'] = np.sum(gradients, axis=0)
        
        # Gradiente de la normalización
        gamma = self.parameters['gamma']
        var_sqrt = np.sqrt(var + self.epsilon)
        
        # Gradiente de la normalización
        dnormalized = gradients * gamma
        
        # Gradiente de la varianza
        dvar = np.sum(dnormalized * (inputs - mean), axis=0) * -0.5 * (var + self.epsilon) ** -1.5
        
        # Gradiente de la media
        dmean = np.sum(dnormalized * -1 / var_sqrt, axis=0) + dvar * np.sum(-2 * (inputs - mean), axis=0) / m
        
        # Gradiente de la entrada
        input_gradients = dnormalized / var_sqrt + dvar * 2 * (inputs - mean) / m + dmean / m
        
        return input_gradients

class FlattenLayer(Layer):
    """Capa de aplanado para conectar capas convolucionales con densas"""
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.parameters = {}
        self.gradients = {}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        batch_size = inputs.shape[0]
        output = inputs.reshape(batch_size, -1)
        
        # Guardar forma original para backward
        self.cache['original_shape'] = inputs.shape
        
        return output
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """Propagación hacia atrás"""
        original_shape = self.cache['original_shape']
        input_gradients = gradients.reshape(original_shape)
        
        return input_gradients
