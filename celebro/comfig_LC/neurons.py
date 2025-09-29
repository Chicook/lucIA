"""
Neuronas y capas neuronales
Versión: 0.6.0
Implementación de neuronas básicas (perceptrón) y neuronas de disparo (LIF)
compatibles con la arquitectura de capas existente.
"""

import numpy as np
import logging
from typing import Tuple

from .layers import Layer


logger = logging.getLogger('Neural_Neurons')


class NeuronLayer(Layer):
    """
    Capa de neuronas tipo perceptrón (equivalente funcional a Dense),
    expone semántica de "neuronas" para facilitar razonamiento/visualización.
    """

    def __init__(self, num_neurons: int, activation: str = 'relu', use_bias: bool = True, name: str = None):
        super().__init__(name)
        self.num_neurons = num_neurons
        self.activation = activation
        self.use_bias = use_bias

    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape
        input_units = input_shape[-1]

        # Inicialización estilo Xavier/Glorot para estabilidad
        limit = np.sqrt(6.0 / (input_units + self.num_neurons))
        self.parameters['weights'] = np.random.uniform(-limit, limit, (input_units, self.num_neurons))
        if self.use_bias:
            self.parameters['bias'] = np.zeros(self.num_neurons)

        self.output_shape = (self.num_neurons,)
        self.gradients = {key: np.zeros_like(val) for key, val in self.parameters.items()}

    def _activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, x)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        if self.activation == 'tanh':
            return np.tanh(x)
        if self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        if self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        return x

    def _activation_derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        if self.activation == 'sigmoid':
            return y * (1 - y)
        if self.activation == 'tanh':
            return 1 - y ** 2
        if self.activation == 'softmax':
            return y * (1 - y)
        if self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        return np.ones_like(x)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.cache['inputs'] = inputs
        z = np.dot(inputs, self.parameters['weights'])
        if self.use_bias:
            z += self.parameters['bias']
        a = self._activation(z)
        self.cache['z'] = z
        self.cache['a'] = a
        return a

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        inputs = self.cache['inputs']
        z = self.cache['z']
        a = self.cache['a']
        act_grad = self._activation_derivative(z, a)
        dz = gradients * act_grad

        self.gradients['weights'] = np.dot(inputs.T, dz)
        if self.use_bias:
            self.gradients['bias'] = np.sum(dz, axis=0)

        dx = np.dot(dz, self.parameters['weights'].T)
        return dx


class SpikingLIFLayer(Layer):
    """
    Capa de neuronas de disparo LIF (Leaky Integrate-and-Fire) simplificada.
    No es diferenciable completa, pero es útil para simulaciones o como
    bloque no entrenable con backprop estándar.
    """

    def __init__(self, num_neurons: int, tau: float = 20.0, v_th: float = 1.0, v_reset: float = 0.0, leak: float = 0.95, name: str = None):
        super().__init__(name)
        self.num_neurons = num_neurons
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.leak = leak
        self.state = None  # potencial de membrana

    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        self.input_shape = input_shape
        input_units = input_shape[-1]
        limit = np.sqrt(6.0 / (input_units + self.num_neurons))
        self.parameters['weights'] = np.random.uniform(-limit, limit, (input_units, self.num_neurons))
        self.parameters['bias'] = np.zeros(self.num_neurons)
        self.output_shape = (self.num_neurons,)
        self.gradients = {key: np.zeros_like(val) for key, val in self.parameters.items()}
        self.state = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Inicializar estado por lote si no existe
        batch = inputs.shape[0]
        if self.state is None or self.state.shape[0] != batch:
            self.state = np.zeros((batch, self.num_neurons))

        i_t = np.dot(inputs, self.parameters['weights']) + self.parameters['bias']
        # Dinámica LIF discretizada: v = leak * v + i_t ; spike si v >= v_th
        v_t = self.leak * self.state + i_t
        spikes = (v_t >= self.v_th).astype(np.float32)
        # Reset donde hay spike
        v_t = np.where(spikes > 0, self.v_reset, v_t)

        self.state = v_t
        self.cache['inputs'] = inputs
        self.cache['spikes'] = spikes
        return spikes

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        # Aproximación surrogate para la derivada del evento de disparo
        spikes = self.cache.get('spikes')
        inputs = self.cache.get('inputs')
        # Derivada surrogate: pequeña banda alrededor del umbral
        surrogate = 0.1 * np.ones_like(spikes)
        dz = gradients * surrogate

        self.gradients['weights'] = np.dot(inputs.T, dz)
        self.gradients['bias'] = np.sum(dz, axis=0)
        dx = np.dot(dz, self.parameters['weights'].T)
        return dx


