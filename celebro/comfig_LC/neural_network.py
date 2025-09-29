"""
Red Neuronal Principal
Versión: 0.6.0
Implementación completa de red neuronal con arquitectura modular
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import json
from datetime import datetime

logger = logging.getLogger('Neural_Network')

@dataclass
class NetworkConfig:
    """Configuración de la red neuronal"""
    input_size: int
    hidden_layers: List[int]
    output_size: int
    activation: str = 'relu'
    output_activation: str = 'softmax'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    regularization: str = 'l2'
    regularization_rate: float = 0.01
    dropout_rate: float = 0.2
    batch_normalization: bool = True
    optimizer: str = 'adam'
    loss_function: str = 'categorical_crossentropy'

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
    
    def _get_activation_function(self, activation: str):
        """Obtiene la función de activación"""
        activations = {
            'linear': lambda x: x,
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': lambda x: np.tanh(x),
            'softmax': lambda x: self._softmax(x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1)
        }
        return activations.get(activation, activations['linear'])
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Implementación de softmax numéricamente estable"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
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
        else:  # zeros
            self.parameters['kernel'] = np.zeros((input_units, self.units))
        
        # Inicializar sesgos
        if self.use_bias:
            if self.bias_initializer == 'zeros':
                self.parameters['bias'] = np.zeros(self.units)
            else:  # ones
                self.parameters['bias'] = np.ones(self.units)
        
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
        if self.activation == 'relu':
            activation_grad = np.where(linear_output > 0, 1, 0)
        elif self.activation == 'sigmoid':
            sigmoid_output = self.cache['output']
            activation_grad = sigmoid_output * (1 - sigmoid_output)
        elif self.activation == 'tanh':
            tanh_output = self.cache['output']
            activation_grad = 1 - tanh_output ** 2
        elif self.activation == 'softmax':
            softmax_output = self.cache['output']
            activation_grad = softmax_output * (1 - softmax_output)
        elif self.activation == 'leaky_relu':
            activation_grad = np.where(linear_output > 0, 1, 0.01)
        elif self.activation == 'elu':
            activation_grad = np.where(linear_output > 0, 1, np.exp(linear_output))
        else:  # linear
            activation_grad = 1
        
        # Gradiente de la salida
        linear_gradients = gradients * activation_grad
        
        # Gradientes de los parámetros
        self.gradients['kernel'] = np.dot(inputs.T, linear_gradients)
        if self.use_bias:
            self.gradients['bias'] = np.sum(linear_gradients, axis=0)
        
        # Gradiente de la entrada
        input_gradients = np.dot(linear_gradients, self.parameters['kernel'].T)
        
        return input_gradients

class DropoutLayer(Layer):
    """Capa de dropout para regularización"""
    
    def __init__(self, rate: float = 0.5, name: str = None):
        super().__init__(name)
        self.rate = rate
        self.scale = 1.0 / (1.0 - rate)
    
    def initialize_parameters(self, input_shape: Tuple[int, ...]) -> None:
        """Inicializa los parámetros de la capa"""
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.parameters = {}
        self.gradients = {}
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        if self.is_training:
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
        if self.is_training:
            # Aplicar la misma máscara que en forward
            mask = self.cache['mask']
            input_gradients = gradients * mask * self.scale
        else:
            input_gradients = gradients
        
        return input_gradients

class BatchNormalizationLayer(Layer):
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

class NeuralNetwork:
    """Red neuronal principal con arquitectura modular"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.layers: List[Layer] = []
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.is_compiled = False
        self.optimizer = None
        self.loss_function = None
        
        logger.info(f"Red neuronal creada con configuración: {config}")
    
    def add_layer(self, layer: Layer) -> 'NeuralNetwork':
        """Agrega una capa a la red"""
        self.layers.append(layer)
        logger.info(f"Capa agregada: {layer.name}")
        return self
    
    def build(self) -> 'NeuralNetwork':
        """Construye la arquitectura de la red"""
        try:
            # Inicializar primera capa
            current_shape = (self.config.input_size,)
            
            for i, layer in enumerate(self.layers):
                layer.initialize_parameters(current_shape)
                current_shape = layer.output_shape
                logger.info(f"Capa {i+1} inicializada: {layer.name} - Forma: {current_shape}")
            
            # Verificar que la última capa tenga el tamaño correcto
            if current_shape[-1] != self.config.output_size:
                raise ValueError(f"El tamaño de salida de la última capa ({current_shape[-1]}) "
                               f"no coincide con el tamaño esperado ({self.config.output_size})")
            
            logger.info("Arquitectura de red construida exitosamente")
            return self
            
        except Exception as e:
            logger.error(f"Error construyendo la red: {e}")
            raise
    
    def compile(self, optimizer: str = 'adam', loss: str = 'categorical_crossentropy', 
                metrics: List[str] = None) -> 'NeuralNetwork':
        """Compila la red con optimizador y función de pérdida"""
        try:
            # Configurar optimizador
            self.optimizer = self._get_optimizer(optimizer)
            
            # Configurar función de pérdida
            self.loss_function = self._get_loss_function(loss)
            
            # Configurar métricas
            self.metrics = metrics or ['accuracy']
            
            self.is_compiled = True
            logger.info(f"Red compilada con optimizador: {optimizer}, pérdida: {loss}")
            return self
            
        except Exception as e:
            logger.error(f"Error compilando la red: {e}")
            raise
    
    def _get_optimizer(self, optimizer_name: str):
        """Obtiene el optimizador especificado"""
        optimizers = {
            'sgd': SGD(learning_rate=self.config.learning_rate),
            'adam': Adam(learning_rate=self.config.learning_rate),
            'rmsprop': RMSprop(learning_rate=self.config.learning_rate),
            'adagrad': Adagrad(learning_rate=self.config.learning_rate),
            'adamw': AdamW(learning_rate=self.config.learning_rate)
        }
        return optimizers.get(optimizer_name, optimizers['adam'])
    
    def _get_loss_function(self, loss_name: str):
        """Obtiene la función de pérdida especificada"""
        loss_functions = {
            'mse': MSE(),
            'categorical_crossentropy': CrossEntropy(),
            'binary_crossentropy': BinaryCrossEntropy(),
            'huber': Huber()
        }
        return loss_functions.get(loss_name, loss_functions['categorical_crossentropy'])
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagación hacia adelante"""
        if not self.is_compiled:
            raise ValueError("La red debe ser compilada antes de usar")
        
        current_input = inputs
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        return current_input
    
    def backward(self, gradients: np.ndarray) -> None:
        """Propagación hacia atrás"""
        current_gradients = gradients
        
        for layer in reversed(self.layers):
            current_gradients = layer.backward(current_gradients)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predicción en modo inferencia"""
        # Establecer todas las capas en modo inferencia
        for layer in self.layers:
            layer.set_training(False)
        
        predictions = self.forward(inputs)
        
        # Restaurar modo de entrenamiento
        for layer in self.layers:
            layer.set_training(True)
        
        return predictions
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evalúa la red en datos de prueba"""
        predictions = self.predict(x_test)
        loss = self.loss_function.compute(y_test, predictions)
        
        # Calcular precisión
        if self.config.output_activation == 'softmax':
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
        else:
            # Para regresión
            accuracy = 1.0 - (loss / np.var(y_test))
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
    
    def save_model(self, filepath: str) -> None:
        """Guarda el modelo en un archivo"""
        try:
            model_data = {
                'config': self.config,
                'layers': [],
                'history': self.history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar parámetros de cada capa
            for layer in self.layers:
                layer_data = {
                    'name': layer.name,
                    'class': layer.__class__.__name__,
                    'parameters': layer.get_parameters(),
                    'input_shape': layer.input_shape,
                    'output_shape': layer.output_shape
                }
                model_data['layers'].append(layer_data)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise
    
    def load_model(self, filepath: str) -> 'NeuralNetwork':
        """Carga un modelo desde un archivo"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restaurar configuración
            self.config = model_data['config']
            self.history = model_data['history']
            
            # Reconstruir capas
            self.layers = []
            for layer_data in model_data['layers']:
                # Crear instancia de capa (simplificado)
                if layer_data['class'] == 'DenseLayer':
                    layer = DenseLayer(
                        units=layer_data['output_shape'][0],
                        name=layer_data['name']
                    )
                elif layer_data['class'] == 'DropoutLayer':
                    layer = DropoutLayer(name=layer_data['name'])
                elif layer_data['class'] == 'BatchNormalizationLayer':
                    layer = BatchNormalizationLayer(name=layer_data['name'])
                else:
                    continue
                
                # Restaurar parámetros
                layer.input_shape = layer_data['input_shape']
                layer.output_shape = layer_data['output_shape']
                layer.set_parameters(layer_data['parameters'])
                
                self.layers.append(layer)
            
            logger.info(f"Modelo cargado desde: {filepath}")
            return self
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def get_summary(self) -> str:
        """Obtiene un resumen de la arquitectura de la red"""
        summary = "=" * 60 + "\n"
        summary += "RESUMEN DE LA RED NEURONAL\n"
        summary += "=" * 60 + "\n"
        summary += f"Configuración: {self.config}\n"
        summary += f"Capas: {len(self.layers)}\n"
        summary += f"Compilada: {self.is_compiled}\n"
        summary += "-" * 60 + "\n"
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = sum(param.size for param in layer.get_parameters().values())
            total_params += layer_params
            
            summary += f"Capa {i+1}: {layer.name}\n"
            summary += f"  Entrada: {layer.input_shape}\n"
            summary += f"  Salida: {layer.output_shape}\n"
            summary += f"  Parámetros: {layer_params:,}\n"
            summary += "-" * 60 + "\n"
        
        summary += f"Total de parámetros: {total_params:,}\n"
        summary += "=" * 60
        
        return summary

# Importar clases necesarias (se definirán en otros archivos)
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

class Adam:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

class RMSprop:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

class Adagrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

class AdamW:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

class MSE:
    def compute(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

class CrossEntropy:
    def compute(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

class BinaryCrossEntropy:
    def compute(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

class Huber:
    def compute(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        return np.mean(np.where(np.abs(error) <= delta, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta)))
