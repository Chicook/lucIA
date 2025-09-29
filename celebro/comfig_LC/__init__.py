"""
@red_neuronal - Sistema de Redes Neuronales Completo
Versión: 0.6.0
Sistema modular de redes neuronales para @celebro
"""

from .neural_network import NeuralNetwork
from .layers import DenseLayer, ConvLayer, PoolingLayer, DropoutLayer, BatchNormLayer
from .activations import ActivationFunction, ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU
from .optimizers import Optimizer, SGD, Adam, RMSprop, Adagrad, AdamW
from .loss_functions import LossFunction, MSE, CrossEntropy, BinaryCrossEntropy, Huber
from .neural_core import NeuralCore
from .neurons import NeuronLayer, SpikingLIFLayer
from .training import Trainer
from .gemini_integration import GeminiIntegration, analyze_network, suggest_architecture, explain_results

__version__ = "0.6.0"
__author__ = "LucIA Development Team"

__all__ = [
    "NeuralNetwork",
    "DenseLayer", "ConvLayer", "PoolingLayer", "DropoutLayer", "BatchNormLayer",
    "ActivationFunction", "ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "ELU",
    "Optimizer", "SGD", "Adam", "RMSprop", "Adagrad", "AdamW",
    "LossFunction", "MSE", "CrossEntropy", "BinaryCrossEntropy", "Huber",
    "NeuralCore", "Trainer",
    "NeuronLayer", "SpikingLIFLayer",
    "GeminiIntegration", "analyze_network", "suggest_architecture", "explain_results"
]
