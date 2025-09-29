"""
Módulos neuronales componibles para construir redes profundas por capas.
"""

from .layer_module import LayerModule, LayerSpec
from .builder import DeepNetworkModulesBuilder, generate_constant_specs, generate_pyramid_specs

__all__ = [
    "LayerModule",
    "LayerSpec",
    "DeepNetworkModulesBuilder",
    "generate_constant_specs",
    "generate_pyramid_specs",
]


