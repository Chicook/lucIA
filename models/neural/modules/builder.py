from typing import List

from .layer_module import LayerModule, LayerSpec


def generate_constant_specs(depth: int, width: int, activation: str = 'relu', dropout: float = 0.2, batch_norm: bool = True) -> List[LayerSpec]:
    return [LayerSpec(units=width, activation=activation, dropout=dropout, batch_norm=batch_norm, name=f"hidden_{i+1}") for i in range(max(1, depth))]


def generate_pyramid_specs(depth: int, top_width: int, activation: str = 'relu', dropout: float = 0.2, batch_norm: bool = True, decay: float = 0.7) -> List[LayerSpec]:
    specs: List[LayerSpec] = []
    current = float(top_width)
    for i in range(max(1, depth)):
        specs.append(LayerSpec(units=max(8, int(round(current))), activation=activation, dropout=dropout, batch_norm=batch_norm, name=f"hidden_{i+1}"))
        current *= max(0.1, min(decay, 0.95))
    return specs


class DeepNetworkModulesBuilder:
    """
    Ensambla una red profunda a partir de módulos de capa, reflejando la imagen
    con múltiples capas densas entre la entrada y la salida.
    """

    def __init__(self, input_size: int, output_size: int, specs: List[LayerSpec], output_activation: str = 'softmax') -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.specs = specs
        self.output_activation = output_activation

    def build(self):
        from celebro.red_neuronal.neural_network import NeuralNetwork, NetworkConfig
        from celebro.red_neuronal.layers import DenseLayer

        config = NetworkConfig(
            input_size=self.input_size,
            hidden_layers=[s.units for s in self.specs],
            output_size=self.output_size,
        )
        net = NeuralNetwork(config)

        for spec in self.specs:
            LayerModule(spec).build_into(net)

        net.add_layer(DenseLayer(self.output_size, activation=self.output_activation))
        net.build().compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return net

    def create_and_register(self, name: str, description: str = "Modular Deep Network") -> str:
        """Construye la red y la guarda usando el gestor de modelos neuronales."""
        from models.neural.neural_models import neural_model_manager, ModelType

        net = self.build()
        model_id = neural_model_manager.save_model(
            model=net,
            name=name,
            model_type=ModelType.CLASSIFICATION,
            description=description,
            version="1.0.0",
            input_shape=(self.input_size,),
            output_shape=(self.output_size,),
            parameters={
                'specs': [s.__dict__ for s in self.specs],
                'output_activation': self.output_activation
            },
            performance_metrics={}
        )
        return model_id


