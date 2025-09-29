from dataclasses import dataclass
from typing import Optional


@dataclass
class LayerSpec:
    """Especificación simple de una capa neuronal densa."""
    units: int
    activation: str = 'relu'
    dropout: float = 0.0
    batch_norm: bool = True
    name: Optional[str] = None


@dataclass
class LayerModule:
    """Módulo/cápsula de una capa, para ensamblar redes profundas por etapas."""
    spec: LayerSpec

    def build_into(self, network) -> None:
        """Inserta las capas correspondientes en una instancia de NeuralNetwork."""
        from celebro.red_neuronal.layers import DenseLayer, DropoutLayer, BatchNormLayer

        network.add_layer(DenseLayer(self.spec.units, activation=self.spec.activation))
        if self.spec.batch_norm:
            network.add_layer(BatchNormLayer())
        if self.spec.dropout and self.spec.dropout > 0:
            network.add_layer(DropoutLayer(self.spec.dropout))


