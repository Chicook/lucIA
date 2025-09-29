#!/usr/bin/env python3
"""
Sistema de Modelos Neuronales - LucIA
Versión: 0.6.0
Sistema para gestión y almacenamiento de modelos de redes neuronales
"""

import os
import json
import pickle
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Neural_Models')

# Importaciones para construir redes neuronales profundas
try:
    from celebro.red_neuronal.neural_network import NeuralNetwork, NetworkConfig
    from celebro.red_neuronal.layers import DenseLayer, DropoutLayer, BatchNormLayer
    # Opcional: nuevas neuronas
    try:
        from celebro.red_neuronal.neurons import NeuronLayer
    except Exception:
        NeuronLayer = None  # Fallback si no existe
except Exception as _import_exc:
    NeuralNetwork = None
    NetworkConfig = None
    DenseLayer = None
    DropoutLayer = None
    BatchNormLayer = None
    NeuronLayer = None
    logger.warning(f"No se pudieron importar componentes de red neuronal: {_import_exc}")

class ModelType(Enum):
    """Tipos de modelos neuronales"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    GENERATIVE = "generative"
    REINFORCEMENT = "reinforcement"

class ModelStatus(Enum):
    """Estados de los modelos"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    ERROR = "error"

@dataclass
class ModelMetadata:
    """Metadatos de un modelo"""
    model_id: str
    name: str
    model_type: ModelType
    status: ModelStatus
    created_at: datetime
    last_updated: datetime
    version: str
    description: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    file_path: str
    file_size: int
    checksum: str

class NeuralModelManager:
    """
    Gestor de modelos de redes neuronales
    """
    
    def __init__(self, models_dir: str = "models/neural"):
        self.models_dir = models_dir
        self.models: Dict[str, ModelMetadata] = {}
        self.metadata_file = os.path.join(models_dir, "models_metadata.json")
        
        # Crear directorio si no existe
        os.makedirs(models_dir, exist_ok=True)
        
        # Cargar metadatos existentes
        self._load_metadata()
        
        logger.info("Sistema de modelos neuronales inicializado")
    
    def save_model(self, model: Any, name: str, model_type: ModelType,
                   description: str = "", version: str = "1.0.0",
                   input_shape: Tuple[int, ...] = None,
                   output_shape: Tuple[int, ...] = None,
                   parameters: Dict[str, Any] = None,
                   performance_metrics: Dict[str, float] = None) -> str:
        """
        Guarda un modelo neuronal
        
        Args:
            model: Modelo a guardar
            name: Nombre del modelo
            model_type: Tipo de modelo
            description: Descripción del modelo
            version: Versión del modelo
            input_shape: Forma de entrada
            output_shape: Forma de salida
            parameters: Parámetros del modelo
            performance_metrics: Métricas de rendimiento
            
        Returns:
            ID del modelo guardado
        """
        try:
            # Generar ID único
            model_id = self._generate_model_id(name, version)
            
            # Crear nombre de archivo
            filename = f"{model_id}.pkl"
            file_path = os.path.join(self.models_dir, filename)
            
            # Guardar modelo
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Obtener información del archivo
            file_size = os.path.getsize(file_path)
            checksum = self._calculate_checksum(file_path)
            
            # Crear metadatos
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                model_type=model_type,
                status=ModelStatus.TRAINED,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                version=version,
                description=description,
                input_shape=input_shape or (),
                output_shape=output_shape or (),
                parameters=parameters or {},
                performance_metrics=performance_metrics or {},
                file_path=file_path,
                file_size=file_size,
                checksum=checksum
            )
            
            # Registrar modelo
            self.models[model_id] = metadata
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Modelo guardado: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Carga un modelo neuronal
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Modelo cargado o None si no existe
        """
        try:
            if model_id not in self.models:
                logger.warning(f"Modelo no encontrado: {model_id}")
                return None
            
            metadata = self.models[model_id]
            
            # Verificar que el archivo existe
            if not os.path.exists(metadata.file_path):
                logger.error(f"Archivo del modelo no encontrado: {metadata.file_path}")
                return None
            
            # Verificar checksum
            current_checksum = self._calculate_checksum(metadata.file_path)
            if current_checksum != metadata.checksum:
                logger.warning(f"Checksum del modelo {model_id} no coincide")
            
            # Cargar modelo
            with open(metadata.file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Actualizar último acceso
            metadata.last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Modelo cargado: {metadata.name} ({model_id})")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Obtiene metadatos de un modelo"""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """
        Lista modelos con filtros opcionales
        
        Args:
            model_type: Filtrar por tipo de modelo
            status: Filtrar por estado
            
        Returns:
            Lista de metadatos de modelos
        """
        filtered_models = []
        
        for metadata in self.models.values():
            if model_type and metadata.model_type != model_type:
                continue
            if status and metadata.status != status:
                continue
            filtered_models.append(metadata)
        
        # Ordenar por fecha de creación (más recientes primero)
        filtered_models.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_models
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Actualiza el estado de un modelo"""
        try:
            if model_id not in self.models:
                return False
            
            self.models[model_id].status = status
            self.models[model_id].last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Estado del modelo {model_id} actualizado a {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando estado del modelo: {e}")
            return False
    
    def update_performance_metrics(self, model_id: str, 
                                 metrics: Dict[str, float]) -> bool:
        """Actualiza las métricas de rendimiento de un modelo"""
        try:
            if model_id not in self.models:
                return False
            
            self.models[model_id].performance_metrics.update(metrics)
            self.models[model_id].last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Métricas del modelo {model_id} actualizadas")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando métricas: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Elimina un modelo"""
        try:
            if model_id not in self.models:
                return False
            
            metadata = self.models[model_id]
            
            # Eliminar archivo físico
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # Eliminar del registro
            del self.models[model_id]
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Modelo eliminado: {metadata.name} ({model_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando modelo: {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los modelos"""
        try:
            total_models = len(self.models)
            
            # Estadísticas por tipo
            type_counts = {}
            for model_type in ModelType:
                type_counts[model_type.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.model_type == model_type
                )
            
            # Estadísticas por estado
            status_counts = {}
            for status in ModelStatus:
                status_counts[status.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.status == status
                )
            
            # Tamaño total
            total_size = sum(metadata.file_size for metadata in self.models.values())
            
            # Modelos más recientes
            recent_models = sorted(
                self.models.values(),
                key=lambda x: x.last_updated,
                reverse=True
            )[:5]
            
            return {
                'total_models': total_models,
                'total_size': total_size,
                'type_distribution': type_counts,
                'status_distribution': status_counts,
                'recent_models': [
                    {
                        'id': m.model_id,
                        'name': m.name,
                        'type': m.model_type.value,
                        'status': m.status.value,
                        'last_updated': m.last_updated.isoformat()
                    }
                    for m in recent_models
                ]
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Genera un ID único para el modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{version}_{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcula el checksum de un archivo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _save_metadata(self):
        """Guarda metadatos de modelos"""
        try:
            metadata = {
                'models': {
                    model_id: {
                        'model_id': metadata.model_id,
                        'name': metadata.name,
                        'model_type': metadata.model_type.value,
                        'status': metadata.status.value,
                        'created_at': metadata.created_at.isoformat(),
                        'last_updated': metadata.last_updated.isoformat(),
                        'version': metadata.version,
                        'description': metadata.description,
                        'input_shape': metadata.input_shape,
                        'output_shape': metadata.output_shape,
                        'parameters': metadata.parameters,
                        'performance_metrics': metadata.performance_metrics,
                        'file_path': metadata.file_path,
                        'file_size': metadata.file_size,
                        'checksum': metadata.checksum
                    }
                    for model_id, metadata in self.models.items()
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _load_metadata(self):
        """Carga metadatos de modelos"""
        try:
            if not os.path.exists(self.metadata_file):
                return
            
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Cargar modelos
            for model_id, model_data in data.get('models', {}).items():
                metadata = ModelMetadata(
                    model_id=model_data['model_id'],
                    name=model_data['name'],
                    model_type=ModelType(model_data['model_type']),
                    status=ModelStatus(model_data['status']),
                    created_at=datetime.fromisoformat(model_data['created_at']),
                    last_updated=datetime.fromisoformat(model_data['last_updated']),
                    version=model_data['version'],
                    description=model_data['description'],
                    input_shape=tuple(model_data['input_shape']),
                    output_shape=tuple(model_data['output_shape']),
                    parameters=model_data['parameters'],
                    performance_metrics=model_data['performance_metrics'],
                    file_path=model_data['file_path'],
                    file_size=model_data['file_size'],
                    checksum=model_data['checksum']
                )
                
                # Solo cargar si el archivo existe
                if os.path.exists(metadata.file_path):
                    self.models[model_id] = metadata
            
            logger.info(f"Cargados {len(self.models)} modelos neuronales")
            
        except Exception as e:
            logger.error(f"Error cargando metadatos: {e}")

# Instancia global del gestor de modelos neuronales
neural_model_manager = NeuralModelManager()


# ===== FÁBRICAS DE MODELOS PROFUNDOS =====
def build_deep_dense_network(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = None,
    activation: str = 'relu',
    output_activation: str = 'softmax',
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True,
    optimizer: str = 'adam',
    loss: str = 'categorical_crossentropy'
):
    """
    Construye una red profunda tipo MLP con múltiples capas densas, similar
    a la arquitectura de la imagen (varias capas ocultas totalmente conectadas).
    """
    if NeuralNetwork is None or NetworkConfig is None:
        raise RuntimeError("Componentes de red neuronal no disponibles")

    hidden_layers = hidden_layers or [512, 512, 256, 256, 128]

    config = NetworkConfig(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        activation=activation,
        output_activation=output_activation,
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        validation_split=0.15,
        early_stopping=True,
        patience=8,
        regularization='l2',
        regularization_rate=0.0005,
        dropout_rate=dropout_rate,
        batch_normalization=use_batch_norm,
        optimizer=optimizer,
        loss_function=loss
    )

    net = NeuralNetwork(config)

    # Capa de entrada -> primera oculta
    first_units = hidden_layers[0]
    net.add_layer(DenseLayer(first_units, activation=activation))
    if use_batch_norm:
        net.add_layer(BatchNormLayer())
    if dropout_rate and dropout_rate > 0:
        net.add_layer(DropoutLayer(dropout_rate))

    # Capas ocultas adicionales
    for units in hidden_layers[1:]:
        net.add_layer(DenseLayer(units, activation=activation))
        if use_batch_norm:
            net.add_layer(BatchNormLayer())
        if dropout_rate and dropout_rate > 0:
            net.add_layer(DropoutLayer(dropout_rate))

    # Capa de salida
    out_activation = output_activation if output_activation else 'linear'
    net.add_layer(DenseLayer(output_size, activation=out_activation))

    # Construir y compilar
    net.build()
    net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return net


def build_neuron_layer_network(
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = None,
    activation: str = 'relu',
    output_activation: str = 'softmax',
):
    """
    Variante que utiliza `NeuronLayer` si está disponible, para exponer semántica
    de neuronas. Fallback a DenseLayer si no está disponible.
    """
    if NeuralNetwork is None or NetworkConfig is None:
        raise RuntimeError("Componentes de red neuronal no disponibles")

    hidden_layers = hidden_layers or [256, 256, 128, 128]
    config = NetworkConfig(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        activation=activation,
        output_activation=output_activation,
        learning_rate=0.001
    )
    net = NeuralNetwork(config)

    LayerClass = NeuronLayer if NeuronLayer is not None else DenseLayer
    for units in hidden_layers:
        net.add_layer(LayerClass(units, activation=activation))
    net.add_layer(LayerClass(output_size, activation=output_activation))

    net.build().compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return net


def create_and_register_deep_model(
    name: str,
    input_size: int,
    output_size: int,
    hidden_layers: List[int] = None,
    save: bool = True,
    description: str = "Deep MLP auto-generado"
) -> str:
    """Crea un modelo profundo y lo registra en el gestor, devolviendo su ID."""
    model = build_deep_dense_network(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers or [512, 512, 256, 256, 128]
    )

    if save:
        model_id = neural_model_manager.save_model(
            model=model,
            name=name,
            model_type=ModelType.CLASSIFICATION,
            description=description,
            version="1.0.0",
            input_shape=(input_size,),
            output_shape=(output_size,),
            parameters={
                'hidden_layers': hidden_layers or [512, 512, 256, 256, 128],
                'activation': model.config.activation,
                'output_activation': model.config.output_activation
            },
            performance_metrics={}
        )
        return model_id
    return ""


# ===== UTILIDADES Y PRESETS AÚN MÁS PROFUNDOS =====
def generate_constant_layers(depth: int, width: int) -> List[int]:
    """Genera una lista de 'depth' capas ocultas con 'width' neuronas cada una."""
    return [int(width) for _ in range(max(1, depth))]


def generate_pyramid_layers(depth: int, top_width: int, decay: float = 0.5) -> List[int]:
    """Genera capas en pirámide: anchas al inicio y decreciendo por 'decay'."""
    layers = []
    current = float(top_width)
    for _ in range(max(1, depth)):
        layers.append(max(8, int(round(current))))
        current *= max(0.1, min(decay, 0.95))
    return layers


def build_very_deep_dense_network(
    input_size: int,
    output_size: int,
    depth: int = 10,
    width: int = 1024,
    mode: str = 'constant',  # 'constant' | 'pyramid'
    activation: str = 'relu',
    output_activation: str = 'softmax',
    dropout_rate: float = 0.3,
    use_batch_norm: bool = True,
    optimizer: str = 'adam',
    loss: str = 'categorical_crossentropy'
):
    """
    Construye una red MUY profunda (10+ capas) para parecerse a la imagen
    con muchas neuronas y múltiples capas totalmente conectadas.
    """
    if mode == 'pyramid':
        hidden_layers = generate_pyramid_layers(depth=depth, top_width=width, decay=0.7)
    else:
        hidden_layers = generate_constant_layers(depth=depth, width=width)

    return build_deep_dense_network(
        input_size=input_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        activation=activation,
        output_activation=output_activation,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        optimizer=optimizer,
        loss=loss
    )


def create_and_register_very_deep_model(
    name: str,
    input_size: int,
    output_size: int,
    depth: int = 12,
    width: int = 1024,
    mode: str = 'constant',
    description: str = "Very Deep MLP (parecido a imagen)"
) -> str:
    """Crea y registra un modelo muy profundo (12 capas x 1024 neuronas por defecto)."""
    model = build_very_deep_dense_network(
        input_size=input_size,
        output_size=output_size,
        depth=depth,
        width=width,
        mode=mode
    )

    model_id = neural_model_manager.save_model(
        model=model,
        name=name,
        model_type=ModelType.CLASSIFICATION,
        description=description,
        version="1.0.0",
        input_shape=(input_size,),
        output_shape=(output_size,),
        parameters={
            'depth': depth,
            'width': width,
            'mode': mode,
            'activation': model.config.activation,
            'output_activation': model.config.output_activation
        },
        performance_metrics={}
    )
    return model_id
