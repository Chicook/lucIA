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
