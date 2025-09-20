#!/usr/bin/env python3
"""
Sistema de Gestión de Datos de Aprendizaje - LucIA
Versión: 0.6.0
Sistema para gestión de datos de aprendizaje y entrenamiento
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib
import sqlite3

logger = logging.getLogger('Learning_Data_Manager')

class DataType(Enum):
    """Tipos de datos de aprendizaje"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    RAW = "raw"
    PROCESSED = "processed"
    FEATURES = "features"
    LABELS = "labels"

class DataFormat(Enum):
    """Formatos de datos"""
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    NUMPY = "numpy"
    PANDAS = "pandas"
    SQLITE = "sqlite"

@dataclass
class LearningDataset:
    """Dataset de aprendizaje"""
    dataset_id: str
    name: str
    data_type: DataType
    format: DataFormat
    created_at: datetime
    last_updated: datetime
    description: str
    file_path: str
    file_size: int
    rows: int
    columns: int
    features: List[str]
    target_column: Optional[str]
    metadata: Dict[str, Any]
    checksum: str

class LearningDataManager:
    """
    Gestor de datos de aprendizaje
    """
    
    def __init__(self, data_dir: str = "data/learning"):
        self.data_dir = data_dir
        self.datasets: Dict[str, LearningDataset] = {}
        self.metadata_file = os.path.join(data_dir, "learning_datasets_metadata.json")
        self.db_file = os.path.join(data_dir, "learning_data.db")
        
        # Crear directorio si no existe
        os.makedirs(data_dir, exist_ok=True)
        
        # Inicializar base de datos
        self._init_database()
        
        # Cargar metadatos existentes
        self._load_metadata()
        
        logger.info("Sistema de gestión de datos de aprendizaje inicializado")
    
    def _init_database(self):
        """Inicializa la base de datos SQLite"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Crear tabla de datasets
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    rows INTEGER,
                    columns INTEGER,
                    features TEXT,
                    target_column TEXT,
                    metadata TEXT,
                    checksum TEXT
                )
            ''')
            
            # Crear tabla de versiones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    version_number TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    changes TEXT,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
    
    def save_dataset(self, data: Union[pd.DataFrame, np.ndarray, List[Dict], str],
                    name: str, data_type: DataType, description: str = "",
                    features: List[str] = None, target_column: str = None,
                    metadata: Dict[str, Any] = None) -> str:
        """
        Guarda un dataset de aprendizaje
        
        Args:
            data: Datos a guardar
            name: Nombre del dataset
            data_type: Tipo de datos
            description: Descripción del dataset
            features: Lista de características
            target_column: Columna objetivo
            metadata: Metadatos adicionales
            
        Returns:
            ID del dataset guardado
        """
        try:
            # Generar ID único
            dataset_id = self._generate_dataset_id(name)
            
            # Determinar formato y procesar datos
            data_format, processed_data, rows, columns = self._process_data(data)
            
            # Crear nombre de archivo
            filename = f"{dataset_id}.{data_format.value}"
            file_path = os.path.join(self.data_dir, filename)
            
            # Guardar datos
            self._save_data_to_file(processed_data, file_path, data_format)
            
            # Obtener información del archivo
            file_size = os.path.getsize(file_path)
            checksum = self._calculate_checksum(file_path)
            
            # Crear metadatos
            dataset = LearningDataset(
                dataset_id=dataset_id,
                name=name,
                data_type=data_type,
                format=data_format,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                description=description,
                file_path=file_path,
                file_size=file_size,
                rows=rows,
                columns=columns,
                features=features or [],
                target_column=target_column,
                metadata=metadata or {},
                checksum=checksum
            )
            
            # Registrar dataset
            self.datasets[dataset_id] = dataset
            
            # Guardar en base de datos
            self._save_to_database(dataset)
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Dataset guardado: {name} ({dataset_id})")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error guardando dataset: {e}")
            raise
    
    def load_dataset(self, dataset_id: str) -> Optional[Union[pd.DataFrame, np.ndarray, List[Dict]]]:
        """
        Carga un dataset de aprendizaje
        
        Args:
            dataset_id: ID del dataset
            
        Returns:
            Dataset cargado o None si no existe
        """
        try:
            if dataset_id not in self.datasets:
                logger.warning(f"Dataset no encontrado: {dataset_id}")
                return None
            
            dataset = self.datasets[dataset_id]
            
            # Verificar que el archivo existe
            if not os.path.exists(dataset.file_path):
                logger.error(f"Archivo del dataset no encontrado: {dataset.file_path}")
                return None
            
            # Verificar checksum
            current_checksum = self._calculate_checksum(dataset.file_path)
            if current_checksum != dataset.checksum:
                logger.warning(f"Checksum del dataset {dataset_id} no coincide")
            
            # Cargar datos
            data = self._load_data_from_file(dataset.file_path, dataset.format)
            
            # Actualizar último acceso
            dataset.last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Dataset cargado: {dataset.name} ({dataset_id})")
            return data
            
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            return None
    
    def get_dataset_info(self, dataset_id: str) -> Optional[LearningDataset]:
        """Obtiene información de un dataset"""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self, data_type: Optional[DataType] = None,
                     format: Optional[DataFormat] = None) -> List[LearningDataset]:
        """
        Lista datasets con filtros opcionales
        
        Args:
            data_type: Filtrar por tipo de datos
            format: Filtrar por formato
            
        Returns:
            Lista de datasets
        """
        filtered_datasets = []
        
        for dataset in self.datasets.values():
            if data_type and dataset.data_type != data_type:
                continue
            if format and dataset.format != format:
                continue
            filtered_datasets.append(dataset)
        
        # Ordenar por fecha de creación (más recientes primero)
        filtered_datasets.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_datasets
    
    def split_dataset(self, dataset_id: str, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, str]:
        """
        Divide un dataset en conjuntos de entrenamiento, validación y prueba
        
        Args:
            dataset_id: ID del dataset
            train_ratio: Proporción de entrenamiento
            val_ratio: Proporción de validación
            test_ratio: Proporción de prueba
            
        Returns:
            Diccionario con IDs de los datasets divididos
        """
        try:
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("Las proporciones deben sumar 1.0")
            
            # Cargar dataset original
            data = self.load_dataset(dataset_id)
            if data is None:
                raise ValueError("Dataset no encontrado")
            
            original_dataset = self.datasets[dataset_id]
            
            # Convertir a DataFrame si es necesario
            if isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Mezclar datos
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calcular índices de división
            n_total = len(df)
            train_end = int(n_total * train_ratio)
            val_end = train_end + int(n_total * val_ratio)
            
            # Dividir datos
            train_data = df[:train_end]
            val_data = df[train_end:val_end]
            test_data = df[val_end:]
            
            # Guardar datasets divididos
            train_id = self.save_dataset(
                train_data, 
                f"{original_dataset.name}_train",
                DataType.TRAINING,
                f"Conjunto de entrenamiento de {original_dataset.name}",
                original_dataset.features,
                original_dataset.target_column,
                {**original_dataset.metadata, 'split_from': dataset_id}
            )
            
            val_id = self.save_dataset(
                val_data,
                f"{original_dataset.name}_val", 
                DataType.VALIDATION,
                f"Conjunto de validación de {original_dataset.name}",
                original_dataset.features,
                original_dataset.target_column,
                {**original_dataset.metadata, 'split_from': dataset_id}
            )
            
            test_id = self.save_dataset(
                test_data,
                f"{original_dataset.name}_test",
                DataType.TEST,
                f"Conjunto de prueba de {original_dataset.name}",
                original_dataset.features,
                original_dataset.target_column,
                {**original_dataset.metadata, 'split_from': dataset_id}
            )
            
            logger.info(f"Dataset dividido: {dataset_id} -> {train_id}, {val_id}, {test_id}")
            
            return {
                'train_id': train_id,
                'validation_id': val_id,
                'test_id': test_id
            }
            
        except Exception as e:
            logger.error(f"Error dividiendo dataset: {e}")
            raise
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los datasets"""
        try:
            total_datasets = len(self.datasets)
            
            # Estadísticas por tipo
            type_counts = {}
            for data_type in DataType:
                type_counts[data_type.value] = sum(
                    1 for dataset in self.datasets.values()
                    if dataset.data_type == data_type
                )
            
            # Estadísticas por formato
            format_counts = {}
            for data_format in DataFormat:
                format_counts[data_format.value] = sum(
                    1 for dataset in self.datasets.values()
                    if dataset.format == data_format
                )
            
            # Tamaño total
            total_size = sum(dataset.file_size for dataset in self.datasets.values())
            
            # Total de filas
            total_rows = sum(dataset.rows for dataset in self.datasets.values())
            
            return {
                'total_datasets': total_datasets,
                'total_size': total_size,
                'total_rows': total_rows,
                'type_distribution': type_counts,
                'format_distribution': format_counts,
                'average_rows_per_dataset': total_rows / total_datasets if total_datasets > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _process_data(self, data: Union[pd.DataFrame, np.ndarray, List[Dict], str]) -> Tuple[DataFormat, Any, int, int]:
        """Procesa datos y determina el formato"""
        if isinstance(data, pd.DataFrame):
            return DataFormat.PANDAS, data, len(data), len(data.columns)
        elif isinstance(data, np.ndarray):
            return DataFormat.NUMPY, data, data.shape[0], data.shape[1] if len(data.shape) > 1 else 1
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return DataFormat.JSON, data, len(data), len(data[0]) if data else 0
            else:
                return DataFormat.PICKLE, data, len(data), 1
        elif isinstance(data, str):
            return DataFormat.CSV, data, 0, 0
        else:
            return DataFormat.PICKLE, data, 0, 0
    
    def _save_data_to_file(self, data: Any, file_path: str, data_format: DataFormat):
        """Guarda datos en un archivo según el formato"""
        if data_format == DataFormat.CSV:
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                with open(file_path, 'w') as f:
                    f.write(data)
        elif data_format == DataFormat.JSON:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif data_format == DataFormat.PICKLE:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif data_format == DataFormat.NUMPY:
            np.save(file_path, data)
        elif data_format == DataFormat.PANDAS:
            data.to_pickle(file_path)
    
    def _load_data_from_file(self, file_path: str, data_format: DataFormat) -> Any:
        """Carga datos de un archivo según el formato"""
        if data_format == DataFormat.CSV:
            return pd.read_csv(file_path)
        elif data_format == DataFormat.JSON:
            with open(file_path, 'r') as f:
                return json.load(f)
        elif data_format == DataFormat.PICKLE:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif data_format == DataFormat.NUMPY:
            return np.load(file_path)
        elif data_format == DataFormat.PANDAS:
            return pd.read_pickle(file_path)
    
    def _generate_dataset_id(self, name: str) -> str:
        """Genera un ID único para el dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcula el checksum de un archivo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _save_to_database(self, dataset: LearningDataset):
        """Guarda dataset en la base de datos"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset.dataset_id,
                dataset.name,
                dataset.data_type.value,
                dataset.format.value,
                dataset.created_at.isoformat(),
                dataset.last_updated.isoformat(),
                dataset.description,
                dataset.file_path,
                dataset.file_size,
                dataset.rows,
                dataset.columns,
                json.dumps(dataset.features),
                dataset.target_column,
                json.dumps(dataset.metadata),
                dataset.checksum
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error guardando en base de datos: {e}")
    
    def _save_metadata(self):
        """Guarda metadatos de datasets"""
        try:
            metadata = {
                'datasets': {
                    dataset_id: {
                        'dataset_id': dataset.dataset_id,
                        'name': dataset.name,
                        'data_type': dataset.data_type.value,
                        'format': dataset.format.value,
                        'created_at': dataset.created_at.isoformat(),
                        'last_updated': dataset.last_updated.isoformat(),
                        'description': dataset.description,
                        'file_path': dataset.file_path,
                        'file_size': dataset.file_size,
                        'rows': dataset.rows,
                        'columns': dataset.columns,
                        'features': dataset.features,
                        'target_column': dataset.target_column,
                        'metadata': dataset.metadata,
                        'checksum': dataset.checksum
                    }
                    for dataset_id, dataset in self.datasets.items()
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _load_metadata(self):
        """Carga metadatos de datasets"""
        try:
            if not os.path.exists(self.metadata_file):
                return
            
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Cargar datasets
            for dataset_id, dataset_data in data.get('datasets', {}).items():
                dataset = LearningDataset(
                    dataset_id=dataset_data['dataset_id'],
                    name=dataset_data['name'],
                    data_type=DataType(dataset_data['data_type']),
                    format=DataFormat(dataset_data['format']),
                    created_at=datetime.fromisoformat(dataset_data['created_at']),
                    last_updated=datetime.fromisoformat(dataset_data['last_updated']),
                    description=dataset_data['description'],
                    file_path=dataset_data['file_path'],
                    file_size=dataset_data['file_size'],
                    rows=dataset_data['rows'],
                    columns=dataset_data['columns'],
                    features=dataset_data['features'],
                    target_column=dataset_data['target_column'],
                    metadata=dataset_data['metadata'],
                    checksum=dataset_data['checksum']
                )
                
                # Solo cargar si el archivo existe
                if os.path.exists(dataset.file_path):
                    self.datasets[dataset_id] = dataset
            
            logger.info(f"Cargados {len(self.datasets)} datasets de aprendizaje")
            
        except Exception as e:
            logger.error(f"Error cargando metadatos: {e}")

# Instancia global del gestor de datos de aprendizaje
learning_data_manager = LearningDataManager()
