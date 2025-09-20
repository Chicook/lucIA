#!/usr/bin/env python3
"""
Sistema de Modelos de Optimización - LucIA
Versión: 0.6.0
Sistema para gestión de modelos de optimización y algoritmos genéticos
"""

import os
import json
import pickle
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Optimization_Models')

class OptimizationType(Enum):
    """Tipos de modelos de optimización"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"
    LINEAR_PROGRAMMING = "linear_programming"
    INTEGER_PROGRAMMING = "integer_programming"
    CONSTRAINT_OPTIMIZATION = "constraint_optimization"

class OptimizationStatus(Enum):
    """Estados de los modelos de optimización"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    STOPPED = "stopped"
    ERROR = "error"
    ARCHIVED = "archived"

@dataclass
class OptimizationResult:
    """Resultado de una optimización"""
    best_solution: Any
    best_fitness: float
    convergence_history: List[float]
    iterations: int
    execution_time: float
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class OptimizationModelMetadata:
    """Metadatos de un modelo de optimización"""
    model_id: str
    name: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    created_at: datetime
    last_updated: datetime
    version: str
    description: str
    objective_function: str
    constraints: List[str]
    parameters: Dict[str, Any]
    best_result: Optional[OptimizationResult]
    performance_metrics: Dict[str, float]
    file_path: str
    file_size: int
    checksum: str

class OptimizationModelManager:
    """
    Gestor de modelos de optimización
    """
    
    def __init__(self, models_dir: str = "models/optimization"):
        self.models_dir = models_dir
        self.models: Dict[str, OptimizationModelMetadata] = {}
        self.metadata_file = os.path.join(models_dir, "optimization_models_metadata.json")
        
        # Crear directorio si no existe
        os.makedirs(models_dir, exist_ok=True)
        
        # Cargar metadatos existentes
        self._load_metadata()
        
        logger.info("Sistema de modelos de optimización inicializado")
    
    def save_optimization_model(self, model: Any, name: str, optimization_type: OptimizationType,
                               description: str = "", version: str = "1.0.0",
                               objective_function: str = "",
                               constraints: List[str] = None,
                               parameters: Dict[str, Any] = None,
                               best_result: Optional[OptimizationResult] = None,
                               performance_metrics: Dict[str, float] = None) -> str:
        """
        Guarda un modelo de optimización
        
        Args:
            model: Modelo a guardar
            name: Nombre del modelo
            optimization_type: Tipo de optimización
            description: Descripción del modelo
            version: Versión del modelo
            objective_function: Función objetivo
            constraints: Restricciones
            parameters: Parámetros del modelo
            best_result: Mejor resultado obtenido
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
            metadata = OptimizationModelMetadata(
                model_id=model_id,
                name=name,
                optimization_type=optimization_type,
                status=OptimizationStatus.STOPPED,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                version=version,
                description=description,
                objective_function=objective_function,
                constraints=constraints or [],
                parameters=parameters or {},
                best_result=best_result,
                performance_metrics=performance_metrics or {},
                file_path=file_path,
                file_size=file_size,
                checksum=checksum
            )
            
            # Registrar modelo
            self.models[model_id] = metadata
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Modelo de optimización guardado: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error guardando modelo de optimización: {e}")
            raise
    
    def load_optimization_model(self, model_id: str) -> Optional[Any]:
        """
        Carga un modelo de optimización
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Modelo cargado o None si no existe
        """
        try:
            if model_id not in self.models:
                logger.warning(f"Modelo de optimización no encontrado: {model_id}")
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
            
            logger.info(f"Modelo de optimización cargado: {metadata.name} ({model_id})")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo de optimización: {e}")
            return None
    
    def run_optimization(self, model_id: str, initial_solution: Any = None,
                        max_iterations: int = 1000, 
                        convergence_threshold: float = 1e-6) -> OptimizationResult:
        """
        Ejecuta una optimización usando un modelo
        
        Args:
            model_id: ID del modelo
            initial_solution: Solución inicial
            max_iterations: Máximo número de iteraciones
            convergence_threshold: Umbral de convergencia
            
        Returns:
            Resultado de la optimización
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo no encontrado: {model_id}")
            
            metadata = self.models[model_id]
            
            # Cargar modelo
            model = self.load_optimization_model(model_id)
            if model is None:
                raise ValueError("No se pudo cargar el modelo")
            
            # Actualizar estado
            metadata.status = OptimizationStatus.RUNNING
            self._save_metadata()
            
            # Ejecutar optimización (simplificado)
            start_time = datetime.now()
            
            # Simular optimización
            best_solution = initial_solution or np.random.random(10)
            best_fitness = float('inf')
            convergence_history = []
            
            for iteration in range(max_iterations):
                # Simular evaluación de fitness
                current_fitness = np.random.random()
                convergence_history.append(current_fitness)
                
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_solution = np.random.random(10)  # Simular nueva solución
                
                # Verificar convergencia
                if len(convergence_history) > 10:
                    recent_improvement = abs(convergence_history[-1] - convergence_history[-10])
                    if recent_improvement < convergence_threshold:
                        break
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Crear resultado
            result = OptimizationResult(
                best_solution=best_solution,
                best_fitness=best_fitness,
                convergence_history=convergence_history,
                iterations=len(convergence_history),
                execution_time=execution_time,
                parameters=metadata.parameters,
                metadata={
                    'model_id': model_id,
                    'optimization_type': metadata.optimization_type.value,
                    'convergence_threshold': convergence_threshold
                }
            )
            
            # Actualizar metadatos
            metadata.best_result = result
            metadata.status = OptimizationStatus.CONVERGED
            metadata.last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Optimización completada para modelo {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error ejecutando optimización: {e}")
            # Actualizar estado a error
            if model_id in self.models:
                self.models[model_id].status = OptimizationStatus.ERROR
                self._save_metadata()
            raise
    
    def get_optimization_result(self, model_id: str) -> Optional[OptimizationResult]:
        """Obtiene el mejor resultado de optimización de un modelo"""
        if model_id not in self.models:
            return None
        return self.models[model_id].best_result
    
    def get_model_metadata(self, model_id: str) -> Optional[OptimizationModelMetadata]:
        """Obtiene metadatos de un modelo de optimización"""
        return self.models.get(model_id)
    
    def list_models(self, optimization_type: Optional[OptimizationType] = None,
                   status: Optional[OptimizationStatus] = None) -> List[OptimizationModelMetadata]:
        """
        Lista modelos con filtros opcionales
        
        Args:
            optimization_type: Filtrar por tipo de optimización
            status: Filtrar por estado
            
        Returns:
            Lista de metadatos de modelos
        """
        filtered_models = []
        
        for metadata in self.models.values():
            if optimization_type and metadata.optimization_type != optimization_type:
                continue
            if status and metadata.status != status:
                continue
            filtered_models.append(metadata)
        
        # Ordenar por fecha de creación (más recientes primero)
        filtered_models.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_models
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los modelos de optimización"""
        try:
            total_models = len(self.models)
            
            # Estadísticas por tipo
            type_counts = {}
            for optimization_type in OptimizationType:
                type_counts[optimization_type.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.optimization_type == optimization_type
                )
            
            # Estadísticas por estado
            status_counts = {}
            for status in OptimizationStatus:
                status_counts[status.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.status == status
                )
            
            # Tamaño total
            total_size = sum(metadata.file_size for metadata in self.models.values())
            
            # Modelos con resultados
            models_with_results = sum(
                1 for metadata in self.models.values()
                if metadata.best_result is not None
            )
            
            # Mejor fitness promedio
            fitness_values = [
                metadata.best_result.best_fitness
                for metadata in self.models.values()
                if metadata.best_result is not None
            ]
            avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0
            
            return {
                'total_models': total_models,
                'total_size': total_size,
                'models_with_results': models_with_results,
                'type_distribution': type_counts,
                'status_distribution': status_counts,
                'average_fitness': avg_fitness,
                'success_rate': models_with_results / total_models if total_models > 0 else 0
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
        """Guarda metadatos de modelos de optimización"""
        try:
            metadata = {
                'models': {
                    model_id: {
                        'model_id': metadata.model_id,
                        'name': metadata.name,
                        'optimization_type': metadata.optimization_type.value,
                        'status': metadata.status.value,
                        'created_at': metadata.created_at.isoformat(),
                        'last_updated': metadata.last_updated.isoformat(),
                        'version': metadata.version,
                        'description': metadata.description,
                        'objective_function': metadata.objective_function,
                        'constraints': metadata.constraints,
                        'parameters': metadata.parameters,
                        'best_result': {
                            'best_solution': metadata.best_result.best_solution.tolist() if metadata.best_result and hasattr(metadata.best_result.best_solution, 'tolist') else str(metadata.best_result.best_solution) if metadata.best_result else None,
                            'best_fitness': metadata.best_result.best_fitness if metadata.best_result else None,
                            'convergence_history': metadata.best_result.convergence_history if metadata.best_result else None,
                            'iterations': metadata.best_result.iterations if metadata.best_result else None,
                            'execution_time': metadata.best_result.execution_time if metadata.best_result else None,
                            'parameters': metadata.best_result.parameters if metadata.best_result else None,
                            'metadata': metadata.best_result.metadata if metadata.best_result else None
                        } if metadata.best_result else None,
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
        """Carga metadatos de modelos de optimización"""
        try:
            if not os.path.exists(self.metadata_file):
                return
            
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Cargar modelos
            for model_id, model_data in data.get('models', {}).items():
                # Cargar mejor resultado
                best_result = None
                if model_data.get('best_result'):
                    result_data = model_data['best_result']
                    best_result = OptimizationResult(
                        best_solution=np.array(result_data['best_solution']) if result_data['best_solution'] else None,
                        best_fitness=result_data['best_fitness'],
                        convergence_history=result_data['convergence_history'],
                        iterations=result_data['iterations'],
                        execution_time=result_data['execution_time'],
                        parameters=result_data['parameters'],
                        metadata=result_data['metadata']
                    )
                
                metadata = OptimizationModelMetadata(
                    model_id=model_data['model_id'],
                    name=model_data['name'],
                    optimization_type=OptimizationType(model_data['optimization_type']),
                    status=OptimizationStatus(model_data['status']),
                    created_at=datetime.fromisoformat(model_data['created_at']),
                    last_updated=datetime.fromisoformat(model_data['last_updated']),
                    version=model_data['version'],
                    description=model_data['description'],
                    objective_function=model_data['objective_function'],
                    constraints=model_data['constraints'],
                    parameters=model_data['parameters'],
                    best_result=best_result,
                    performance_metrics=model_data['performance_metrics'],
                    file_path=model_data['file_path'],
                    file_size=model_data['file_size'],
                    checksum=model_data['checksum']
                )
                
                # Solo cargar si el archivo existe
                if os.path.exists(metadata.file_path):
                    self.models[model_id] = metadata
            
            logger.info(f"Cargados {len(self.models)} modelos de optimización")
            
        except Exception as e:
            logger.error(f"Error cargando metadatos: {e}")

# Instancia global del gestor de modelos de optimización
optimization_model_manager = OptimizationModelManager()
