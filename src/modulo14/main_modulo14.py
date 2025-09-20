#!/usr/bin/env python3
"""
Módulo 14 - Sistema Integrador de Infraestructura - LucIA
Versión: 0.6.0
Sistema que integra todos los sistemas de infraestructura con el motor principal
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Importar todos los sistemas de infraestructura
from cache.intelligent_cache import IntelligentCache, CacheStrategy
from temp.temp_manager import TempFileManager, TempFileType
from models.neural.neural_models import NeuralModelManager, ModelType, ModelStatus
from models.decision.decision_models import DecisionModelManager, DecisionType, DecisionStatus
from models.optimization.optimization_models import OptimizationModelManager, OptimizationType, OptimizationStatus
from data.learning.learning_data_manager import LearningDataManager, DataType, DataFormat

logger = logging.getLogger('Modulo14_Infraestructura')

class InfrastructureIntegrator:
    """
    Integrador de sistemas de infraestructura
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.is_initialized = False
        
        # Inicializar sistemas de infraestructura
        self.cache_system = IntelligentCache()
        self.temp_manager = TempFileManager()
        self.neural_models = NeuralModelManager()
        self.decision_models = DecisionModelManager()
        self.optimization_models = OptimizationModelManager()
        self.learning_data = LearningDataManager()
        
        logger.info("Integrador de infraestructura inicializado")
    
    async def initialize_module(self, core_engine):
        """Inicializa el módulo de infraestructura"""
        try:
            self.core_engine = core_engine
            self.is_initialized = True
            
            # Cargar datos persistentes
            await self._load_persistent_data()
            
            logger.info("Módulo 14 - Sistema Integrador de Infraestructura inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando módulo de infraestructura: {e}")
            self.is_initialized = False
    
    async def _load_persistent_data(self):
        """Carga datos persistentes de todos los sistemas"""
        try:
            # Los sistemas ya cargan sus datos automáticamente en el constructor
            logger.info("Datos persistentes cargados")
            
        except Exception as e:
            logger.error(f"Error cargando datos persistentes: {e}")
    
    # ===== MÉTODOS DE CACHÉ =====
    
    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None,
                       strategy: Optional[CacheStrategy] = None) -> bool:
        """Almacena un valor en el caché"""
        return self.cache_system.set(key, value, ttl, strategy)
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        return self.cache_system.get(key)
    
    async def cache_delete(self, key: str) -> bool:
        """Elimina un valor del caché"""
        return self.cache_system.delete(key)
    
    async def cache_clear(self) -> bool:
        """Limpia todo el caché"""
        return self.cache_system.clear()
    
    async def cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        return self.cache_system.get_info()
    
    # ===== MÉTODOS DE ARCHIVOS TEMPORALES =====
    
    async def create_temp_file(self, file_type: TempFileType, content: Optional[str] = None,
                              extension: str = ".tmp", ttl: Optional[int] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Crea un archivo temporal"""
        return self.temp_manager.create_temp_file(file_type, content, extension, ttl, metadata)
    
    async def read_temp_file(self, file_id: str, as_text: bool = True) -> Optional[Union[str, bytes]]:
        """Lee un archivo temporal"""
        return self.temp_manager.read_temp_file(file_id, as_text)
    
    async def write_temp_file(self, file_id: str, content: Union[str, bytes]) -> bool:
        """Escribe en un archivo temporal"""
        return self.temp_manager.write_temp_file(file_id, content)
    
    async def delete_temp_file(self, file_id: str) -> bool:
        """Elimina un archivo temporal"""
        return self.temp_manager.delete_temp_file(file_id)
    
    async def temp_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de archivos temporales"""
        return self.temp_manager.get_stats()
    
    # ===== MÉTODOS DE MODELOS NEURONALES =====
    
    async def save_neural_model(self, model: Any, name: str, model_type: ModelType,
                               description: str = "", version: str = "1.0.0",
                               input_shape: tuple = None, output_shape: tuple = None,
                               parameters: Dict[str, Any] = None,
                               performance_metrics: Dict[str, float] = None) -> str:
        """Guarda un modelo neuronal"""
        return self.neural_models.save_model(
            model, name, model_type, description, version,
            input_shape, output_shape, parameters, performance_metrics
        )
    
    async def load_neural_model(self, model_id: str) -> Optional[Any]:
        """Carga un modelo neuronal"""
        return self.neural_models.load_model(model_id)
    
    async def list_neural_models(self, model_type: Optional[ModelType] = None,
                                status: Optional[ModelStatus] = None) -> List[Any]:
        """Lista modelos neuronales"""
        return self.neural_models.list_models(model_type, status)
    
    async def neural_models_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de modelos neuronales"""
        return self.neural_models.get_model_stats()
    
    # ===== MÉTODOS DE MODELOS DE DECISIÓN =====
    
    async def save_decision_model(self, model: Any, name: str, decision_type: DecisionType,
                                 description: str = "", version: str = "1.0.0",
                                 input_features: List[str] = None,
                                 output_classes: List[str] = None,
                                 rules: List[Any] = None,
                                 parameters: Dict[str, Any] = None,
                                 performance_metrics: Dict[str, float] = None) -> str:
        """Guarda un modelo de decisión"""
        return self.decision_models.save_decision_model(
            model, name, decision_type, description, version,
            input_features, output_classes, rules, parameters, performance_metrics
        )
    
    async def load_decision_model(self, model_id: str) -> Optional[Any]:
        """Carga un modelo de decisión"""
        return self.decision_models.load_decision_model(model_id)
    
    async def evaluate_decision(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa una decisión usando un modelo"""
        return self.decision_models.evaluate_decision(model_id, input_data)
    
    async def decision_models_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de modelos de decisión"""
        return self.decision_models.get_model_stats()
    
    # ===== MÉTODOS DE MODELOS DE OPTIMIZACIÓN =====
    
    async def save_optimization_model(self, model: Any, name: str, optimization_type: OptimizationType,
                                     description: str = "", version: str = "1.0.0",
                                     objective_function: str = "",
                                     constraints: List[str] = None,
                                     parameters: Dict[str, Any] = None,
                                     performance_metrics: Dict[str, float] = None) -> str:
        """Guarda un modelo de optimización"""
        return self.optimization_models.save_optimization_model(
            model, name, optimization_type, description, version,
            objective_function, constraints, parameters, performance_metrics
        )
    
    async def load_optimization_model(self, model_id: str) -> Optional[Any]:
        """Carga un modelo de optimización"""
        return self.optimization_models.load_optimization_model(model_id)
    
    async def run_optimization(self, model_id: str, initial_solution: Any = None,
                              max_iterations: int = 1000,
                              convergence_threshold: float = 1e-6) -> Dict[str, Any]:
        """Ejecuta una optimización"""
        return self.optimization_models.run_optimization(
            model_id, initial_solution, max_iterations, convergence_threshold
        )
    
    async def optimization_models_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de modelos de optimización"""
        return self.optimization_models.get_model_stats()
    
    # ===== MÉTODOS DE DATOS DE APRENDIZAJE =====
    
    async def save_learning_dataset(self, data: Any, name: str, data_type: DataType,
                                   description: str = "", features: List[str] = None,
                                   target_column: str = None,
                                   metadata: Dict[str, Any] = None) -> str:
        """Guarda un dataset de aprendizaje"""
        return self.learning_data.save_dataset(
            data, name, data_type, description, features, target_column, metadata
        )
    
    async def load_learning_dataset(self, dataset_id: str) -> Optional[Any]:
        """Carga un dataset de aprendizaje"""
        return self.learning_data.load_dataset(dataset_id)
    
    async def split_learning_dataset(self, dataset_id: str, train_ratio: float = 0.7,
                                    val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, str]:
        """Divide un dataset de aprendizaje"""
        return self.learning_data.split_dataset(dataset_id, train_ratio, val_ratio, test_ratio)
    
    async def learning_data_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de datos de aprendizaje"""
        return self.learning_data.get_dataset_stats()
    
    # ===== MÉTODOS DE INTEGRACIÓN =====
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los sistemas de infraestructura"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'is_initialized': self.is_initialized,
                'cache_stats': await self.cache_stats(),
                'temp_stats': await self.temp_stats(),
                'neural_models_stats': await self.neural_models_stats(),
                'decision_models_stats': await self.decision_models_stats(),
                'optimization_models_stats': await self.optimization_models_stats(),
                'learning_data_stats': await self.learning_data_stats(),
                'overall_health': self._calculate_overall_health()
            }
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_health(self) -> str:
        """Calcula la salud general del sistema"""
        try:
            # Verificar que todos los sistemas estén funcionando
            systems_ok = 0
            total_systems = 6
            
            # Verificar caché
            if self.cache_system:
                systems_ok += 1
            
            # Verificar temp manager
            if self.temp_manager:
                systems_ok += 1
            
            # Verificar modelos neuronales
            if self.neural_models:
                systems_ok += 1
            
            # Verificar modelos de decisión
            if self.decision_models:
                systems_ok += 1
            
            # Verificar modelos de optimización
            if self.optimization_models:
                systems_ok += 1
            
            # Verificar datos de aprendizaje
            if self.learning_data:
                systems_ok += 1
            
            health_percentage = (systems_ok / total_systems) * 100
            
            if health_percentage >= 90:
                return "excellent"
            elif health_percentage >= 75:
                return "good"
            elif health_percentage >= 50:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error calculando salud del sistema: {e}")
            return "unknown"
    
    async def optimize_systems(self) -> Dict[str, Any]:
        """Optimiza todos los sistemas de infraestructura"""
        try:
            optimization_results = {}
            
            # Optimizar caché
            cache_optimization = self.cache_system.optimize()
            optimization_results['cache'] = cache_optimization
            
            # Limpiar archivos temporales expirados
            temp_cleanup = self.temp_manager.cleanup_expired()
            optimization_results['temp_cleanup'] = temp_cleanup
            
            # Obtener estadísticas generales
            optimization_results['overall_stats'] = await self.get_system_status()
            
            logger.info("Optimización de sistemas completada")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizando sistemas: {e}")
            return {'error': str(e)}
    
    async def save_state(self):
        """Guarda el estado de todos los sistemas"""
        try:
            # Los sistemas individuales ya guardan su estado automáticamente
            logger.info("Estado de sistemas de infraestructura guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")

# Función de inicialización para el motor principal
async def initialize_module(core_engine):
    """Función de inicialización requerida por el motor principal"""
    infrastructure_module = InfrastructureIntegrator(core_engine)
    await infrastructure_module.initialize_module(core_engine)
    
    # Agregar el módulo al diccionario de módulos del core
    core_engine.modules["infrastructure"] = infrastructure_module
    
    return infrastructure_module
