"""
Módulo 8: Sistema de Evaluación
Versión: 0.6.0
Funcionalidad: Evaluación de rendimiento, métricas y análisis de calidad
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger('LucIA_Evaluation')

class MetricType(Enum):
    """Tipos de métricas"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    CUSTOM = "custom"

@dataclass
class EvaluationResult:
    """Resultado de evaluación"""
    id: str
    metric_type: MetricType
    value: float
    confidence: float
    timestamp: datetime
    context: Dict[str, Any]

class EvaluationSystem:
    """
    Sistema de evaluación para LucIA.
    Gestiona métricas de rendimiento y análisis de calidad.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.evaluation_results = {}
        self.performance_history = []
        self.benchmarks = {}
        self.thresholds = {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.7,
            "f1_score": 0.7,
            "mse": 0.1,
            "rmse": 0.3,
            "mae": 0.2,
            "r2_score": 0.8
        }
        
        # Estadísticas
        self.total_evaluations = 0
        self.passed_evaluations = 0
        self.failed_evaluations = 0
        
        logger.info("Sistema de evaluación inicializado")
    
    async def evaluate_performance(self, predictions: List[Any], 
                                 actuals: List[Any], 
                                 metric_types: List[MetricType] = None) -> Dict[str, EvaluationResult]:
        """
        Evalúa el rendimiento de predicciones
        
        Args:
            predictions: Predicciones del modelo
            actuals: Valores reales
            metric_types: Tipos de métricas a calcular
        
        Returns:
            Diccionario con resultados de evaluación
        """
        try:
            self.total_evaluations += 1
            
            if metric_types is None:
                metric_types = [MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1_SCORE]
            
            results = {}
            
            for metric_type in metric_types:
                value = await self._calculate_metric(predictions, actuals, metric_type)
                confidence = self._calculate_confidence(value, metric_type)
                
                result = EvaluationResult(
                    id=f"eval_{int(datetime.now().timestamp())}_{metric_type.value}",
                    metric_type=metric_type,
                    value=value,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    context={
                        "predictions_count": len(predictions),
                        "actuals_count": len(actuals),
                        "threshold": self.thresholds.get(metric_type.value, 0.5)
                    }
                )
                
                results[metric_type.value] = result
                self.evaluation_results[result.id] = result
                
                # Verificar si pasa el umbral
                if value >= self.thresholds.get(metric_type.value, 0.5):
                    self.passed_evaluations += 1
                else:
                    self.failed_evaluations += 1
            
            # Agregar a historial
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "results": {k: v.value for k, v in results.items()},
                "overall_score": statistics.mean([r.value for r in results.values()])
            })
            
            logger.info(f"Evaluación completada: {len(results)} métricas calculadas")
            return results
            
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            return {}
    
    async def _calculate_metric(self, predictions: List[Any], actuals: List[Any], 
                              metric_type: MetricType) -> float:
        """Calcula una métrica específica"""
        try:
            if len(predictions) != len(actuals):
                raise ValueError("Las listas de predicciones y valores reales deben tener la misma longitud")
            
            if metric_type == MetricType.ACCURACY:
                return self._calculate_accuracy(predictions, actuals)
            elif metric_type == MetricType.PRECISION:
                return self._calculate_precision(predictions, actuals)
            elif metric_type == MetricType.RECALL:
                return self._calculate_recall(predictions, actuals)
            elif metric_type == MetricType.F1_SCORE:
                return self._calculate_f1_score(predictions, actuals)
            elif metric_type == MetricType.MSE:
                return self._calculate_mse(predictions, actuals)
            elif metric_type == MetricType.RMSE:
                return self._calculate_rmse(predictions, actuals)
            elif metric_type == MetricType.MAE:
                return self._calculate_mae(predictions, actuals)
            elif metric_type == MetricType.R2_SCORE:
                return self._calculate_r2_score(predictions, actuals)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculando métrica {metric_type.value}: {e}")
            return 0.0
    
    def _calculate_accuracy(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula la precisión (accuracy)"""
        try:
            correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
            return correct / len(predictions) if predictions else 0.0
        except:
            return 0.0
    
    def _calculate_precision(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula la precisión (precision)"""
        try:
            # Para clasificación binaria
            true_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
            false_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
            
            if true_positives + false_positives == 0:
                return 0.0
            
            return true_positives / (true_positives + false_positives)
        except:
            return 0.0
    
    def _calculate_recall(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula la sensibilidad (recall)"""
        try:
            # Para clasificación binaria
            true_positives = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
            false_negatives = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
            
            if true_positives + false_negatives == 0:
                return 0.0
            
            return true_positives / (true_positives + false_negatives)
        except:
            return 0.0
    
    def _calculate_f1_score(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula el F1-score"""
        try:
            precision = self._calculate_precision(predictions, actuals)
            recall = self._calculate_recall(predictions, actuals)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * (precision * recall) / (precision + recall)
        except:
            return 0.0
    
    def _calculate_mse(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula el error cuadrático medio (MSE)"""
        try:
            predictions = np.array(predictions, dtype=float)
            actuals = np.array(actuals, dtype=float)
            
            return np.mean((predictions - actuals) ** 2)
        except:
            return float('inf')
    
    def _calculate_rmse(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula la raíz del error cuadrático medio (RMSE)"""
        try:
            mse = self._calculate_mse(predictions, actuals)
            return np.sqrt(mse)
        except:
            return float('inf')
    
    def _calculate_mae(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula el error absoluto medio (MAE)"""
        try:
            predictions = np.array(predictions, dtype=float)
            actuals = np.array(actuals, dtype=float)
            
            return np.mean(np.abs(predictions - actuals))
        except:
            return float('inf')
    
    def _calculate_r2_score(self, predictions: List[Any], actuals: List[Any]) -> float:
        """Calcula el coeficiente de determinación (R²)"""
        try:
            predictions = np.array(predictions, dtype=float)
            actuals = np.array(actuals, dtype=float)
            
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            
            if ss_tot == 0:
                return 0.0
            
            return 1 - (ss_res / ss_tot)
        except:
            return 0.0
    
    def _calculate_confidence(self, value: float, metric_type: MetricType) -> float:
        """Calcula la confianza en una métrica"""
        try:
            threshold = self.thresholds.get(metric_type.value, 0.5)
            
            # Confianza basada en qué tan cerca está del umbral
            if value >= threshold:
                return min(1.0, (value - threshold) / (1.0 - threshold) + 0.5)
            else:
                return max(0.0, value / threshold * 0.5)
                
        except:
            return 0.5
    
    async def set_benchmark(self, name: str, values: Dict[str, float]):
        """Establece un benchmark de rendimiento"""
        try:
            self.benchmarks[name] = {
                "values": values,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Benchmark '{name}' establecido")
        except Exception as e:
            logger.error(f"Error estableciendo benchmark: {e}")
    
    async def compare_to_benchmark(self, results: Dict[str, float], benchmark_name: str) -> Dict[str, Any]:
        """Compara resultados con un benchmark"""
        try:
            if benchmark_name not in self.benchmarks:
                return {"error": f"Benchmark '{benchmark_name}' no encontrado"}
            
            benchmark = self.benchmarks[benchmark_name]
            comparison = {}
            
            for metric, value in results.items():
                if metric in benchmark["values"]:
                    benchmark_value = benchmark["values"][metric]
                    improvement = ((value - benchmark_value) / benchmark_value) * 100
                    
                    comparison[metric] = {
                        "current": value,
                        "benchmark": benchmark_value,
                        "improvement": improvement,
                        "better": value > benchmark_value
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando con benchmark: {e}")
            return {"error": str(e)}
    
    async def get_performance_trend(self, metric_type: str, days: int = 7) -> List[Dict[str, Any]]:
        """Obtiene la tendencia de rendimiento de una métrica"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            trend_data = []
            for record in self.performance_history:
                record_date = datetime.fromisoformat(record["timestamp"])
                if record_date >= cutoff_date and metric_type in record["results"]:
                    trend_data.append({
                        "timestamp": record["timestamp"],
                        "value": record["results"][metric_type],
                        "overall_score": record["overall_score"]
                    })
            
            return sorted(trend_data, key=lambda x: x["timestamp"])
            
        except Exception as e:
            logger.error(f"Error obteniendo tendencia de rendimiento: {e}")
            return []
    
    async def get_evaluation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de evaluación"""
        return {
            "total_evaluations": self.total_evaluations,
            "passed_evaluations": self.passed_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "success_rate": (self.passed_evaluations / max(self.total_evaluations, 1)) * 100,
            "total_results": len(self.evaluation_results),
            "benchmarks_count": len(self.benchmarks),
            "performance_history_count": len(self.performance_history)
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de evaluación"""
        try:
            state = {
                "evaluation_results": {
                    result_id: {
                        "id": result.id,
                        "metric_type": result.metric_type.value,
                        "value": result.value,
                        "confidence": result.confidence,
                        "timestamp": result.timestamp.isoformat(),
                        "context": result.context
                    }
                    for result_id, result in self.evaluation_results.items()
                },
                "performance_history": self.performance_history,
                "benchmarks": self.benchmarks,
                "thresholds": self.thresholds,
                "stats": {
                    "total_evaluations": self.total_evaluations,
                    "passed_evaluations": self.passed_evaluations,
                    "failed_evaluations": self.failed_evaluations
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/evaluation_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de evaluación guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de evaluación: {e}")

# Instancia global del sistema de evaluación
evaluation_system = EvaluationSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de evaluación"""
    global evaluation_system
    evaluation_system.core_engine = core_engine
    core_engine.evaluation_system = evaluation_system
    logger.info("Módulo de evaluación inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de evaluación"""
    if isinstance(input_data, dict) and "evaluate" in input_data:
        # Evaluar rendimiento
        eval_data = input_data["evaluate"]
        
        try:
            predictions = eval_data.get("predictions", [])
            actuals = eval_data.get("actuals", [])
            metric_types = [MetricType(m) for m in eval_data.get("metrics", ["accuracy"])]
            
            results = await evaluation_system.evaluate_performance(
                predictions, actuals, metric_types
            )
            
            return {
                "evaluation_completed": True,
                "results": {k: v.value for k, v in results.items()}
            }
            
        except Exception as e:
            return {"evaluation_completed": False, "error": str(e)}
    
    return input_data

def run_modulo8():
    """Función de compatibilidad con el sistema anterior"""
    print("📊 Módulo 8: Sistema de Evaluación")
    print("   - Métricas de rendimiento")
    print("   - Análisis de calidad")
    print("   - Comparación con benchmarks")
    print("   - Tendencias de rendimiento")
    print("   ✅ Módulo inicializado correctamente")