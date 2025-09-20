"""
Módulo 9: Sistema de Optimización
Versión: 0.6.0
Funcionalidad: Optimización de parámetros, recursos y rendimiento
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import math

logger = logging.getLogger('LucIA_Optimization')

class OptimizationAlgorithm(Enum):
    """Algoritmos de optimización"""
    GENETIC = "genetic"
    GRADIENT_DESCENT = "gradient_descent"
    SIMULATED_ANNEALING = "simulated_annealing"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"

@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    id: str
    algorithm: OptimizationAlgorithm
    best_parameters: Dict[str, Any]
    best_score: float
    iterations: int
    convergence_history: List[float]
    timestamp: datetime

class OptimizationSystem:
    """
    Sistema de optimización para LucIA.
    Gestiona la optimización de parámetros y recursos.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.optimization_results = {}
        self.active_optimizations = {}
        self.resource_limits = {
            "max_cpu": 80,
            "max_memory": 80,
            "max_disk": 90
        }
        
        # Estadísticas
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.failed_optimizations = 0
        
        logger.info("Sistema de optimización inicializado")
    
    async def optimize_parameters(self, objective_function: Callable,
                                parameter_space: Dict[str, Tuple[float, float]],
                                algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC,
                                max_iterations: int = 100,
                                population_size: int = 50) -> OptimizationResult:
        """
        Optimiza parámetros usando el algoritmo especificado
        
        Args:
            objective_function: Función objetivo a optimizar
            parameter_space: Espacio de parámetros {nombre: (min, max)}
            algorithm: Algoritmo de optimización
            max_iterations: Número máximo de iteraciones
            population_size: Tamaño de la población (para algoritmos genéticos)
        
        Returns:
            Resultado de la optimización
        """
        try:
            self.total_optimizations += 1
            optimization_id = f"opt_{int(datetime.now().timestamp())}"
            
            # Inicializar optimización
            self.active_optimizations[optimization_id] = {
                "status": "running",
                "start_time": datetime.now(),
                "iterations": 0
            }
            
            # Ejecutar optimización según algoritmo
            if algorithm == OptimizationAlgorithm.GENETIC:
                result = await self._genetic_optimization(
                    objective_function, parameter_space, max_iterations, population_size
                )
            elif algorithm == OptimizationAlgorithm.GRADIENT_DESCENT:
                result = await self._gradient_descent_optimization(
                    objective_function, parameter_space, max_iterations
                )
            elif algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
                result = await self._simulated_annealing_optimization(
                    objective_function, parameter_space, max_iterations
                )
            elif algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    objective_function, parameter_space, max_iterations
                )
            else:
                raise ValueError(f"Algoritmo no soportado: {algorithm}")
            
            # Crear resultado
            optimization_result = OptimizationResult(
                id=optimization_id,
                algorithm=algorithm,
                best_parameters=result["best_parameters"],
                best_score=result["best_score"],
                iterations=result["iterations"],
                convergence_history=result["convergence_history"],
                timestamp=datetime.now()
            )
            
            self.optimization_results[optimization_id] = optimization_result
            self.successful_optimizations += 1
            
            # Limpiar optimización activa
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            
            logger.info(f"Optimización completada: {optimization_id}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            self.failed_optimizations += 1
            raise
    
    async def _genetic_optimization(self, objective_function: Callable,
                                  parameter_space: Dict[str, Tuple[float, float]],
                                  max_iterations: int, population_size: int) -> Dict[str, Any]:
        """Optimización genética"""
        try:
            # Inicializar población
            population = self._initialize_population(parameter_space, population_size)
            best_individual = None
            best_score = float('-inf')
            convergence_history = []
            
            for iteration in range(max_iterations):
                # Evaluar población
                scores = []
                for individual in population:
                    try:
                        score = await objective_function(individual)
                        scores.append(score)
                        
                        if score > best_score:
                            best_score = score
                            best_individual = individual.copy()
                    except Exception as e:
                        scores.append(float('-inf'))
                
                convergence_history.append(best_score)
                
                # Selección, cruce y mutación
                new_population = []
                
                # Elitismo: mantener el mejor individuo
                if best_individual:
                    new_population.append(best_individual)
                
                # Generar nueva población
                while len(new_population) < population_size:
                    # Selección por torneo
                    parent1 = self._tournament_selection(population, scores)
                    parent2 = self._tournament_selection(population, scores)
                    
                    # Cruce
                    child = self._crossover(parent1, parent2, parameter_space)
                    
                    # Mutación
                    child = self._mutate(child, parameter_space)
                    
                    new_population.append(child)
                
                population = new_population
                
                # Verificar convergencia
                if len(convergence_history) > 10:
                    recent_improvement = max(convergence_history[-10:]) - min(convergence_history[-10:])
                    if recent_improvement < 0.001:
                        break
            
            return {
                "best_parameters": best_individual or {},
                "best_score": best_score,
                "iterations": iteration + 1,
                "convergence_history": convergence_history
            }
            
        except Exception as e:
            logger.error(f"Error en optimización genética: {e}")
            raise
    
    async def _gradient_descent_optimization(self, objective_function: Callable,
                                           parameter_space: Dict[str, Tuple[float, float]],
                                           max_iterations: int) -> Dict[str, Any]:
        """Optimización por descenso de gradiente"""
        try:
            # Inicializar parámetros aleatoriamente
            parameters = {}
            for param, (min_val, max_val) in parameter_space.items():
                parameters[param] = random.uniform(min_val, max_val)
            
            best_parameters = parameters.copy()
            best_score = float('-inf')
            convergence_history = []
            learning_rate = 0.01
            
            for iteration in range(max_iterations):
                try:
                    # Evaluar función objetivo
                    score = await objective_function(parameters)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = parameters.copy()
                    
                    convergence_history.append(score)
                    
                    # Calcular gradiente numérico
                    gradient = {}
                    for param in parameters:
                        # Perturbación pequeña
                        eps = 0.001
                        params_plus = parameters.copy()
                        params_plus[param] += eps
                        
                        try:
                            score_plus = await objective_function(params_plus)
                            gradient[param] = (score_plus - score) / eps
                        except:
                            gradient[param] = 0
                    
                    # Actualizar parámetros
                    for param in parameters:
                        parameters[param] += learning_rate * gradient[param]
                        # Mantener dentro de los límites
                        min_val, max_val = parameter_space[param]
                        parameters[param] = max(min_val, min(max_val, parameters[param]))
                    
                except Exception as e:
                    logger.warning(f"Error en iteración {iteration}: {e}")
                    break
            
            return {
                "best_parameters": best_parameters,
                "best_score": best_score,
                "iterations": iteration + 1,
                "convergence_history": convergence_history
            }
            
        except Exception as e:
            logger.error(f"Error en descenso de gradiente: {e}")
            raise
    
    async def _simulated_annealing_optimization(self, objective_function: Callable,
                                              parameter_space: Dict[str, Tuple[float, float]],
                                              max_iterations: int) -> Dict[str, Any]:
        """Optimización por recocido simulado"""
        try:
            # Inicializar parámetros aleatoriamente
            current_parameters = {}
            for param, (min_val, max_val) in parameter_space.items():
                current_parameters[param] = random.uniform(min_val, max_val)
            
            best_parameters = current_parameters.copy()
            best_score = float('-inf')
            convergence_history = []
            
            # Parámetros del recocido simulado
            initial_temperature = 100.0
            final_temperature = 0.1
            temperature = initial_temperature
            
            for iteration in range(max_iterations):
                try:
                    # Evaluar función objetivo
                    current_score = await objective_function(current_parameters)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_parameters = current_parameters.copy()
                    
                    convergence_history.append(current_score)
                    
                    # Generar vecino
                    neighbor_parameters = self._generate_neighbor(current_parameters, parameter_space)
                    neighbor_score = await objective_function(neighbor_parameters)
                    
                    # Criterio de aceptación
                    if neighbor_score > current_score or random.random() < math.exp((neighbor_score - current_score) / temperature):
                        current_parameters = neighbor_parameters
                        current_score = neighbor_score
                    
                    # Enfriar
                    temperature = initial_temperature * (final_temperature / initial_temperature) ** (iteration / max_iterations)
                    
                except Exception as e:
                    logger.warning(f"Error en iteración {iteration}: {e}")
                    break
            
            return {
                "best_parameters": best_parameters,
                "best_score": best_score,
                "iterations": iteration + 1,
                "convergence_history": convergence_history
            }
            
        except Exception as e:
            logger.error(f"Error en recocido simulado: {e}")
            raise
    
    async def _random_search_optimization(self, objective_function: Callable,
                                        parameter_space: Dict[str, Tuple[float, float]],
                                        max_iterations: int) -> Dict[str, Any]:
        """Optimización por búsqueda aleatoria"""
        try:
            best_parameters = {}
            best_score = float('-inf')
            convergence_history = []
            
            for iteration in range(max_iterations):
                try:
                    # Generar parámetros aleatorios
                    parameters = {}
                    for param, (min_val, max_val) in parameter_space.items():
                        parameters[param] = random.uniform(min_val, max_val)
                    
                    # Evaluar función objetivo
                    score = await objective_function(parameters)
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = parameters.copy()
                    
                    convergence_history.append(best_score)
                    
                except Exception as e:
                    logger.warning(f"Error en iteración {iteration}: {e}")
                    break
            
            return {
                "best_parameters": best_parameters,
                "best_score": best_score,
                "iterations": iteration + 1,
                "convergence_history": convergence_history
            }
            
        except Exception as e:
            logger.error(f"Error en búsqueda aleatoria: {e}")
            raise
    
    def _initialize_population(self, parameter_space: Dict[str, Tuple[float, float]], 
                             population_size: int) -> List[Dict[str, float]]:
        """Inicializa población para algoritmo genético"""
        population = []
        for _ in range(population_size):
            individual = {}
            for param, (min_val, max_val) in parameter_space.items():
                individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        return population
    
    def _tournament_selection(self, population: List[Dict[str, float]], 
                            scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_scores.index(max(tournament_scores))]
        return population[winner_index].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float],
                  parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Cruza dos individuos"""
        child = {}
        for param in parent1:
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict[str, float], 
               parameter_space: Dict[str, Tuple[float, float]], 
               mutation_rate: float = 0.1) -> Dict[str, float]:
        """Muta un individuo"""
        mutated = individual.copy()
        for param in mutated:
            if random.random() < mutation_rate:
                min_val, max_val = parameter_space[param]
                mutated[param] = random.uniform(min_val, max_val)
        return mutated
    
    def _generate_neighbor(self, parameters: Dict[str, float],
                          parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Genera un vecino para recocido simulado"""
        neighbor = parameters.copy()
        param_to_change = random.choice(list(parameters.keys()))
        min_val, max_val = parameter_space[param_to_change]
        
        # Perturbación gaussiana
        noise = random.gauss(0, (max_val - min_val) * 0.1)
        neighbor[param_to_change] = max(min_val, min(max_val, parameters[param_to_change] + noise))
        
        return neighbor
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimiza el uso de recursos del sistema"""
        try:
            # Monitorear recursos actuales
            current_resources = await self._monitor_resources()
            
            # Identificar optimizaciones
            optimizations = []
            
            if current_resources["cpu"] > self.resource_limits["max_cpu"]:
                optimizations.append({
                    "type": "cpu",
                    "current": current_resources["cpu"],
                    "limit": self.resource_limits["max_cpu"],
                    "suggestion": "Reducir procesos activos o aumentar límite"
                })
            
            if current_resources["memory"] > self.resource_limits["max_memory"]:
                optimizations.append({
                    "type": "memory",
                    "current": current_resources["memory"],
                    "limit": self.resource_limits["max_memory"],
                    "suggestion": "Limpiar caché o aumentar memoria disponible"
                })
            
            if current_resources["disk"] > self.resource_limits["max_disk"]:
                optimizations.append({
                    "type": "disk",
                    "current": current_resources["disk"],
                    "limit": self.resource_limits["max_disk"],
                    "suggestion": "Limpiar archivos temporales o aumentar espacio"
                })
            
            return {
                "current_resources": current_resources,
                "limits": self.resource_limits,
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizando recursos: {e}")
            return {"error": str(e)}
    
    async def _monitor_resources(self) -> Dict[str, float]:
        """Monitorea recursos del sistema"""
        try:
            import psutil
            
            return {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent
            }
        except ImportError:
            # Valores simulados si psutil no está disponible
            return {
                "cpu": 50.0,
                "memory": 60.0,
                "disk": 70.0
            }
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de optimización"""
        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "failed_optimizations": self.failed_optimizations,
            "success_rate": (self.successful_optimizations / max(self.total_optimizations, 1)) * 100,
            "active_optimizations": len(self.active_optimizations),
            "total_results": len(self.optimization_results)
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de optimización"""
        try:
            state = {
                "optimization_results": {
                    result_id: {
                        "id": result.id,
                        "algorithm": result.algorithm.value,
                        "best_parameters": result.best_parameters,
                        "best_score": result.best_score,
                        "iterations": result.iterations,
                        "convergence_history": result.convergence_history,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result_id, result in self.optimization_results.items()
                },
                "resource_limits": self.resource_limits,
                "stats": {
                    "total_optimizations": self.total_optimizations,
                    "successful_optimizations": self.successful_optimizations,
                    "failed_optimizations": self.failed_optimizations
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/optimization_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de optimización guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de optimización: {e}")

# Instancia global del sistema de optimización
optimization_system = OptimizationSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de optimización"""
    global optimization_system
    optimization_system.core_engine = core_engine
    core_engine.optimization_system = optimization_system
    logger.info("Módulo de optimización inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de optimización"""
    if isinstance(input_data, dict) and "optimize" in input_data:
        # Optimizar parámetros
        opt_data = input_data["optimize"]
        
        try:
            # Crear función objetivo simple
            def objective_function(params):
                # Función de ejemplo: suma de parámetros al cuadrado
                return sum(v**2 for v in params.values())
            
            parameter_space = opt_data.get("parameter_space", {
                "param1": (0, 10),
                "param2": (0, 10)
            })
            
            algorithm = OptimizationAlgorithm(opt_data.get("algorithm", "genetic"))
            
            result = await optimization_system.optimize_parameters(
                objective_function, parameter_space, algorithm
            )
            
            return {
                "optimization_completed": True,
                "best_parameters": result.best_parameters,
                "best_score": result.best_score
            }
            
        except Exception as e:
            return {"optimization_completed": False, "error": str(e)}
    
    return input_data

def run_modulo9():
    """Función de compatibilidad con el sistema anterior"""
    print("⚡ Módulo 9: Sistema de Optimización")
    print("   - Algoritmos genéticos y descenso de gradiente")
    print("   - Optimización de parámetros")
    print("   - Gestión de recursos")
    print("   - Recocido simulado")
    print("   ✅ Módulo inicializado correctamente")