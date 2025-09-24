#!/usr/bin/env python3
"""
Sistema de Caché Inteligente - LucIA
Versión: 0.7.0
Sistema avanzado de caché con estrategias de invalidación y optimización inteligente
Diseñado para proporcionar almacenamiento temporal eficiente con múltiples estrategias de reemplazo
"""

import os
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import asyncio
from pathlib import Path

# Configuración del logger
logger = logging.getLogger('Intelligent_Cache')

class CacheStrategy(Enum):
    """
    Estrategias de caché disponibles
    
    LRU: Elimina el elemento menos recientemente usado
    LFU: Elimina el elemento menos frecuentemente usado
    TTL: Basado en tiempo de vida
    ADAPTIVE: Estrategia adaptativa que combina frecuencia y recencia
    """
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class CacheMetrics:
    """
    Métricas detalladas del rendimiento del caché
    Proporciona información completa sobre el uso y eficiencia
    """
    total_items: int = 0
    total_size: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    memory_usage: float = 0.0
    avg_access_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte las métricas a diccionario"""
        return asdict(self)

class CacheItem:
    """
    Elemento individual del caché con metadatos completos
    Gestiona información de acceso, expiración y scoring para estrategias de reemplazo
    """
    
    def __init__(self, key: str, value: Any, strategy: CacheStrategy = CacheStrategy.LRU):
        """
        Inicializa un elemento del caché
        
        Args:
            key: Identificador único del elemento
            value: Valor a almacenar
            strategy: Estrategia de reemplazo para este elemento
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.strategy = strategy
        self.ttl = None  # Tiempo de vida en segundos
        self.size = self._calculate_size()
        self.hash = self._generate_hash()
    
    def _calculate_size(self) -> int:
        """
        Calcula el tamaño aproximado del elemento en bytes
        Utiliza diferentes métodos según el tipo de dato
        
        Returns:
            Tamaño en bytes del elemento
        """
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value, default=str).encode('utf-8'))
            elif isinstance(self.value, (int, float)):
                return 8  # Tamaño aproximado de números
            else:
                return len(pickle.dumps(self.value))
        except Exception as e:
            logger.warning(f"Error calculando tamaño para {self.key}: {e}")
            return 1024  # Tamaño por defecto seguro
    
    def _generate_hash(self) -> str:
        """
        Genera un hash para verificar integridad del elemento
        
        Returns:
            Hash MD5 del contenido
        """
        try:
            content = f"{self.key}{self.value}{self.created_at}"
            return hashlib.md5(content.encode()).hexdigest()
        except:
            return hashlib.md5(f"{self.key}{time.time()}".encode()).hexdigest()
    
    def access(self) -> None:
        """
        Registra un acceso al elemento
        Actualiza metadatos de uso para estrategias de reemplazo
        """
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """
        Verifica si el elemento ha expirado según su TTL
        
        Returns:
            True si ha expirado, False en caso contrario
        """
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl
    
    def get_replacement_score(self) -> float:
        """
        Calcula el score unificado para estrategias de reemplazo
        Score más bajo = mayor probabilidad de ser eliminado
        
        Returns:
            Score numérico para comparación
        """
        now = datetime.now()
        
        if self.strategy == CacheStrategy.LRU:
            # Score basado en tiempo desde último acceso
            return (now - self.last_accessed).total_seconds()
            
        elif self.strategy == CacheStrategy.LFU:
            # Score inverso basado en frecuencia (menos accesos = score mayor)
            return 1.0 / (self.access_count + 1)
            
        elif self.strategy == CacheStrategy.TTL:
            # Score basado en tiempo restante de vida
            if self.ttl is None:
                return float('inf')  # Nunca expira
            elapsed = (now - self.created_at).total_seconds()
            remaining = self.ttl - elapsed
            return -remaining  # Negativo para que próximos a expirar tengan score alto
            
        else:  # ADAPTIVE
            # Estrategia híbrida: combina recencia, frecuencia y tamaño
            recency_factor = (now - self.last_accessed).total_seconds()
            frequency_factor = self.access_count + 1
            size_factor = self.size / 1024  # Normalizar a KB
            
            # Score adaptativo balanceado
            return (recency_factor * size_factor) / frequency_factor
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtiene información completa del elemento
        
        Returns:
            Diccionario con metadatos del elemento
        """
        return {
            'key': self.key,
            'size': self.size,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'strategy': self.strategy.value,
            'ttl': self.ttl,
            'hash': self.hash,
            'is_expired': self.is_expired(),
            'score': self.get_replacement_score()
        }

class IntelligentCache:
    """
    Sistema de Caché Inteligente con capacidades avanzadas
    
    Características principales:
    - Múltiples estrategias de reemplazo (LRU, LFU, TTL, ADAPTIVE)
    - Persistencia automática de datos
    - Limpieza automática de elementos expirados
    - Métricas detalladas de rendimiento
    - Optimización automática basada en patrones de uso
    - Gestión de memoria inteligente
    """
    
    def __init__(self, 
                 max_size: int = 100 * 1024 * 1024,  # 100MB
                 max_items: int = 1000,
                 default_strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 cache_dir: str = "cache",
                 auto_persist: bool = True,
                 cleanup_interval: int = 300):  # 5 minutos
        """
        Inicializa el sistema de caché inteligente
        
        Args:
            max_size: Tamaño máximo del caché en bytes
            max_items: Número máximo de elementos
            default_strategy: Estrategia por defecto para nuevos elementos
            cache_dir: Directorio para archivos persistentes
            auto_persist: Activar persistencia automática
            cleanup_interval: Intervalo de limpieza automática en segundos
        """
        # Configuración principal
        self.max_size = max_size
        self.max_items = max_items
        self.default_strategy = default_strategy
        self.auto_persist = auto_persist
        self.cleanup_interval = cleanup_interval
        
        # Almacenamiento y sincronización
        self.cache: Dict[str, CacheItem] = {}
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        
        # Configuración de persistencia
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.persist_file = self.cache_dir / "cache_data.pkl"
        self.metrics_file = self.cache_dir / "cache_metrics.json"
        
        # Estado interno
        self._shutdown = False
        self._last_optimization = datetime.now()
        
        # Inicialización
        self._initialize_cache()
        
        logger.info(f"Sistema de caché inteligente inicializado - "
                   f"Max: {max_size//1024//1024}MB, Items: {max_items}, "
                   f"Estrategia: {default_strategy.value}")
    
    def _initialize_cache(self) -> None:
        """
        Inicializa el caché y sus componentes
        Carga datos persistentes y configura tareas automáticas
        """
        try:
            # Cargar datos persistentes si existen
            if self.auto_persist:
                self._load_persistent_data()
            
            # Iniciar tareas automáticas
            self._start_background_tasks()
            
        except Exception as e:
            logger.error(f"Error inicializando caché: {e}")
    
    def _start_background_tasks(self) -> None:
        """
        Inicia tareas de mantenimiento en segundo plano
        - Limpieza de elementos expirados
        - Persistencia automática
        - Optimización periódica
        """
        def maintenance_loop():
            """Bucle principal de mantenimiento"""
            while not self._shutdown:
                try:
                    # Esperar intervalo de limpieza
                    time.sleep(self.cleanup_interval)
                    
                    # Ejecutar tareas de mantenimiento
                    self._unified_maintenance()
                    
                except Exception as e:
                    logger.error(f"Error en mantenimiento automático: {e}")
        
        # Iniciar hilo de mantenimiento
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
        logger.debug("Tareas de mantenimiento automático iniciadas")
    
    def _unified_maintenance(self) -> None:
        """
        Ejecuta todas las tareas de mantenimiento de forma unificada
        Optimiza el rendimiento al procesar todo en una sola pasada
        """
        with self.lock:
            expired_keys = []
            current_time = datetime.now()
            
            # Una sola iteración para todas las verificaciones
            for key, item in list(self.cache.items()):
                # Verificar expiración
                if item.is_expired():
                    expired_keys.append(key)
            
            # Eliminar elementos expirados
            for key in expired_keys:
                del self.cache[key]
                self.metrics.evictions += 1
            
            # Actualizar métricas
            self._update_unified_metrics()
            
            # Persistir datos si está habilitado
            if self.auto_persist:
                self._save_persistent_data()
            
            # Log de mantenimiento
            if expired_keys:
                logger.debug(f"Mantenimiento: eliminados {len(expired_keys)} elementos expirados")
    
    def set(self, key: str, value: Any, 
            ttl: Optional[int] = None, 
            strategy: Optional[CacheStrategy] = None) -> bool:
        """
        Almacena un elemento en el caché con gestión inteligente de espacio
        
        Args:
            key: Clave única del elemento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos (opcional)
            strategy: Estrategia de reemplazo específica (opcional)
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        try:
            with self.lock:
                # Crear elemento del caché
                strategy = strategy or self.default_strategy
                item = CacheItem(key, value, strategy)
                
                # Configurar TTL si se especifica
                if ttl is not None:
                    item.ttl = ttl
                
                # Gestión inteligente de espacio
                if self._requires_space_management(item):
                    freed_space = self._intelligent_eviction(item.size)
                    if freed_space < item.size:
                        logger.warning(f"No se pudo liberar suficiente espacio para {key}")
                        return False
                
                # Almacenar elemento
                self.cache[key] = item
                
                # Actualizar métricas
                self._update_unified_metrics()
                
                logger.debug(f"Elemento almacenado: {key} ({item.size} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando elemento {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Recupera un elemento del caché con validaciones automáticas
        
        Args:
            key: Clave del elemento a recuperar
            
        Returns:
            Valor del elemento o None si no existe/expiró
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Verificar existencia
                if key not in self.cache:
                    self.metrics.misses += 1
                    self._update_hit_rate()
                    return None
                
                item = self.cache[key]
                
                # Verificar expiración
                if item.is_expired():
                    del self.cache[key]
                    self.metrics.misses += 1
                    self.metrics.evictions += 1
                    self._update_hit_rate()
                    return None
                
                # Registrar acceso exitoso
                item.access()
                self.metrics.hits += 1
                
                # Actualizar tiempo promedio de acceso
                access_time = (time.time() - start_time) * 1000  # ms
                self._update_access_time(access_time)
                
                self._update_hit_rate()
                
                logger.debug(f"Elemento recuperado: {key}")
                return item.value
                
        except Exception as e:
            logger.error(f"Error recuperando elemento {key}: {e}")
            self.metrics.misses += 1
            return None
    
    def delete(self, key: str) -> bool:
        """
        Elimina un elemento específico del caché
        
        Args:
            key: Clave del elemento a eliminar
            
        Returns:
            True si se eliminó, False si no existía
        """
        try:
            with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    self._update_unified_metrics()
                    logger.debug(f"Elemento eliminado: {key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando elemento {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Limpia completamente el caché
        
        Returns:
            True si se limpió correctamente
        """
        try:
            with self.lock:
                items_count = len(self.cache)
                self.cache.clear()
                
                # Resetear métricas pero mantener historial acumulado
                self.metrics.total_items = 0
                self.metrics.total_size = 0
                
                logger.info(f"Caché limpiado: {items_count} elementos eliminados")
                return True
                
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
            return False
    
    def _requires_space_management(self, new_item: CacheItem) -> bool:
        """
        Determina si se necesita gestión de espacio para un nuevo elemento
        
        Args:
            new_item: Elemento que se va a agregar
            
        Returns:
            True si se necesita liberar espacio
        """
        current_size = sum(item.size for item in self.cache.values())
        return (current_size + new_item.size > self.max_size or 
                len(self.cache) >= self.max_items)
    
    def _intelligent_eviction(self, required_space: int) -> int:
        """
        Eliminación inteligente de elementos basada en scores unificados
        
        Args:
            required_space: Espacio mínimo a liberar en bytes
            
        Returns:
            Espacio real liberado en bytes
        """
        if not self.cache:
            return 0
        
        # Ordenar elementos por score de reemplazo (menor = más candidato)
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].get_replacement_score()
        )
        
        freed_space = 0
        items_removed = []
        
        # Eliminar elementos hasta liberar el espacio necesario
        for key, item in sorted_items:
            items_removed.append(key)
            freed_space += item.size
            self.metrics.evictions += 1
            
            # Verificar si hemos liberado suficiente espacio
            if freed_space >= required_space:
                break
        
        # Ejecutar eliminaciones
        for key in items_removed:
            del self.cache[key]
        
        logger.debug(f"Eliminación inteligente: {len(items_removed)} elementos, "
                    f"{freed_space} bytes liberados")
        
        return freed_space
    
    def _update_unified_metrics(self) -> None:
        """
        Actualiza todas las métricas del caché de forma unificada
        Optimiza el rendimiento al calcular todo en una sola pasada
        """
        self.metrics.total_items = len(self.cache)
        self.metrics.total_size = sum(item.size for item in self.cache.values())
        
        # Calcular uso de memoria como porcentaje
        if self.max_size > 0:
            self.metrics.memory_usage = (self.metrics.total_size / self.max_size) * 100
        
        # Actualizar tasa de aciertos
        self._update_hit_rate()
    
    def _update_hit_rate(self) -> None:
        """Actualiza la tasa de aciertos del caché"""
        total_requests = self.metrics.hits + self.metrics.misses
        if total_requests > 0:
            self.metrics.hit_rate = (self.metrics.hits / total_requests) * 100
    
    def _update_access_time(self, access_time: float) -> None:
        """
        Actualiza el tiempo promedio de acceso usando media móvil
        
        Args:
            access_time: Tiempo de acceso actual en milisegundos
        """
        if self.metrics.avg_access_time == 0:
            self.metrics.avg_access_time = access_time
        else:
            # Media móvil exponencial con factor de suavizado 0.1
            alpha = 0.1
            self.metrics.avg_access_time = (
                alpha * access_time + 
                (1 - alpha) * self.metrics.avg_access_time
            )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas comprehensivas del sistema de caché
        
        Returns:
            Diccionario con información completa del estado del caché
        """
        with self.lock:
            # Actualizar métricas actuales
            self._update_unified_metrics()
            
            # Análisis de estrategias de reemplazo
            strategy_analysis = self._analyze_strategies()
            
            # Análisis de tamaños y distribución
            size_analysis = self._analyze_sizes()
            
            # Análisis de patrones de acceso
            access_analysis = self._analyze_access_patterns()
            
            return {
                'metricas_basicas': self.metrics.to_dict(),
                'analisis_estrategias': strategy_analysis,
                'analisis_tamanos': size_analysis,
                'analisis_acceso': access_analysis,
                'configuracion': {
                    'tamano_maximo': self.max_size,
                    'items_maximos': self.max_items,
                    'estrategia_por_defecto': self.default_strategy.value,
                    'persistencia_automatica': self.auto_persist,
                    'intervalo_limpieza': self.cleanup_interval
                },
                'estado_sistema': {
                    'tiempo_funcionamiento': (datetime.now() - self._last_optimization).total_seconds(),
                    'elementos_activos': len(self.cache),
                    'memoria_disponible': self.max_size - self.metrics.total_size,
                    'eficiencia': self._calculate_efficiency()
                }
            }
    
    def _analyze_strategies(self) -> Dict[str, Any]:
        """Analiza la distribución y eficiencia de estrategias"""
        strategy_counts = {}
        strategy_performance = {}
        
        for item in self.cache.values():
            strategy = item.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'total_accesos': 0,
                    'tamano_promedio': 0,
                    'elementos': []
                }
            
            strategy_performance[strategy]['total_accesos'] += item.access_count
            strategy_performance[strategy]['elementos'].append(item.size)
        
        # Calcular promedios
        for strategy, data in strategy_performance.items():
            if data['elementos']:
                data['tamano_promedio'] = sum(data['elementos']) / len(data['elementos'])
            del data['elementos']  # Remover lista temporal
        
        return {
            'distribucion': strategy_counts,
            'rendimiento': strategy_performance
        }
    
    def _analyze_sizes(self) -> Dict[str, Any]:
        """Analiza la distribución de tamaños de elementos"""
        if not self.cache:
            return {'elementos': 0}
        
        sizes = [item.size for item in self.cache.values()]
        
        return {
            'elementos': len(sizes),
            'tamano_promedio': sum(sizes) / len(sizes),
            'tamano_minimo': min(sizes),
            'tamano_maximo': max(sizes),
            'tamano_total': sum(sizes),
            'mediana': sorted(sizes)[len(sizes) // 2]
        }
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analiza patrones de acceso y uso"""
        if not self.cache:
            return {'sin_datos': True}
        
        access_counts = [item.access_count for item in self.cache.values()]
        recent_items = sum(1 for item in self.cache.values() 
                          if (datetime.now() - item.last_accessed).total_seconds() < 3600)
        
        return {
            'accesos_promedio': sum(access_counts) / len(access_counts),
            'accesos_total': sum(access_counts),
            'elementos_recientes': recent_items,  # Accedidos en última hora
            'elementos_populares': sum(1 for count in access_counts if count > 5),
            'distribucion_acceso': {
                'minimo': min(access_counts),
                'maximo': max(access_counts),
                'mediana': sorted(access_counts)[len(access_counts) // 2]
            }
        }
    
    def _calculate_efficiency(self) -> float:
        """
        Calcula un score de eficiencia general del caché
        
        Returns:
            Score de eficiencia de 0 a 100
        """
        if self.metrics.hits + self.metrics.misses == 0:
            return 100.0
        
        # Factores de eficiencia
        hit_rate_score = self.metrics.hit_rate
        memory_efficiency = max(0, 100 - self.metrics.memory_usage)
        access_speed_score = max(0, 100 - min(self.metrics.avg_access_time, 100))
        
        # Score ponderado
        efficiency = (
            hit_rate_score * 0.5 +      # 50% tasa de aciertos
            memory_efficiency * 0.3 +    # 30% eficiencia de memoria
            access_speed_score * 0.2     # 20% velocidad de acceso
        )
        
        return round(efficiency, 2)
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Ejecuta optimización inteligente del caché basada en análisis de uso
        
        Returns:
            Informe de optimización con recomendaciones y acciones tomadas
        """
        try:
            with self.lock:
                optimization_start = time.time()
                actions_taken = []
                recommendations = []
                
                # Análisis de rendimiento actual
                current_stats = self.get_comprehensive_stats()
                
                # Optimización 1: Limpieza inteligente
                expired_count = sum(1 for item in self.cache.values() if item.is_expired())
                if expired_count > 0:
                    self._unified_maintenance()
                    actions_taken.append(f"Eliminados {expired_count} elementos expirados")
                
                # Optimización 2: Análisis de estrategias
                strategy_analysis = current_stats['analisis_estrategias']
                if 'adaptive' in strategy_analysis['distribucion']:
                    adaptive_count = strategy_analysis['distribucion']['adaptive']
                    total_items = sum(strategy_analysis['distribucion'].values())
                    
                    if adaptive_count / total_items < 0.3:
                        recommendations.append(
                            "Considera usar más elementos con estrategia ADAPTIVE para mejor rendimiento"
                        )
                
                # Optimización 3: Gestión de memoria
                if current_stats['metricas_basicas']['memory_usage'] > 85:
                    # Liberar espacio preventivo
                    freed = self._intelligent_eviction(self.max_size * 0.1)  # Liberar 10%
                    if freed > 0:
                        actions_taken.append(f"Liberados {freed} bytes preventivamente")
                
                # Optimización 4: Recomendaciones basadas en patrones
                hit_rate = current_stats['metricas_basicas']['hit_rate']
                if hit_rate < 70:
                    recommendations.append(
                        "Tasa de aciertos baja. Considera aumentar el tamaño del caché"
                    )
                elif hit_rate > 95:
                    recommendations.append(
                        "Excelente tasa de aciertos. El tamaño actual es óptimo"
                    )
                
                # Actualizar timestamp de optimización
                self._last_optimization = datetime.now()
                
                optimization_time = (time.time() - optimization_start) * 1000
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'tiempo_optimizacion_ms': round(optimization_time, 2),
                    'acciones_ejecutadas': actions_taken,
                    'recomendaciones': recommendations,
                    'metricas_post_optimizacion': {
                        'elementos_totales': len(self.cache),
                        'uso_memoria': self.metrics.memory_usage,
                        'tasa_aciertos': self.metrics.hit_rate,
                        'score_eficiencia': self._calculate_efficiency()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error durante optimización: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _save_persistent_data(self) -> None:
        """
        Guarda datos persistentes del caché de forma segura
        Utiliza escritura atómica para evitar corrupción
        """
        if not self.auto_persist:
            return
            
        try:
            # Filtrar elementos válidos (no expirados)
            valid_items = {
                key: item for key, item in self.cache.items()
                if not item.is_expired()
            }
            
            # Escritura atómica usando archivo temporal
            temp_file = self.persist_file.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                pickle.dump(valid_items, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Mover archivo temporal al definitivo
            temp_file.replace(self.persist_file)
            
            # Guardar métricas
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            
            logger.debug(f"Datos persistentes guardados: {len(valid_items)} elementos")
            
        except Exception as e:
            logger.error(f"Error guardando datos persistentes: {e}")
    
    def _load_persistent_data(self) -> None:
        """
        Carga datos persistentes del caché con validación
        """
        try:
            # Cargar elementos del caché
            if self.persist_file.exists():
                with open(self.persist_file, 'rb') as f:
                    loaded_items = pickle.load(f)
                
                # Validar y cargar elementos no expirados
                loaded_count = 0
                for key, item in loaded_items.items():
                    if isinstance(item, CacheItem) and not item.is_expired():
                        self.cache[key] = item
                        loaded_count += 1
                
                logger.info(f"Cargados {loaded_count} elementos del caché persistente")
            
            # Cargar métricas
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    
                # Restaurar métricas acumulativas
                self.metrics.hits = metrics_data.get('hits', 0)
                self.metrics.misses = metrics_data.get('misses', 0)
                self.metrics.evictions = metrics_data.get('evictions', 0)
                self._update_hit_rate()
                
                logger.info("Métricas del caché restauradas")
            
        except Exception as e:
            logger.error(f"Error cargando datos persistentes: {e}")
            # En caso de error, partir con caché limpio
            self.cache.clear()
            self.metrics = CacheMetrics()
    
    def shutdown(self) -> None:
        """
        Cierra el sistema de caché de forma ordenada
        Guarda datos pendientes y detiene tareas de fondo
        """
        try:
            logger.info("Iniciando cierre ordenado del sistema de caché")
            
            # Señalar cierre a tareas de fondo
            self._shutdown = True
            
            # Guardar datos finales
            if self.auto_persist:
                self._save_persistent_data()
            
            # Obtener estadísticas finales
            final_stats = self.get_comprehensive_stats()
            
            logger.info(f"Sistema de caché cerrado - Estadísticas finales: "
                       f"Items: {final_stats['metricas_basicas']['total_items']}, "
                       f"Hit Rate: {final_stats['metricas_basicas']['hit_rate']:.1f}%, "
                       f"Eficiencia: {final_stats['estado_sistema']['eficiencia']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error durante cierre del caché: {e}")

# Instancia global del caché inteligente
intelligent_cache = IntelligentCache()

# Función de conveniencia para acceso global
def get_cache() -> IntelligentCache:
    """
    Obtiene la instancia global del caché inteligente
    
    Returns:
        Instancia del sistema de caché
    """
    return intelligent_cache
