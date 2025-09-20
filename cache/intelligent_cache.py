#!/usr/bin/env python3
"""
Sistema de Caché Inteligente - LucIA
Versión: 0.6.0
Sistema avanzado de caché con estrategias de invalidación y optimización
"""

import os
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio

logger = logging.getLogger('Intelligent_Cache')

class CacheStrategy(Enum):
    """Estrategias de caché"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptativa basada en uso

class CacheItem:
    """Elemento del caché"""
    def __init__(self, key: str, value: Any, strategy: CacheStrategy = CacheStrategy.LRU):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.strategy = strategy
        self.ttl = None  # Time to live en segundos
        self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calcula el tamaño del elemento en bytes"""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value).encode('utf-8'))
            else:
                return len(pickle.dumps(self.value))
        except:
            return 1024  # Tamaño por defecto
    
    def access(self):
        """Registra un acceso al elemento"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """Verifica si el elemento ha expirado"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def get_score(self) -> float:
        """Calcula el score del elemento para estrategias de reemplazo"""
        if self.strategy == CacheStrategy.LRU:
            return (datetime.now() - self.last_accessed).total_seconds()
        elif self.strategy == CacheStrategy.LFU:
            return self.access_count
        elif self.strategy == CacheStrategy.TTL:
            if self.ttl is None:
                return float('inf')
            remaining = self.ttl - (datetime.now() - self.created_at).total_seconds()
            return remaining
        else:  # ADAPTIVE
            recency = (datetime.now() - self.last_accessed).total_seconds()
            frequency = self.access_count
            return frequency / (recency + 1)  # Evitar división por cero

@dataclass
class CacheStats:
    """Estadísticas del caché"""
    total_items: int = 0
    total_size: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    memory_usage: float = 0.0

class IntelligentCache:
    """
    Sistema de caché inteligente con múltiples estrategias
    """
    
    def __init__(self, max_size: int = 100 * 1024 * 1024,  # 100MB por defecto
                 max_items: int = 1000,
                 default_strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.max_items = max_items
        self.default_strategy = default_strategy
        self.cache: Dict[str, CacheItem] = {}
        self.stats = CacheStats()
        self.lock = threading.RLock()
        
        # Configuración de persistencia
        self.persist_file = "cache/cache_data.pkl"
        self.stats_file = "cache/cache_stats.json"
        
        # Crear directorio si no existe
        os.makedirs("cache", exist_ok=True)
        
        # Cargar datos persistentes
        self._load_persistent_data()
        
        # Iniciar limpieza automática
        self._start_cleanup_task()
        
        logger.info("Sistema de caché inteligente inicializado")
    
    def _start_cleanup_task(self):
        """Inicia la tarea de limpieza automática"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # Limpiar cada 5 minutos
                    self._cleanup_expired()
                    self._save_persistent_data()
                except Exception as e:
                    logger.error(f"Error en limpieza automática: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            strategy: Optional[CacheStrategy] = None) -> bool:
        """
        Almacena un valor en el caché
        
        Args:
            key: Clave del elemento
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
            strategy: Estrategia de caché
        """
        try:
            with self.lock:
                strategy = strategy or self.default_strategy
                
                # Crear elemento del caché
                item = CacheItem(key, value, strategy)
                if ttl:
                    item.ttl = ttl
                
                # Verificar si necesitamos hacer espacio
                if self._needs_eviction(item):
                    self._evict_items(item.size)
                
                # Almacenar el elemento
                self.cache[key] = item
                
                # Actualizar estadísticas
                self.stats.total_items = len(self.cache)
                self.stats.total_size = sum(item.size for item in self.cache.values())
                
                logger.debug(f"Elemento almacenado en caché: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando en caché: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché
        
        Args:
            key: Clave del elemento
            
        Returns:
            Valor del elemento o None si no existe
        """
        try:
            with self.lock:
                if key not in self.cache:
                    self.stats.misses += 1
                    self._update_hit_rate()
                    return None
                
                item = self.cache[key]
                
                # Verificar si ha expirado
                if item.is_expired():
                    del self.cache[key]
                    self.stats.misses += 1
                    self._update_hit_rate()
                    return None
                
                # Registrar acceso
                item.access()
                self.stats.hits += 1
                self._update_hit_rate()
                
                logger.debug(f"Elemento obtenido del caché: {key}")
                return item.value
                
        except Exception as e:
            logger.error(f"Error obteniendo del caché: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Elimina un elemento del caché
        
        Args:
            key: Clave del elemento
            
        Returns:
            True si se eliminó, False si no existía
        """
        try:
            with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    self.stats.total_items = len(self.cache)
                    self.stats.total_size = sum(item.size for item in self.cache.values())
                    logger.debug(f"Elemento eliminado del caché: {key}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando del caché: {e}")
            return False
    
    def clear(self) -> bool:
        """Limpia todo el caché"""
        try:
            with self.lock:
                self.cache.clear()
                self.stats = CacheStats()
                logger.info("Caché limpiado completamente")
                return True
                
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
            return False
    
    def _needs_eviction(self, new_item: CacheItem) -> bool:
        """Verifica si necesitamos eliminar elementos para hacer espacio"""
        current_size = sum(item.size for item in self.cache.values())
        return (current_size + new_item.size > self.max_size or 
                len(self.cache) >= self.max_items)
    
    def _evict_items(self, required_space: int):
        """Elimina elementos para hacer espacio"""
        if not self.cache:
            return
        
        # Ordenar elementos por score (menor score = más candidato para eliminación)
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].get_score()
        )
        
        freed_space = 0
        items_to_remove = []
        
        for key, item in sorted_items:
            items_to_remove.append(key)
            freed_space += item.size
            self.stats.evictions += 1
            
            if freed_space >= required_space:
                break
        
        # Eliminar elementos seleccionados
        for key in items_to_remove:
            del self.cache[key]
        
        # Actualizar estadísticas
        self.stats.total_items = len(self.cache)
        self.stats.total_size = sum(item.size for item in self.cache.values())
        
        logger.debug(f"Evicted {len(items_to_remove)} items, freed {freed_space} bytes")
    
    def _cleanup_expired(self):
        """Limpia elementos expirados"""
        with self.lock:
            expired_keys = [
                key for key, item in self.cache.items()
                if item.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.evictions += 1
            
            if expired_keys:
                self.stats.total_items = len(self.cache)
                self.stats.total_size = sum(item.size for item in self.cache.values())
                logger.debug(f"Cleaned up {len(expired_keys)} expired items")
    
    def _update_hit_rate(self):
        """Actualiza la tasa de aciertos"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def get_stats(self) -> CacheStats:
        """Obtiene estadísticas del caché"""
        with self.lock:
            self.stats.total_items = len(self.cache)
            self.stats.total_size = sum(item.size for item in self.cache.values())
            self.stats.memory_usage = (self.stats.total_size / self.max_size) * 100
            return self.stats
    
    def get_info(self) -> Dict[str, Any]:
        """Obtiene información detallada del caché"""
        stats = self.get_stats()
        
        # Análisis de estrategias
        strategy_counts = {}
        for item in self.cache.values():
            strategy = item.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Análisis de tamaños
        sizes = [item.size for item in self.cache.values()]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        return {
            'stats': {
                'total_items': stats.total_items,
                'total_size': stats.total_size,
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': stats.hit_rate,
                'evictions': stats.evictions,
                'memory_usage': stats.memory_usage
            },
            'strategy_distribution': strategy_counts,
            'size_analysis': {
                'average_size': avg_size,
                'min_size': min(sizes) if sizes else 0,
                'max_size': max(sizes) if sizes else 0
            },
            'configuration': {
                'max_size': self.max_size,
                'max_items': self.max_items,
                'default_strategy': self.default_strategy.value
            }
        }
    
    def _save_persistent_data(self):
        """Guarda datos persistentes del caché"""
        try:
            # Guardar elementos del caché (solo los no expirados)
            valid_items = {
                key: item for key, item in self.cache.items()
                if not item.is_expired()
            }
            
            with open(self.persist_file, 'wb') as f:
                pickle.dump(valid_items, f)
            
            # Guardar estadísticas
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'hits': self.stats.hits,
                    'misses': self.stats.misses,
                    'evictions': self.stats.evictions
                }, f)
            
            logger.debug("Datos persistentes del caché guardados")
            
        except Exception as e:
            logger.error(f"Error guardando datos persistentes: {e}")
    
    def _load_persistent_data(self):
        """Carga datos persistentes del caché"""
        try:
            # Cargar elementos del caché
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'rb') as f:
                    loaded_items = pickle.load(f)
                
                # Verificar que no estén expirados
                current_time = datetime.now()
                for key, item in loaded_items.items():
                    if not item.is_expired():
                        self.cache[key] = item
                
                logger.info(f"Cargados {len(self.cache)} elementos del caché persistente")
            
            # Cargar estadísticas
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    stats_data = json.load(f)
                    self.stats.hits = stats_data.get('hits', 0)
                    self.stats.misses = stats_data.get('misses', 0)
                    self.stats.evictions = stats_data.get('evictions', 0)
                    self._update_hit_rate()
            
        except Exception as e:
            logger.error(f"Error cargando datos persistentes: {e}")
    
    def optimize(self) -> Dict[str, Any]:
        """Optimiza el caché basado en patrones de uso"""
        try:
            with self.lock:
                # Análisis de patrones de acceso
                access_patterns = {}
                for item in self.cache.values():
                    strategy = item.strategy.value
                    if strategy not in access_patterns:
                        access_patterns[strategy] = {
                            'total_accesses': 0,
                            'avg_frequency': 0,
                            'avg_recency': 0
                        }
                    
                    access_patterns[strategy]['total_accesses'] += item.access_count
                
                # Calcular promedios
                for strategy_data in access_patterns.values():
                    if self.cache:
                        strategy_data['avg_frequency'] = strategy_data['total_accesses'] / len(self.cache)
                        strategy_data['avg_recency'] = sum(
                            (datetime.now() - item.last_accessed).total_seconds()
                            for item in self.cache.values()
                        ) / len(self.cache)
                
                # Recomendaciones de optimización
                recommendations = []
                
                if self.stats.hit_rate < 0.7:
                    recommendations.append("Considera aumentar el tamaño del caché o ajustar las estrategias")
                
                if self.stats.evictions > self.stats.hits:
                    recommendations.append("Demasiadas eliminaciones, considera estrategia LFU o aumentar tamaño")
                
                if self.stats.memory_usage > 90:
                    recommendations.append("Uso de memoria alto, considera limpiar elementos antiguos")
                
                return {
                    'access_patterns': access_patterns,
                    'recommendations': recommendations,
                    'optimization_score': min(100, self.stats.hit_rate * 100)
                }
                
        except Exception as e:
            logger.error(f"Error optimizando caché: {e}")
            return {'error': str(e)}

# Instancia global del caché inteligente
intelligent_cache = IntelligentCache()
