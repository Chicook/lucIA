"""
Módulo 1: Sistema de Memoria Persistente
Versión: 0.6.0
Funcionalidad: Gestión de memoria a largo plazo, aprendizaje y persistencia de datos
"""

import json
import os
import sqlite3
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging

logger = logging.getLogger('LucIA_Memory')

class MemorySystem:
    """
    Sistema de memoria persistente para LucIA.
    Gestiona almacenamiento, recuperación y organización de información.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.memory_db_path = "data/memory/lucia_memory.db"
        self.cache_path = "cache/memory_cache.pkl"
        self.max_memory_size = 1000000
        self.memory_usage = 0
        self.access_patterns = {}
        self.learning_cycles = 0
        
        # Inicializar base de datos
        self._init_database()
        self._load_cache()
        
        logger.info("Sistema de memoria inicializado")
    
    def _init_database(self):
        """Inicializa la base de datos SQLite para memoria persistente"""
        os.makedirs(os.path.dirname(self.memory_db_path), exist_ok=True)
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla principal de memoria
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    tags TEXT
                )
            ''')
            
            # Tabla de asociaciones entre memorias
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_associations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id_1 INTEGER,
                    memory_id_2 INTEGER,
                    association_strength REAL DEFAULT 0.5,
                    association_type TEXT DEFAULT 'related',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (memory_id_1) REFERENCES memories (id),
                    FOREIGN KEY (memory_id_2) REFERENCES memories (id)
                )
            ''')
            
            # Tabla de patrones de acceso
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success_rate REAL DEFAULT 0.5
                )
            ''')
            
            # Índices para optimización
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories (memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories (importance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories (last_accessed)')
            
            conn.commit()
            logger.info("Base de datos de memoria inicializada")
    
    def _load_cache(self):
        """Carga el caché de memoria desde disco"""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.memory_usage = cache_data.get('memory_usage', 0)
                    self.access_patterns = cache_data.get('access_patterns', {})
                    self.learning_cycles = cache_data.get('learning_cycles', 0)
                logger.info("Caché de memoria cargado")
        except Exception as e:
            logger.error(f"Error cargando caché: {e}")
    
    def _save_cache(self):
        """Guarda el caché de memoria en disco"""
        try:
            cache_data = {
                'memory_usage': self.memory_usage,
                'access_patterns': self.access_patterns,
                'learning_cycles': self.learning_cycles,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug("Caché de memoria guardado")
        except Exception as e:
            logger.error(f"Error guardando caché: {e}")
    
    async def store_memory(self, content: Any, memory_type: str = "general", 
                          importance: float = 0.5, metadata: Dict = None, 
                          tags: List[str] = None) -> str:
        """
        Almacena una nueva memoria en el sistema
        
        Args:
            content: Contenido a almacenar
            memory_type: Tipo de memoria (fact, experience, skill, etc.)
            importance: Puntuación de importancia (0.0 - 1.0)
            metadata: Metadatos adicionales
            tags: Etiquetas para categorización
        
        Returns:
            Hash único de la memoria almacenada
        """
        try:
            # Serializar contenido
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)
            
            # Generar hash único
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()
            
            # Verificar si ya existe
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM memories WHERE content_hash = ?",
                    (content_hash,)
                )
                
                if cursor.fetchone():
                    logger.debug("Memoria ya existe, actualizando acceso")
                    cursor.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE content_hash = ?",
                        (content_hash,)
                    )
                    return content_hash
                
                # Insertar nueva memoria
                cursor.execute('''
                    INSERT INTO memories (content_hash, content, memory_type, importance_score, metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    content_hash,
                    content_str,
                    memory_type,
                    importance,
                    json.dumps(metadata or {}),
                    json.dumps(tags or [])
                ))
                
                conn.commit()
                self.memory_usage += len(content_str)
                self.learning_cycles += 1
                
                logger.info(f"Memoria almacenada: {content_hash[:8]}...")
                return content_hash
                
        except Exception as e:
            logger.error(f"Error almacenando memoria: {e}")
            return None
    
    async def retrieve_memory(self, query: str, memory_type: str = None, 
                            limit: int = 10, min_importance: float = 0.0) -> List[Dict]:
        """
        Recupera memorias basadas en una consulta
        
        Args:
            query: Consulta de búsqueda
            memory_type: Filtrar por tipo de memoria
            limit: Número máximo de resultados
            min_importance: Importancia mínima requerida
        
        Returns:
            Lista de memorias encontradas
        """
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Construir consulta SQL
                sql = '''
                    SELECT id, content, memory_type, importance_score, 
                           access_count, last_accessed, metadata, tags
                    FROM memories 
                    WHERE content LIKE ? AND importance_score >= ?
                '''
                params = [f'%{query}%', min_importance]
                
                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type)
                
                sql += " ORDER BY importance_score DESC, access_count DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                # Convertir a diccionarios
                memories = []
                for row in results:
                    memory = {
                        'id': row[0],
                        'content': row[1],
                        'memory_type': row[2],
                        'importance_score': row[3],
                        'access_count': row[4],
                        'last_accessed': row[5],
                        'metadata': json.loads(row[6] or '{}'),
                        'tags': json.loads(row[7] or '[]')
                    }
                    memories.append(memory)
                    
                    # Actualizar contador de acceso
                    cursor.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                        (row[0],)
                    )
                
                conn.commit()
                logger.info(f"Recuperadas {len(memories)} memorias para consulta: {query[:20]}...")
                return memories
                
        except Exception as e:
            logger.error(f"Error recuperando memorias: {e}")
            return []
    
    async def create_association(self, memory_id_1: int, memory_id_2: int, 
                               strength: float = 0.5, assoc_type: str = "related"):
        """Crea una asociación entre dos memorias"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO memory_associations 
                    (memory_id_1, memory_id_2, association_strength, association_type)
                    VALUES (?, ?, ?, ?)
                ''', (memory_id_1, memory_id_2, strength, assoc_type))
                conn.commit()
                logger.debug(f"Asociación creada entre memorias {memory_id_1} y {memory_id_2}")
        except Exception as e:
            logger.error(f"Error creando asociación: {e}")
    
    async def get_related_memories(self, memory_id: int, limit: int = 5) -> List[Dict]:
        """Obtiene memorias relacionadas a través de asociaciones"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT m.id, m.content, m.memory_type, m.importance_score,
                           ma.association_strength, ma.association_type
                    FROM memories m
                    JOIN memory_associations ma ON (
                        (ma.memory_id_1 = ? AND m.id = ma.memory_id_2) OR
                        (ma.memory_id_2 = ? AND m.id = ma.memory_id_1)
                    )
                    ORDER BY ma.association_strength DESC
                    LIMIT ?
                ''', (memory_id, memory_id, limit))
                
                results = cursor.fetchall()
                related = []
                for row in results:
                    related.append({
                        'id': row[0],
                        'content': row[1],
                        'memory_type': row[2],
                        'importance_score': row[3],
                        'association_strength': row[4],
                        'association_type': row[5]
                    })
                
                logger.debug(f"Encontradas {len(related)} memorias relacionadas")
                return related
                
        except Exception as e:
            logger.error(f"Error obteniendo memorias relacionadas: {e}")
            return []
    
    async def update_importance(self, memory_id: int, new_importance: float):
        """Actualiza la puntuación de importancia de una memoria"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memories SET importance_score = ? WHERE id = ?",
                    (new_importance, memory_id)
                )
                conn.commit()
                logger.debug(f"Importancia actualizada para memoria {memory_id}")
        except Exception as e:
            logger.error(f"Error actualizando importancia: {e}")
    
    async def cleanup_old_memories(self, days_threshold: int = 30, 
                                 min_importance: float = 0.1):
        """Limpia memorias antiguas y poco importantes"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Contar memorias a eliminar
                cursor.execute('''
                    SELECT COUNT(*) FROM memories 
                    WHERE last_accessed < ? AND importance_score < ?
                ''', (cutoff_date, min_importance))
                
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Eliminar memorias antiguas
                    cursor.execute('''
                        DELETE FROM memories 
                        WHERE last_accessed < ? AND importance_score < ?
                    ''', (cutoff_date, min_importance))
                    
                    conn.commit()
                    logger.info(f"Eliminadas {count} memorias antiguas")
                else:
                    logger.debug("No hay memorias antiguas para eliminar")
                    
        except Exception as e:
            logger.error(f"Error en limpieza de memorias: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de memoria"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Estadísticas generales
                cursor.execute("SELECT COUNT(*) FROM memories")
                total_memories = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(importance_score) FROM memories")
                avg_importance = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM memory_associations")
                total_associations = cursor.fetchone()[0]
                
                # Memorias por tipo
                cursor.execute('''
                    SELECT memory_type, COUNT(*) 
                    FROM memories 
                    GROUP BY memory_type
                ''')
                memories_by_type = dict(cursor.fetchall())
                
                return {
                    'total_memories': total_memories,
                    'average_importance': round(avg_importance, 3),
                    'total_associations': total_associations,
                    'memories_by_type': memories_by_type,
                    'memory_usage_bytes': self.memory_usage,
                    'learning_cycles': self.learning_cycles
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    async def save_state(self):
        """Guarda el estado actual del sistema de memoria"""
        self._save_cache()
        logger.info("Estado del sistema de memoria guardado")

# Instancia global del sistema de memoria
memory_system = MemorySystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de memoria"""
    global memory_system
    memory_system.core_engine = core_engine
    core_engine.memory_system = memory_system
    logger.info("Módulo de memoria inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de memoria"""
    if isinstance(input_data, str):
        # Buscar memorias relacionadas
        memories = await memory_system.retrieve_memory(input_data)
        return {'query': input_data, 'memories': memories}
    elif isinstance(input_data, dict) and 'store' in input_data:
        # Almacenar nueva memoria
        memory_id = await memory_system.store_memory(
            input_data['content'],
            input_data.get('type', 'general'),
            input_data.get('importance', 0.5),
            input_data.get('metadata'),
            input_data.get('tags')
        )
        return {'stored': True, 'memory_id': memory_id}
    else:
        return input_data

def run_modulo1():
    """Función de compatibilidad con el sistema anterior"""
    print("🧠 Módulo 1: Sistema de Memoria Persistente")
    print("   - Gestión de memoria a largo plazo")
    print("   - Almacenamiento y recuperación de información")
    print("   - Asociaciones entre memorias")
    print("   - Limpieza automática de datos antiguos")
    print("   ✅ Módulo inicializado correctamente")