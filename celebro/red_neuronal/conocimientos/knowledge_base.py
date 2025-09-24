"""
Base de Conocimientos para @red_neuronal
Versión: 0.7.0 - Optimizada
Sistema de gestión de conocimientos para aprendizaje profundo en ciberseguridad
"""

import json
import sqlite3
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

from .security_topics import SecurityTopics, SecurityTopic
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningSession:
    """Sesión de aprendizaje optimizada"""
    id: str
    topic_id: str
    prompts_used: List[str]
    responses: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    status: str = "active"

@dataclass
class KnowledgeItem:
    """Elemento de conocimiento optimizado"""
    id: str
    topic_id: str
    content: str
    knowledge_type: str
    difficulty_level: str
    tags: List[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    confidence_score: float = 0.0

class KnowledgeBase:
    """Base de conocimientos optimizada para el sistema de aprendizaje profundo"""
    
    def __init__(self, db_path: str = "celebro/red_neuronal/conocimientos/knowledge.db", 
                 pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=pool_size)
        
        # Cache para consultas frecuentes
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Componentes lazy-loaded
        self._security_topics = None
        self._prompt_generator = None
        
        self._initialize_database()
        self._initialize_connection_pool()
    
    @property
    def security_topics(self):
        """Lazy loading para security_topics"""
        if self._security_topics is None:
            self._security_topics = SecurityTopics()
        return self._security_topics
    
    @property
    def prompt_generator(self):
        """Lazy loading para prompt_generator"""
        if self._prompt_generator is None:
            self._prompt_generator = PromptGenerator()
        return self._prompt_generator
    
    def _initialize_connection_pool(self):
        """Inicializa el pool de conexiones"""
        with self._pool_lock:
            for _ in range(self.pool_size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")  # Mejor rendimiento para escrituras concurrentes
                conn.execute("PRAGMA synchronous=NORMAL")  # Balance entre seguridad y rendimiento
                conn.execute("PRAGMA cache_size=10000")  # Cache más grande
                conn.execute("PRAGMA temp_store=memory")  # Usar memoria para tablas temporales
                self._connection_pool.append(conn)
    
    @contextmanager
    def _get_connection(self):
        """Context manager para obtener conexiones del pool"""
        with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
            else:
                # Si no hay conexiones disponibles, crear una nueva
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=memory")
        
        try:
            yield conn
        finally:
            with self._pool_lock:
                if len(self._connection_pool) < self.pool_size:
                    self._connection_pool.append(conn)
                else:
                    conn.close()
    
    def _initialize_database(self):
        """Inicializa la base de datos SQLite con optimizaciones"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Crear tablas con índices optimizados
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        id TEXT PRIMARY KEY,
                        topic_id TEXT NOT NULL,
                        prompts_used TEXT,
                        responses TEXT,
                        performance_metrics TEXT,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        duration_minutes INTEGER,
                        status TEXT NOT NULL DEFAULT 'active'
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_sessions_topic ON learning_sessions(topic_id);
                    CREATE INDEX IF NOT EXISTS idx_sessions_status ON learning_sessions(status);
                    CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON learning_sessions(start_time);
                    
                    CREATE TABLE IF NOT EXISTS knowledge_items (
                        id TEXT PRIMARY KEY,
                        topic_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        difficulty_level TEXT NOT NULL,
                        tags TEXT,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        confidence_score REAL DEFAULT 0.0
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_items(topic_id);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_items(knowledge_type);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_content ON knowledge_items(content);
                    CREATE INDEX IF NOT EXISTS idx_knowledge_score ON knowledge_items(confidence_score DESC);
                    
                    CREATE TABLE IF NOT EXISTS learning_progress (
                        id TEXT PRIMARY KEY,
                        topic_id TEXT NOT NULL,
                        user_id TEXT,
                        completion_percentage REAL DEFAULT 0.0,
                        mastery_level TEXT DEFAULT 'beginner',
                        last_updated TEXT NOT NULL,
                        total_time_spent INTEGER DEFAULT 0,
                        prompts_completed INTEGER DEFAULT 0,
                        correct_responses INTEGER DEFAULT 0
                    );
                    
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_progress_user_topic ON learning_progress(user_id, topic_id);
                    CREATE INDEX IF NOT EXISTS idx_progress_topic ON learning_progress(topic_id);
                    
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        prompt_id TEXT NOT NULL,
                        response_quality REAL,
                        time_taken INTEGER,
                        difficulty_level TEXT,
                        topic_id TEXT,
                        created_at TEXT NOT NULL
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_metrics_session ON performance_metrics(session_id);
                    CREATE INDEX IF NOT EXISTS idx_metrics_topic ON performance_metrics(topic_id);
                    CREATE INDEX IF NOT EXISTS idx_metrics_quality ON performance_metrics(response_quality);
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise Exception(f"Error inicializando base de datos: {e}")
    
    def create_learning_session(self, topic_id: str, user_id: str = None) -> LearningSession:
        """Crea una nueva sesión de aprendizaje optimizada"""
        session_id = f"session_{topic_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now()
        
        session = LearningSession(
            id=session_id,
            topic_id=topic_id,
            prompts_used=[],
            responses=[],
            performance_metrics={},
            start_time=now
        )
        
        # Usar executor para operaciones de DB no bloqueantes
        self._executor.submit(self._save_session_to_db, session)
        
        return session
    
    def _save_session_to_db(self, session: LearningSession):
        """Guarda la sesión en la base de datos (método auxiliar)"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO learning_sessions 
                    (id, topic_id, prompts_used, responses, performance_metrics, 
                     start_time, end_time, duration_minutes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.id, session.topic_id,
                    json.dumps(session.prompts_used),
                    json.dumps(session.responses),
                    json.dumps(session.performance_metrics),
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.duration_minutes,
                    session.status
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error guardando sesión: {e}")
    
    def batch_update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Actualiza múltiples campos de una sesión en una sola operación"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Construir consulta dinámica
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    if key in ['prompts_used', 'responses', 'performance_metrics']:
                        value = json.dumps(value)
                    elif key in ['start_time', 'end_time'] and isinstance(value, datetime):
                        value = value.isoformat()
                    
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                
                params.append(session_id)
                
                cursor.execute(f"""
                    UPDATE learning_sessions 
                    SET {', '.join(set_clauses)}
                    WHERE id = ?
                """, params)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error en actualización batch: {e}")
            raise Exception(f"Error en actualización batch: {e}")
    
    @lru_cache(maxsize=100)
    def get_cached_learning_progress(self, topic_id: str, user_id: str = None) -> Dict[str, Any]:
        """Obtiene el progreso de aprendizaje con cache"""
        cache_key = f"progress_{topic_id}_{user_id or 'anonymous'}"
        
        with self._cache_lock:
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                # Cache válido por 5 minutos
                if (datetime.now() - cache_entry['timestamp']).seconds < 300:
                    return cache_entry['data']
        
        # Si no está en cache o expiró, consultar DB
        progress = self._get_learning_progress_from_db(topic_id, user_id)
        
        with self._cache_lock:
            self._cache[cache_key] = {
                'data': progress,
                'timestamp': datetime.now()
            }
        
        return progress
    
    def _get_learning_progress_from_db(self, topic_id: str, user_id: str = None) -> Dict[str, Any]:
        """Obtiene el progreso desde la base de datos"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT completion_percentage, mastery_level, total_time_spent,
                           prompts_completed, correct_responses, last_updated
                    FROM learning_progress 
                    WHERE topic_id = ? AND (user_id = ? OR user_id IS NULL)
                    ORDER BY last_updated DESC LIMIT 1
                """, (topic_id, user_id))
                
                result = cursor.fetchone()
                
                if result:
                    completion, mastery, time_spent, prompts_completed, correct_responses, last_updated = result
                    return {
                        "topic_id": topic_id,
                        "completion_percentage": completion,
                        "mastery_level": mastery,
                        "total_time_spent": time_spent,
                        "prompts_completed": prompts_completed,
                        "correct_responses": correct_responses,
                        "accuracy_rate": correct_responses / prompts_completed if prompts_completed > 0 else 0,
                        "last_updated": last_updated
                    }
                else:
                    return {
                        "topic_id": topic_id,
                        "completion_percentage": 0.0,
                        "mastery_level": "beginner",
                        "total_time_spent": 0,
                        "prompts_completed": 0,
                        "correct_responses": 0,
                        "accuracy_rate": 0.0,
                        "last_updated": None
                    }
                    
        except Exception as e:
            logger.error(f"Error obteniendo progreso: {e}")
            raise Exception(f"Error obteniendo progreso de aprendizaje: {e}")
    
    def bulk_insert_knowledge_items(self, items: List[Dict[str, Any]]) -> List[KnowledgeItem]:
        """Inserción masiva de elementos de conocimiento"""
        try:
            knowledge_items = []
            insert_data = []
            
            for item_data in items:
                item_id = f"knowledge_{item_data['topic_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                now = datetime.now()
                
                knowledge_item = KnowledgeItem(
                    id=item_id,
                    topic_id=item_data['topic_id'],
                    content=item_data['content'],
                    knowledge_type=item_data['knowledge_type'],
                    difficulty_level=item_data['difficulty_level'],
                    tags=item_data.get('tags', []),
                    created_at=now,
                    last_accessed=now
                )
                
                knowledge_items.append(knowledge_item)
                insert_data.append((
                    knowledge_item.id, knowledge_item.topic_id, knowledge_item.content,
                    knowledge_item.knowledge_type, knowledge_item.difficulty_level,
                    json.dumps(knowledge_item.tags), knowledge_item.created_at.isoformat(),
                    knowledge_item.last_accessed.isoformat(), knowledge_item.access_count,
                    knowledge_item.confidence_score
                ))
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO knowledge_items 
                    (id, topic_id, content, knowledge_type, difficulty_level, 
                     tags, created_at, last_accessed, access_count, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, insert_data)
                conn.commit()
            
            return knowledge_items
            
        except Exception as e:
            logger.error(f"Error en inserción masiva: {e}")
            raise Exception(f"Error agregando elementos de conocimiento: {e}")
    
    def search_knowledge_optimized(self, query: str, topic_id: str = None, 
                                  knowledge_type: str = None, limit: int = 50) -> List[KnowledgeItem]:
        """Búsqueda optimizada de elementos de conocimiento"""
        cache_key = f"search_{hash(query)}_{topic_id}_{knowledge_type}_{limit}"
        
        with self._cache_lock:
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                if (datetime.now() - cache_entry['timestamp']).seconds < 600:  # Cache por 10 minutos
                    return cache_entry['data']
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Usar FTS si está disponible, sino LIKE optimizado
                where_conditions = []
                params = []
                
                if query:
                    where_conditions.append("content LIKE ?")
                    params.append(f"%{query}%")
                
                if topic_id:
                    where_conditions.append("topic_id = ?")
                    params.append(topic_id)
                
                if knowledge_type:
                    where_conditions.append("knowledge_type = ?")
                    params.append(knowledge_type)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                cursor.execute(f"""
                    SELECT id, topic_id, content, knowledge_type, difficulty_level,
                           tags, created_at, last_accessed, access_count, confidence_score
                    FROM knowledge_items 
                    WHERE {where_clause}
                    ORDER BY confidence_score DESC, access_count DESC
                    LIMIT ?
                """, params + [limit])
                
                results = cursor.fetchall()
                
                knowledge_items = []
                for row in results:
                    item = KnowledgeItem(
                        id=row[0], topic_id=row[1], content=row[2], knowledge_type=row[3],
                        difficulty_level=row[4], tags=json.loads(row[5]) if row[5] else [],
                        created_at=datetime.fromisoformat(row[6]),
                        last_accessed=datetime.fromisoformat(row[7]),
                        access_count=row[8], confidence_score=row[9]
                    )
                    knowledge_items.append(item)
                
                # Actualizar cache
                with self._cache_lock:
                    self._cache[cache_key] = {
                        'data': knowledge_items,
                        'timestamp': datetime.now()
                    }
                
                return knowledge_items
                
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise Exception(f"Error buscando conocimiento: {e}")
    
    @lru_cache(maxsize=50)
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de aprendizaje con cache"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Usar una sola consulta con subconsultas para mejor rendimiento
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM learning_sessions) as total_sessions,
                        (SELECT COUNT(*) FROM learning_sessions WHERE status = 'completed') as completed_sessions,
                        (SELECT COUNT(*) FROM performance_metrics) as total_prompts,
                        (SELECT AVG(response_quality) FROM performance_metrics) as avg_quality,
                        (SELECT COUNT(*) FROM knowledge_items) as total_knowledge_items,
                        (SELECT COUNT(DISTINCT topic_id) FROM knowledge_items) as topics_covered
                """)
                
                result = cursor.fetchone()
                
                total_sessions, completed_sessions, total_prompts, avg_quality, total_knowledge_items, topics_covered = result
                
                return {
                    "total_sessions": total_sessions or 0,
                    "completed_sessions": completed_sessions or 0,
                    "completion_rate": (completed_sessions or 0) / (total_sessions or 1),
                    "total_prompts": total_prompts or 0,
                    "average_quality": avg_quality or 0,
                    "total_knowledge_items": total_knowledge_items or 0,
                    "topics_covered": topics_covered or 0
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            raise Exception(f"Error obteniendo estadísticas: {e}")
    
    def cleanup_old_cache(self, max_age_minutes: int = 60):
        """Limpia entradas antiguas del cache"""
        with self._cache_lock:
            now = datetime.now()
            expired_keys = [
                key for key, value in self._cache.items()
                if (now - value['timestamp']).seconds > max_age_minutes * 60
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def close(self):
        """Cierra todas las conexiones y libera recursos"""
        with self._pool_lock:
            for conn in self._connection_pool:
                conn.close()
            self._connection_pool.clear()
        
        self._executor.shutdown(wait=True)
        
        with self._cache_lock:
            self._cache.clear()
    
    def __del__(self):
        """Destructor para limpiar recursos"""
        try:
            self.close()
        except:
            pass