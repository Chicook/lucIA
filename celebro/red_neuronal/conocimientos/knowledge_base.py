"""
Base de Conocimientos para @red_neuronal
Versión: 0.6.0
Sistema de gestión de conocimientos para aprendizaje profundo en ciberseguridad
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from .security_topics import SecurityTopics, SecurityTopic
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel

@dataclass
class LearningSession:
    """Sesión de aprendizaje"""
    id: str
    topic_id: str
    prompts_used: List[str]
    responses: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime]
    duration_minutes: Optional[int]
    status: str  # "active", "completed", "paused"

@dataclass
class KnowledgeItem:
    """Elemento de conocimiento"""
    id: str
    topic_id: str
    content: str
    knowledge_type: str  # "concept", "example", "code", "best_practice"
    difficulty_level: str
    tags: List[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    confidence_score: float

class KnowledgeBase:
    """Base de conocimientos para el sistema de aprendizaje profundo"""
    
    def __init__(self, db_path: str = "celebro/red_neuronal/conocimientos/knowledge.db"):
        self.db_path = db_path
        self.security_topics = SecurityTopics()
        self.prompt_generator = PromptGenerator()
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa la base de datos SQLite"""
        try:
            # Crear directorio si no existe
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabla de sesiones de aprendizaje
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_sessions (
                        id TEXT PRIMARY KEY,
                        topic_id TEXT NOT NULL,
                        prompts_used TEXT,
                        responses TEXT,
                        performance_metrics TEXT,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        duration_minutes INTEGER,
                        status TEXT NOT NULL
                    )
                """)
                
                # Tabla de elementos de conocimiento
                cursor.execute("""
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
                    )
                """)
                
                # Tabla de progreso de aprendizaje
                cursor.execute("""
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
                    )
                """)
                
                # Tabla de métricas de rendimiento
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        prompt_id TEXT NOT NULL,
                        response_quality REAL,
                        time_taken INTEGER,
                        difficulty_level TEXT,
                        topic_id TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            raise Exception(f"Error inicializando base de datos: {e}")
    
    def create_learning_session(self, topic_id: str, user_id: str = None) -> LearningSession:
        """Crea una nueva sesión de aprendizaje"""
        try:
            session_id = f"session_{topic_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            session = LearningSession(
                id=session_id,
                topic_id=topic_id,
                prompts_used=[],
                responses=[],
                performance_metrics={},
                start_time=datetime.now(),
                end_time=None,
                duration_minutes=None,
                status="active"
            )
            
            # Guardar en base de datos
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO learning_sessions 
                    (id, topic_id, prompts_used, responses, performance_metrics, 
                     start_time, end_time, duration_minutes, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.id,
                    session.topic_id,
                    json.dumps(session.prompts_used),
                    json.dumps(session.responses),
                    json.dumps(session.performance_metrics),
                    session.start_time.isoformat(),
                    session.end_time.isoformat() if session.end_time else None,
                    session.duration_minutes,
                    session.status
                ))
                conn.commit()
            
            return session
            
        except Exception as e:
            raise Exception(f"Error creando sesión de aprendizaje: {e}")
    
    def add_prompt_to_session(self, session_id: str, prompt: LearningPrompt) -> None:
        """Agrega un prompt a una sesión de aprendizaje"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener sesión actual
                cursor.execute("SELECT prompts_used FROM learning_sessions WHERE id = ?", (session_id,))
                result = cursor.fetchone()
                
                if result:
                    prompts_used = json.loads(result[0])
                    prompts_used.append(prompt.id)
                    
                    # Actualizar sesión
                    cursor.execute("""
                        UPDATE learning_sessions 
                        SET prompts_used = ? 
                        WHERE id = ?
                    """, (json.dumps(prompts_used), session_id))
                    conn.commit()
                
        except Exception as e:
            raise Exception(f"Error agregando prompt a sesión: {e}")
    
    def record_response(self, session_id: str, prompt_id: str, response: str, 
                       quality_score: float, time_taken: int) -> None:
        """Registra una respuesta en una sesión de aprendizaje"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener sesión actual
                cursor.execute("SELECT responses FROM learning_sessions WHERE id = ?", (session_id,))
                result = cursor.fetchone()
                
                if result:
                    responses = json.loads(result[0])
                    response_data = {
                        "prompt_id": prompt_id,
                        "response": response,
                        "quality_score": quality_score,
                        "time_taken": time_taken,
                        "timestamp": datetime.now().isoformat()
                    }
                    responses.append(response_data)
                    
                    # Actualizar sesión
                    cursor.execute("""
                        UPDATE learning_sessions 
                        SET responses = ? 
                        WHERE id = ?
                    """, (json.dumps(responses), session_id))
                    
                    # Registrar métrica de rendimiento
                    cursor.execute("""
                        INSERT INTO performance_metrics 
                        (id, session_id, prompt_id, response_quality, time_taken, 
                         difficulty_level, topic_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"metric_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        session_id,
                        prompt_id,
                        quality_score,
                        time_taken,
                        "medium",  # TODO: Obtener del prompt
                        "security",  # TODO: Obtener del prompt
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                
        except Exception as e:
            raise Exception(f"Error registrando respuesta: {e}")
    
    def complete_session(self, session_id: str) -> LearningSession:
        """Completa una sesión de aprendizaje"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener sesión actual
                cursor.execute("""
                    SELECT topic_id, prompts_used, responses, performance_metrics, 
                           start_time, status
                    FROM learning_sessions WHERE id = ?
                """, (session_id,))
                result = cursor.fetchone()
                
                if result:
                    topic_id, prompts_used, responses, performance_metrics, start_time, status = result
                    
                    if status == "active":
                        end_time = datetime.now()
                        start_dt = datetime.fromisoformat(start_time)
                        duration = int((end_time - start_dt).total_seconds() / 60)
                        
                        # Calcular métricas de rendimiento
                        responses_data = json.loads(responses) if responses else []
                        if responses_data:
                            avg_quality = sum(r.get('quality_score', 0) for r in responses_data) / len(responses_data)
                            avg_time = sum(r.get('time_taken', 0) for r in responses_data) / len(responses_data)
                            
                            performance_metrics = {
                                "average_quality": avg_quality,
                                "average_time": avg_time,
                                "total_responses": len(responses_data),
                                "completion_rate": len(responses_data) / len(json.loads(prompts_used)) if prompts_used else 0
                            }
                        
                        # Actualizar sesión
                        cursor.execute("""
                            UPDATE learning_sessions 
                            SET end_time = ?, duration_minutes = ?, 
                                performance_metrics = ?, status = 'completed'
                            WHERE id = ?
                        """, (
                            end_time.isoformat(),
                            duration,
                            json.dumps(performance_metrics),
                            session_id
                        ))
                        
                        conn.commit()
                        
                        # Crear objeto de sesión
                        session = LearningSession(
                            id=session_id,
                            topic_id=topic_id,
                            prompts_used=json.loads(prompts_used) if prompts_used else [],
                            responses=responses_data,
                            performance_metrics=performance_metrics,
                            start_time=start_dt,
                            end_time=end_time,
                            duration_minutes=duration,
                            status="completed"
                        )
                        
                        return session
                
        except Exception as e:
            raise Exception(f"Error completando sesión: {e}")
    
    def get_learning_progress(self, topic_id: str, user_id: str = None) -> Dict[str, Any]:
        """Obtiene el progreso de aprendizaje para un tema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener progreso del tema
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
            raise Exception(f"Error obteniendo progreso de aprendizaje: {e}")
    
    def update_learning_progress(self, topic_id: str, user_id: str, 
                                completion_percentage: float, mastery_level: str,
                                time_spent: int, prompts_completed: int, 
                                correct_responses: int) -> None:
        """Actualiza el progreso de aprendizaje"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Verificar si existe progreso previo
                cursor.execute("""
                    SELECT id FROM learning_progress 
                    WHERE topic_id = ? AND user_id = ?
                """, (topic_id, user_id))
                result = cursor.fetchone()
                
                if result:
                    # Actualizar progreso existente
                    cursor.execute("""
                        UPDATE learning_progress 
                        SET completion_percentage = ?, mastery_level = ?, 
                            total_time_spent = ?, prompts_completed = ?, 
                            correct_responses = ?, last_updated = ?
                        WHERE topic_id = ? AND user_id = ?
                    """, (
                        completion_percentage, mastery_level, time_spent,
                        prompts_completed, correct_responses, datetime.now().isoformat(),
                        topic_id, user_id
                    ))
                else:
                    # Crear nuevo progreso
                    progress_id = f"progress_{topic_id}_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    cursor.execute("""
                        INSERT INTO learning_progress 
                        (id, topic_id, user_id, completion_percentage, mastery_level,
                         total_time_spent, prompts_completed, correct_responses, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        progress_id, topic_id, user_id, completion_percentage,
                        mastery_level, time_spent, prompts_completed, correct_responses,
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            raise Exception(f"Error actualizando progreso de aprendizaje: {e}")
    
    def add_knowledge_item(self, topic_id: str, content: str, knowledge_type: str,
                          difficulty_level: str, tags: List[str] = None) -> KnowledgeItem:
        """Agrega un elemento de conocimiento"""
        try:
            item_id = f"knowledge_{topic_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            now = datetime.now()
            
            knowledge_item = KnowledgeItem(
                id=item_id,
                topic_id=topic_id,
                content=content,
                knowledge_type=knowledge_type,
                difficulty_level=difficulty_level,
                tags=tags or [],
                created_at=now,
                last_accessed=now,
                access_count=0,
                confidence_score=0.0
            )
            
            # Guardar en base de datos
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO knowledge_items 
                    (id, topic_id, content, knowledge_type, difficulty_level, 
                     tags, created_at, last_accessed, access_count, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge_item.id,
                    knowledge_item.topic_id,
                    knowledge_item.content,
                    knowledge_item.knowledge_type,
                    knowledge_item.difficulty_level,
                    json.dumps(knowledge_item.tags),
                    knowledge_item.created_at.isoformat(),
                    knowledge_item.last_accessed.isoformat(),
                    knowledge_item.access_count,
                    knowledge_item.confidence_score
                ))
                conn.commit()
            
            return knowledge_item
            
        except Exception as e:
            raise Exception(f"Error agregando elemento de conocimiento: {e}")
    
    def search_knowledge(self, query: str, topic_id: str = None, 
                        knowledge_type: str = None) -> List[KnowledgeItem]:
        """Busca elementos de conocimiento"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Construir consulta
                where_conditions = ["content LIKE ?"]
                params = [f"%{query}%"]
                
                if topic_id:
                    where_conditions.append("topic_id = ?")
                    params.append(topic_id)
                
                if knowledge_type:
                    where_conditions.append("knowledge_type = ?")
                    params.append(knowledge_type)
                
                where_clause = " AND ".join(where_conditions)
                
                cursor.execute(f"""
                    SELECT id, topic_id, content, knowledge_type, difficulty_level,
                           tags, created_at, last_accessed, access_count, confidence_score
                    FROM knowledge_items 
                    WHERE {where_clause}
                    ORDER BY confidence_score DESC, access_count DESC
                """, params)
                
                results = cursor.fetchall()
                
                knowledge_items = []
                for row in results:
                    item = KnowledgeItem(
                        id=row[0],
                        topic_id=row[1],
                        content=row[2],
                        knowledge_type=row[3],
                        difficulty_level=row[4],
                        tags=json.loads(row[5]) if row[5] else [],
                        created_at=datetime.fromisoformat(row[6]),
                        last_accessed=datetime.fromisoformat(row[7]),
                        access_count=row[8],
                        confidence_score=row[9]
                    )
                    knowledge_items.append(item)
                
                return knowledge_items
                
        except Exception as e:
            raise Exception(f"Error buscando conocimiento: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de aprendizaje"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Estadísticas de sesiones
                cursor.execute("SELECT COUNT(*) FROM learning_sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM learning_sessions WHERE status = 'completed'")
                completed_sessions = cursor.fetchone()[0]
                
                # Estadísticas de prompts
                cursor.execute("SELECT COUNT(*) FROM performance_metrics")
                total_prompts = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(response_quality) FROM performance_metrics")
                avg_quality = cursor.fetchone()[0] or 0
                
                # Estadísticas de conocimiento
                cursor.execute("SELECT COUNT(*) FROM knowledge_items")
                total_knowledge_items = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT topic_id) FROM knowledge_items")
                topics_covered = cursor.fetchone()[0]
                
                return {
                    "total_sessions": total_sessions,
                    "completed_sessions": completed_sessions,
                    "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
                    "total_prompts": total_prompts,
                    "average_quality": avg_quality,
                    "total_knowledge_items": total_knowledge_items,
                    "topics_covered": topics_covered
                }
                
        except Exception as e:
            raise Exception(f"Error obteniendo estadísticas: {e}")
    
    def export_knowledge_base(self, filepath: str) -> None:
        """Exporta la base de conocimientos a archivo"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener todos los datos
                cursor.execute("SELECT * FROM learning_sessions")
                sessions = cursor.fetchall()
                
                cursor.execute("SELECT * FROM knowledge_items")
                knowledge_items = cursor.fetchall()
                
                cursor.execute("SELECT * FROM learning_progress")
                progress = cursor.fetchall()
                
                cursor.execute("SELECT * FROM performance_metrics")
                metrics = cursor.fetchall()
                
                # Crear estructura de datos
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "learning_sessions": sessions,
                    "knowledge_items": knowledge_items,
                    "learning_progress": progress,
                    "performance_metrics": metrics
                }
                
                # Guardar archivo
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                print(f"Base de conocimientos exportada a: {filepath}")
                
        except Exception as e:
            raise Exception(f"Error exportando base de conocimientos: {e}")
