"""
Entrenador de Aprendizaje Profundo para @red_neuronal
Versión: 0.7.0
Sistema optimizado de entrenamiento de IA usando prompts educativos de ciberseguridad

Mejoras implementadas:
- Optimización de memoria y procesamiento
- Cacheo de embeddings para mejor rendimiento
- Validación robusta de datos
- Pool de threads para paralelización
- Gestión mejorada de recursos
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import os
import pickle
from pathlib import Path

from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel
from .knowledge_base import KnowledgeBase, LearningSession
from .learning_curriculum import LearningCurriculum, LearningPath, LearningPhase
from ..neural_core import NeuralCore, NetworkConfig, TrainingConfig

# Configuración del logger con formato mejorado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Deep_Learning_Trainer')

@dataclass
class TrainingSession:
    """
    Sesión de entrenamiento de IA con mejores validaciones y metadatos
    
    Attributes:
        id: Identificador único de la sesión
        model_id: ID del modelo asociado
        curriculum_path: Ruta del currículum utilizado
        prompts_used: Lista de IDs de prompts utilizados
        training_data: Datos de entrenamiento procesados
        performance_metrics: Métricas de rendimiento del modelo
        start_time: Tiempo de inicio del entrenamiento
        end_time: Tiempo de finalización (None si está activo)
        status: Estado actual ("active", "completed", "failed", "paused")
        config: Configuración utilizada para el entrenamiento
        checkpoint_path: Ruta donde se guardan los checkpoints
    """
    id: str
    model_id: str
    curriculum_path: str
    prompts_used: List[str] = field(default_factory=list)
    training_data: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "initialized"  # "initialized", "active", "completed", "failed", "paused"
    config: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None

@dataclass
class LearningProgress:
    """
    Progreso de aprendizaje mejorado con más métricas
    
    Attributes:
        topic_id: ID del tema de seguridad
        mastery_level: Nivel de dominio (0.0 - 1.0)
        confidence_score: Puntuación de confianza (0.0 - 1.0)
        prompts_completed: Número de prompts completados
        correct_responses: Número de respuestas correctas
        learning_velocity: Velocidad de aprendizaje
        last_updated: Última actualización del progreso
        difficulty_preference: Dificultad preferida por la IA
    """
    topic_id: str
    mastery_level: float = 0.0
    confidence_score: float = 0.0
    prompts_completed: int = 0
    correct_responses: int = 0
    learning_velocity: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    difficulty_preference: DifficultyLevel = DifficultyLevel.MEDIO

class DeepLearningTrainer:
    """
    Entrenador optimizado de aprendizaje profundo para IA en ciberseguridad
    
    Características principales:
    - Cacheo inteligente de embeddings para mejor rendimiento
    - Procesamiento paralelo de datos
    - Gestión avanzada de memoria
    - Checkpoints automáticos
    - Validación robusta de datos
    """
    
    def __init__(self, cache_dir: str = "./cache", max_workers: int = 4):
        """
        Inicializa el entrenador con configuraciones optimizadas
        
        Args:
            cache_dir: Directorio para cachear embeddings y modelos
            max_workers: Número máximo de threads para procesamiento paralelo
        """
        # Inicialización de componentes principales
        self.security_topics = SecurityTopics()
        self.prompt_generator = PromptGenerator()
        self.knowledge_base = KnowledgeBase()
        self.curriculum = LearningCurriculum()
        self.neural_core = NeuralCore()
        
        # Configuración de caché y optimización
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.max_workers = max_workers
        
        # Estado del entrenador
        self.training_sessions: List[TrainingSession] = []
        self.learning_progress: Dict[str, LearningProgress] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Cargar caché de embeddings si existe
        self._load_embedding_cache()
        
        # Pool de threads para procesamiento paralelo
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Entrenador de Aprendizaje Profundo inicializado (workers: {max_workers})")
    
    def __del__(self):
        """Limpieza de recursos al destruir el objeto"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self._save_embedding_cache()
    
    def create_training_session(self, 
                              curriculum_path: str, 
                              model_config: Optional[Dict[str, Any]] = None,
                              checkpoint_interval: int = 100) -> TrainingSession:
        """
        Crea una nueva sesión de entrenamiento con validaciones mejoradas
        
        Args:
            curriculum_path: Ruta del currículum a utilizar
            model_config: Configuración personalizada del modelo
            checkpoint_interval: Intervalo de epochs para guardar checkpoints
            
        Returns:
            TrainingSession: Nueva sesión de entrenamiento creada
            
        Raises:
            ValueError: Si la configuración es inválida
            RuntimeError: Si hay problemas en la creación del modelo
        """
        try:
            # Validar entrada
            if not curriculum_path or not isinstance(curriculum_path, str):
                raise ValueError("curriculum_path debe ser una cadena válida")
            
            # Generar ID único con timestamp más preciso
            session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Configuración optimizada por defecto
            default_config = {
                "input_size": 512,
                "hidden_layers": [256, 128, 64],
                "output_size": 10,
                "activation": "relu",
                "output_activation": "softmax",
                "learning_rate": 0.001,
                "dropout_rate": 0.3,
                "batch_size": 32,
                "checkpoint_interval": checkpoint_interval,
                "early_stopping_patience": 10,
                "validation_split": 0.2
            }
            
            # Fusionar configuración personalizada con la por defecto
            final_config = {**default_config, **(model_config or {})}
            
            # Validar configuración del modelo
            self._validate_model_config(final_config)
            
            # Crear directorio para checkpoints
            checkpoint_dir = self.cache_dir / "checkpoints" / session_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar red neuronal
            network_config = NetworkConfig(
                input_size=final_config["input_size"],
                hidden_layers=final_config["hidden_layers"],
                output_size=final_config["output_size"],
                activation=final_config["activation"],
                output_activation=final_config["output_activation"],
                learning_rate=final_config["learning_rate"],
                dropout_rate=final_config["dropout_rate"]
            )
            
            # Crear red neuronal
            network = self.neural_core.create_network(network_config)
            
            # Crear sesión de entrenamiento
            session = TrainingSession(
                id=session_id,
                model_id=f"model_{session_id}",
                curriculum_path=curriculum_path,
                config=final_config,
                checkpoint_path=str(checkpoint_dir),
                status="initialized"
            )
            
            self.training_sessions.append(session)
            
            logger.info(f"Sesión de entrenamiento creada: {session_id}")
            logger.debug(f"Configuración aplicada: {final_config}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error creando sesión de entrenamiento: {e}")
            raise RuntimeError(f"Fallo al crear sesión de entrenamiento: {e}")
    
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """
        Valida la configuración del modelo
        
        Args:
            config: Configuración a validar
            
        Raises:
            ValueError: Si la configuración no es válida
        """
        required_keys = ["input_size", "hidden_layers", "output_size", "learning_rate"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Configuración faltante: {key}")
        
        # Validaciones específicas
        if config["input_size"] <= 0:
            raise ValueError("input_size debe ser positivo")
        
        if not isinstance(config["hidden_layers"], list) or not config["hidden_layers"]:
            raise ValueError("hidden_layers debe ser una lista no vacía")
        
        if config["output_size"] <= 0:
            raise ValueError("output_size debe ser positivo")
        
        if not (0 < config["learning_rate"] <= 1):
            raise ValueError("learning_rate debe estar entre 0 y 1")
    
    def generate_training_data(self, 
                             topic_ids: List[str], 
                             num_prompts_per_topic: int = 5,
                             use_parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Genera datos de entrenamiento de forma optimizada y paralela
        
        Args:
            topic_ids: Lista de IDs de temas de seguridad
            num_prompts_per_topic: Número de prompts por tema
            use_parallel: Si usar procesamiento paralelo
            
        Returns:
            List[Dict[str, Any]]: Datos de entrenamiento generados
            
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay errores en la generación
        """
        try:
            # Validar entradas
            if not topic_ids or not all(isinstance(tid, str) for tid in topic_ids):
                raise ValueError("topic_ids debe ser una lista de strings válida")
            
            if num_prompts_per_topic <= 0:
                raise ValueError("num_prompts_per_topic debe ser positivo")
            
            # Filtrar temas válidos
            valid_topics = []
            for topic_id in topic_ids:
                topic = self.security_topics.get_topic(topic_id)
                if topic:
                    valid_topics.append(topic_id)
                else:
                    logger.warning(f"Tema no encontrado: {topic_id}")
            
            if not valid_topics:
                raise ValueError("No se encontraron temas válidos")
            
            logger.info(f"Generando datos para {len(valid_topics)} temas válidos")
            
            # Tipos de prompts a generar
            prompt_types = [PromptType.CONCEPTUAL, PromptType.PRACTICO, PromptType.CODIGO]
            
            if use_parallel:
                # Procesamiento paralelo
                training_data = self._generate_training_data_parallel(
                    valid_topics, num_prompts_per_topic, prompt_types
                )
            else:
                # Procesamiento secuencial
                training_data = self._generate_training_data_sequential(
                    valid_topics, num_prompts_per_topic, prompt_types
                )
            
            logger.info(f"Generados {len(training_data)} ejemplos de entrenamiento")
            return training_data
            
        except Exception as e:
            logger.error(f"Error generando datos de entrenamiento: {e}")
            raise RuntimeError(f"Fallo en generación de datos: {e}")
    
    def _generate_training_data_parallel(self, 
                                       topic_ids: List[str],
                                       num_prompts_per_topic: int,
                                       prompt_types: List[PromptType]) -> List[Dict[str, Any]]:
        """
        Genera datos de entrenamiento usando procesamiento paralelo
        
        Args:
            topic_ids: Lista de IDs de temas
            num_prompts_per_topic: Número de prompts por tema
            prompt_types: Tipos de prompts a generar
            
        Returns:
            List[Dict[str, Any]]: Datos de entrenamiento generados
        """
        training_data = []
        
        # Crear tareas para el pool de threads
        tasks = []
        for topic_id in topic_ids:
            for i in range(num_prompts_per_topic):
                for prompt_type in prompt_types:
                    task = self.thread_pool.submit(
                        self._generate_single_prompt_data,
                        topic_id, prompt_type, DifficultyLevel.MEDIO
                    )
                    tasks.append(task)
        
        # Recopilar resultados
        for future in as_completed(tasks):
            try:
                result = future.result(timeout=30)  # Timeout de 30 segundos
                if result:
                    training_data.append(result)
            except Exception as e:
                logger.warning(f"Error en tarea paralela: {e}")
        
        return training_data
    
    def _generate_training_data_sequential(self,
                                         topic_ids: List[str],
                                         num_prompts_per_topic: int,
                                         prompt_types: List[PromptType]) -> List[Dict[str, Any]]:
        """
        Genera datos de entrenamiento de forma secuencial
        
        Args:
            topic_ids: Lista de IDs de temas
            num_prompts_per_topic: Número de prompts por tema
            prompt_types: Tipos de prompts a generar
            
        Returns:
            List[Dict[str, Any]]: Datos de entrenamiento generados
        """
        training_data = []
        
        for topic_id in topic_ids:
            for i in range(num_prompts_per_topic):
                for prompt_type in prompt_types:
                    try:
                        result = self._generate_single_prompt_data(
                            topic_id, prompt_type, DifficultyLevel.MEDIO
                        )
                        if result:
                            training_data.append(result)
                    except Exception as e:
                        logger.warning(f"Error generando prompt para {topic_id}: {e}")
        
        return training_data
    
    def _generate_single_prompt_data(self, 
                                   topic_id: str, 
                                   prompt_type: PromptType,
                                   difficulty: DifficultyLevel) -> Optional[Dict[str, Any]]:
        """
        Genera un único ejemplo de datos de entrenamiento
        
        Args:
            topic_id: ID del tema
            prompt_type: Tipo de prompt
            difficulty: Nivel de dificultad
            
        Returns:
            Optional[Dict[str, Any]]: Datos del prompt o None si hay error
        """
        try:
            # Generar prompt
            prompt = self.prompt_generator.generate_prompt(topic_id, prompt_type, difficulty)
            
            # Crear datos de entrenamiento
            training_sample = {
                "prompt_id": prompt.id,
                "topic_id": topic_id,
                "prompt_type": prompt_type.value,
                "input_text": prompt.content,
                "expected_output": prompt.expected_response,
                "learning_objectives": prompt.learning_objectives,
                "keywords": prompt.keywords,
                "difficulty": prompt.difficulty.value,
                "created_at": prompt.created_at.isoformat(),
                "metadata": {
                    "text_length": len(prompt.content),
                    "word_count": len(prompt.content.split()),
                    "keyword_count": len(prompt.keywords)
                }
            }
            
            return training_sample
            
        except Exception as e:
            logger.debug(f"Error generando prompt individual: {e}")
            return None
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Obtiene embedding de caché usando hash del texto
        
        Args:
            text_hash: Hash del texto para búsqueda en caché
            
        Returns:
            Optional[np.ndarray]: Embedding cacheado o None
        """
        return self.embedding_cache.get(text_hash)
    
    def preprocess_training_data(self, 
                               training_data: List[Dict[str, Any]], 
                               use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa datos de entrenamiento con optimizaciones de memoria y caché
        
        Args:
            training_data: Datos de entrenamiento a procesar
            use_cache: Si usar caché de embeddings
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays X (entrada) e y (salida) procesados
            
        Raises:
            ValueError: Si los datos no son válidos
            RuntimeError: Si hay errores en el procesamiento
        """
        try:
            if not training_data:
                raise ValueError("training_data no puede estar vacío")
            
            logger.info(f"Preprocesando {len(training_data)} muestras de entrenamiento")
            
            # Pre-allocar arrays para mejor eficiencia de memoria
            embedding_size = 512
            num_samples = len(training_data)
            X = np.zeros((num_samples, embedding_size), dtype=np.float32)
            
            # Mapeos optimizados
            prompt_type_mapping = self._get_prompt_type_mapping()
            topic_mapping = self._get_topic_mapping()
            
            # Array para etiquetas
            y_labels = []
            
            # Procesar cada muestra
            for i, sample in enumerate(training_data):
                try:
                    # Validar estructura de la muestra
                    required_keys = ["input_text", "prompt_type", "topic_id", "keywords"]
                    if not all(key in sample for key in required_keys):
                        logger.warning(f"Muestra {i} tiene claves faltantes, omitiendo")
                        continue
                    
                    # Extraer datos
                    input_text = sample["input_text"]
                    prompt_type = sample["prompt_type"]
                    topic_id = sample["topic_id"]
                    keywords = sample.get("keywords", [])
                    
                    # Crear hash para caché
                    text_hash = hash(f"{input_text}_{prompt_type}_{topic_id}")
                    
                    # Intentar obtener de caché
                    embedding = None
                    if use_cache:
                        embedding = self._get_cached_embedding(str(text_hash))
                    
                    # Generar embedding si no está en caché
                    if embedding is None:
                        embedding = self._create_optimized_embedding(
                            input_text, prompt_type, topic_id, keywords, 
                            prompt_type_mapping, topic_mapping
                        )
                        
                        # Guardar en caché
                        if use_cache:
                            self.embedding_cache[str(text_hash)] = embedding
                    
                    X[i] = embedding
                    
                    # Mapear etiqueta de salida
                    output_label = prompt_type_mapping.get(prompt_type, 0)
                    y_labels.append(output_label)
                    
                except Exception as e:
                    logger.warning(f"Error procesando muestra {i}: {e}")
                    # Usar embedding por defecto en caso de error
                    X[i] = np.zeros(embedding_size, dtype=np.float32)
                    y_labels.append(0)
            
            # Convertir etiquetas a one-hot encoding
            num_classes = len(prompt_type_mapping)
            y_onehot = np.eye(num_classes, dtype=np.float32)[y_labels]
            
            logger.info(f"Preprocesamiento completado: {X.shape[0]} muestras, {X.shape[1]} características")
            
            return X, y_onehot
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            raise RuntimeError(f"Fallo en preprocesamiento de datos: {e}")
    
    def _get_prompt_type_mapping(self) -> Dict[str, int]:
        """
        Obtiene mapeo de tipos de prompt a índices numéricos
        
        Returns:
            Dict[str, int]: Mapeo de tipos de prompt
        """
        return {
            "conceptual": 0,
            "practico": 1,
            "codigo": 2,
            "caso_estudio": 3,
            "evaluacion": 4,
            "simulacion": 5
        }
    
    def _get_topic_mapping(self) -> Dict[str, int]:
        """
        Obtiene mapeo de temas a índices numéricos con caché
        
        Returns:
            Dict[str, int]: Mapeo de temas
        """
        if not hasattr(self, '_cached_topic_mapping'):
            all_topics = self.security_topics.get_all_topics()
            self._cached_topic_mapping = {topic.id: i for i, topic in enumerate(all_topics)}
        
        return self._cached_topic_mapping
    
    def _create_optimized_embedding(self, 
                                  text: str, 
                                  prompt_type: str, 
                                  topic_id: str, 
                                  keywords: List[str],
                                  prompt_type_mapping: Dict[str, int],
                                  topic_mapping: Dict[str, int]) -> np.ndarray:
        """
        Crea embedding optimizado con mejor representación de características
        
        Args:
            text: Texto de entrada
            prompt_type: Tipo de prompt
            topic_id: ID del tema
            keywords: Lista de palabras clave
            prompt_type_mapping: Mapeo de tipos de prompt
            topic_mapping: Mapeo de temas
            
        Returns:
            np.ndarray: Embedding optimizado
        """
        embedding_size = 512
        
        # Características básicas del texto (normalizadas)
        text_length = min(len(text) / 1000.0, 1.0)  # Limitar a 1.0
        word_count = min(len(text.split()) / 100.0, 1.0)
        keyword_count = min(len([k for k in keywords if k.lower() in text.lower()]) / 10.0, 1.0)
        
        # Características del tipo de prompt (one-hot)
        prompt_features = np.zeros(len(prompt_type_mapping))
        if prompt_type in prompt_type_mapping:
            prompt_features[prompt_type_mapping[prompt_type]] = 1.0
        
        # Características del tema (one-hot)
        topic_features = np.zeros(len(topic_mapping))
        if topic_id in topic_mapping:
            topic_features[topic_mapping[topic_id]] = 1.0
        
        # Características adicionales
        has_code = 1.0 if any(marker in text.lower() for marker in ['def ', 'class ', 'import ', '```']) else 0.0
        has_urls = 1.0 if any(marker in text.lower() for marker in ['http', 'www', '.com', '.org']) else 0.0
        question_count = text.count('?') / 10.0  # Normalizado
        
        # Combinar todas las características
        features = [
            text_length,
            word_count,
            keyword_count,
            has_code,
            has_urls,
            question_count
        ]
        
        # Agregar características categóricas
        features.extend(prompt_features)
        features.extend(topic_features)
        
        # Rellenar o truncar hasta el tamaño deseado
        while len(features) < embedding_size:
            features.append(0.0)
        
        features = features[:embedding_size]
        
        return np.array(features, dtype=np.float32)
    
    def train_model(self, 
                   session_id: str, 
                   training_data: List[Dict[str, Any]], 
                   epochs: int = 50, 
                   batch_size: Optional[int] = None,
                   validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Entrena modelo con mejoras de eficiencia y monitoreo
        
        Args:
            session_id: ID de la sesión de entrenamiento
            training_data: Datos de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño del lote (usa config si es None)
            validation_data: Datos de validación opcionales
            
        Returns:
            Dict[str, Any]: Resultados del entrenamiento
            
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay errores durante el entrenamiento
        """
        try:
            # Validar y obtener sesión
            session = self._get_session(session_id)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            # Validar datos de entrenamiento
            if not training_data:
                raise ValueError("training_data no puede estar vacío")
            
            logger.info(f"Iniciando entrenamiento para sesión {session_id}")
            session.status = "active"
            
            # Usar batch_size de configuración si no se especifica
            if batch_size is None:
                batch_size = session.config.get("batch_size", 32)
            
            # Preprocesar datos con optimizaciones
            X, y = self.preprocess_training_data(training_data, use_cache=True)
            
            # Preprocesar datos de validación si existen
            X_val, y_val = None, None
            if validation_data:
                X_val, y_val = self.preprocess_training_data(validation_data, use_cache=True)
            
            # Configurar entrenamiento
            training_config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=session.config.get("learning_rate", 0.001),
                validation_split=session.config.get("validation_split", 0.2) if X_val is None else 0.0,
                early_stopping=True,
                patience=session.config.get("early_stopping_patience", 10),
                verbose=1
            )
            
            # Entrenar modelo con datos de validación si están disponibles
            if X_val is not None and y_val is not None:
                history = self.neural_core.train(X, y, validation_data=(X_val, y_val), config=training_config)
            else:
                history = self.neural_core.train(X, y, config=training_config)
            
            # Calcular métricas finales
            final_metrics = self._calculate_final_metrics(history)
            
            # Actualizar sesión
            session.training_data = training_data
            session.performance_metrics = final_metrics
            session.end_time = datetime.now()
            session.status = "completed"
            
            # Guardar checkpoint final
            self._save_model_checkpoint(session)
            
            # Guardar caché de embeddings
            self._save_embedding_cache()
            
            logger.info(f"Entrenamiento completado para sesión {session_id}")
            logger.info(f"Métricas finales: {final_metrics}")
            
            return {
                "session_id": session_id,
                "status": "completed",
                "performance_metrics": final_metrics,
                "training_duration": (session.end_time - session.start_time).total_seconds(),
                "total_samples": len(training_data),
                "training_history": [
                    {
                        "epoch": h.epoch,
                        "loss": h.train_loss,
                        "accuracy": h.train_accuracy,
                        "val_loss": h.val_loss,
                        "val_accuracy": h.val_accuracy
                    } for h in history
                ]
            }
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            
            # Marcar sesión como fallida
            session = self._get_session(session_id)
            if session:
                session.status = "failed"
                session.end_time = datetime.now()
            
            raise RuntimeError(f"Fallo en entrenamiento del modelo: {e}")
    
    def _get_session(self, session_id: str) -> Optional[TrainingSession]:
        """
        Obtiene sesión por ID de forma eficiente
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Optional[TrainingSession]: Sesión encontrada o None
        """
        return next((s for s in self.training_sessions if s.id == session_id), None)
    
    def _calculate_final_metrics(self, history: List[Any]) -> Dict[str, float]:
        """
        Calcula métricas finales del entrenamiento
        
        Args:
            history: Historial de entrenamiento
            
        Returns:
            Dict[str, float]: Métricas calculadas
        """
        if not history:
            return {"final_loss": 0.0, "final_accuracy": 0.0, "epochs_completed": 0}
        
        final_epoch = history[-1]
        
        return {
            "final_loss": getattr(final_epoch, 'train_loss', 0.0),
            "final_accuracy": getattr(final_epoch, 'train_accuracy', 0.0),
            "final_val_loss": getattr(final_epoch, 'val_loss', 0.0),
            "final_val_accuracy": getattr(final_epoch, 'val_accuracy', 0.0),
            "epochs_completed": len(history),
            "best_accuracy": max((getattr(h, 'train_accuracy', 0.0) for h in history), default=0.0),
            "best_val_accuracy": max((getattr(h, 'val_accuracy', 0.0) for h in history), default=0.0)
        }
    
    def _save_model_checkpoint(self, session: TrainingSession) -> None:
        """
        Guarda checkpoint del modelo entrenado
        
        Args:
            session: Sesión de entrenamiento
        """
        try:
            if session.checkpoint_path:
                checkpoint_file = Path(session.checkpoint_path) / "final_model.pkl"
                
                # Guardar estado del modelo (simplificado)
                checkpoint_data = {
                    "session_id": session.id,
                    "model_id": session.model_id,
                    "config": session.config,
                    "performance_metrics": session.performance_metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                logger.info(f"Checkpoint guardado: {checkpoint_file}")
                
        except Exception as e:
            logger.warning(f"Error guardando checkpoint: {e}")
    
    def _load_embedding_cache(self) -> None:
        """Carga caché de embeddings desde disco"""
        try:
            if self.embedding_cache_file.exists():
                with open(self.embedding_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Caché de embeddings cargado: {len(self.embedding_cache)} entradas")
        except Exception as e:
            logger.warning(f"Error cargando caché de embeddings: {e}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self) -> None:
        """Guarda caché de embeddings en disco"""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Caché de embeddings guardado: {len(self.embedding_cache)} entradas")
        except Exception as e:
            logger.warning(f"Error guardando caché de embeddings: {e}")
    
    def evaluate_model(self, 
                      session_id: str, 
                      test_data: List[Dict[str, Any]],
                      detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Evalúa modelo con análisis detallado y métricas mejoradas
        
        Args:
            session_id: ID de la sesión
            test_data: Datos de prueba
            detailed_analysis: Si incluir análisis detallado
            
        Returns:
            Dict[str, Any]: Resultados de evaluación
            
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay errores en la evaluación
        """
        try:
            # Validar sesión
            session = self._get_session(session_id)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            if session.status != "completed":
                raise ValueError("El modelo debe estar entrenado antes de la evaluación")
            
            if not test_data:
                raise ValueError("test_data no puede estar vacío")
            
            logger.info(f"Evaluando modelo para sesión {session_id}")
            
            # Preprocesar datos de prueba
            X_test, y_test = self.preprocess_training_data(test_data, use_cache=True)
            
            # Evaluar modelo
            metrics = self.neural_core.evaluate(X_test, y_test)
            
            # Realizar predicciones
            predictions = self.neural_core.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # Calcular métricas básicas
            accuracy = np.mean(predicted_classes == true_classes)
            
            # Calcular métricas avanzadas
            precision = self._calculate_precision(true_classes, predicted_classes)
            recall = self._calculate_recall(true_classes, predicted_classes)
            f1_score = self._calculate_f1_score(precision, recall)
            
            # Resultados básicos
            evaluation_results = {
                "session_id": session_id,
                "test_samples": len(test_data),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "loss": float(metrics.get("loss", 0.0)),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            # Análisis detallado opcional
            if detailed_analysis:
                detailed_metrics = self._perform_detailed_analysis(
                    true_classes, predicted_classes, predictions, test_data
                )
                evaluation_results.update(detailed_metrics)
            
            logger.info(f"Evaluación completada: Accuracy={accuracy:.4f}, F1={f1_score:.4f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error en evaluación del modelo: {e}")
            raise RuntimeError(f"Fallo en evaluación: {e}")
    
    def _perform_detailed_analysis(self, 
                                 true_classes: np.ndarray,
                                 predicted_classes: np.ndarray,
                                 predictions: np.ndarray,
                                 test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Realiza análisis detallado de la evaluación
        
        Args:
            true_classes: Clases verdaderas
            predicted_classes: Clases predichas
            predictions: Probabilidades de predicción
            test_data: Datos de prueba originales
            
        Returns:
            Dict[str, Any]: Métricas detalladas
        """
        try:
            # Matriz de confusión
            num_classes = len(np.unique(true_classes))
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
            
            for true_cls, pred_cls in zip(true_classes, predicted_classes):
                confusion_matrix[true_cls, pred_cls] += 1
            
            # Métricas por clase
            class_metrics = {}
            for cls in range(num_classes):
                class_mask_true = (true_classes == cls)
                class_mask_pred = (predicted_classes == cls)
                
                tp = np.sum(class_mask_true & class_mask_pred)
                fp = np.sum(~class_mask_true & class_mask_pred)
                fn = np.sum(class_mask_true & ~class_mask_pred)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                class_metrics[f"class_{cls}"] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "support": int(np.sum(class_mask_true))
                }
            
            # Confianza de predicciones
            prediction_confidence = np.max(predictions, axis=1)
            
            return {
                "confusion_matrix": confusion_matrix.tolist(),
                "class_metrics": class_metrics,
                "mean_prediction_confidence": float(np.mean(prediction_confidence)),
                "std_prediction_confidence": float(np.std(prediction_confidence)),
                "min_confidence": float(np.min(prediction_confidence)),
                "max_confidence": float(np.max(prediction_confidence))
            }
            
        except Exception as e:
            logger.warning(f"Error en análisis detallado: {e}")
            return {}
    
    def _calculate_precision(self, true_classes: np.ndarray, predicted_classes: np.ndarray) -> float:
        """
        Calcula precisión macro-promedio optimizada
        
        Args:
            true_classes: Clases verdaderas
            predicted_classes: Clases predichas
            
        Returns:
            float: Precisión promedio
        """
        try:
            unique_classes = np.unique(true_classes)
            precisions = []
            
            for cls in unique_classes:
                tp = np.sum((predicted_classes == cls) & (true_classes == cls))
                fp = np.sum((predicted_classes == cls) & (true_classes != cls))
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    precisions.append(precision)
            
            return float(np.mean(precisions)) if precisions else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando precisión: {e}")
            return 0.0
    
    def _calculate_recall(self, true_classes: np.ndarray, predicted_classes: np.ndarray) -> float:
        """
        Calcula recall macro-promedio optimizado
        
        Args:
            true_classes: Clases verdaderas
            predicted_classes: Clases predichas
            
        Returns:
            float: Recall promedio
        """
        try:
            unique_classes = np.unique(true_classes)
            recalls = []
            
            for cls in unique_classes:
                tp = np.sum((predicted_classes == cls) & (true_classes == cls))
                fn = np.sum((predicted_classes != cls) & (true_classes == cls))
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    recalls.append(recall)
            
            return float(np.mean(recalls)) if recalls else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando recall: {e}")
            return 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calcula F1 score con validación
        
        Args:
            precision: Valor de precisión
            recall: Valor de recall
            
        Returns:
            float: F1 score
        """
        try:
            if precision + recall > 0:
                return 2 * (precision * recall) / (precision + recall)
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculando F1 score: {e}")
            return 0.0
    
    def generate_learning_curriculum(self, 
                                   path: LearningPath, 
                                   num_prompts_per_topic: int = 10,
                                   auto_train: bool = False) -> Dict[str, Any]:
        """
        Genera currículum completo con opción de entrenamiento automático
        
        Args:
            path: Ruta de aprendizaje
            num_prompts_per_topic: Número de prompts por tema
            auto_train: Si entrenar automáticamente
            
        Returns:
            Dict[str, Any]: Currículum generado y resultados
        """
        try:
            logger.info(f"Generando currículum para ruta: {path.value}")
            
            # Obtener plan de aprendizaje
            learning_plan = self.curriculum.generate_learning_plan(path)
            
            # Recopilar todos los temas únicos
            all_topics = set()
            for module in self.curriculum.get_learning_path(path):
                all_topics.update(module.topics)
            
            unique_topics = list(all_topics)
            logger.info(f"Temas únicos encontrados: {len(unique_topics)}")
            
            # Generar datos de entrenamiento
            training_data = self.generate_training_data(
                unique_topics, 
                num_prompts_per_topic,
                use_parallel=True
            )
            
            # Crear sesión de entrenamiento
            session = self.create_training_session(path.value)
            
            # Entrenamiento automático opcional
            training_results = None
            if auto_train and training_data:
                try:
                    training_results = self.train_model(
                        session.id, 
                        training_data,
                        epochs=30,
                        batch_size=32
                    )
                except Exception as e:
                    logger.error(f"Error en entrenamiento automático: {e}")
                    training_results = {"error": str(e)}
            
            curriculum_results = {
                "learning_plan": learning_plan,
                "session_id": session.id,
                "total_prompts": len(training_data),
                "topics_covered": unique_topics,
                "training_data_sample": training_data[:5],  # Muestra de datos
                "generation_timestamp": datetime.now().isoformat()
            }
            
            # Agregar resultados de entrenamiento si están disponibles
            if training_results:
                curriculum_results["training_results"] = training_results
            
            logger.info(f"Currículum generado exitosamente: {len(training_data)} prompts")
            return curriculum_results
            
        except Exception as e:
            logger.error(f"Error generando currículum de aprendizaje: {e}")
            raise RuntimeError(f"Fallo en generación de currículum: {e}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas completas y optimizadas de entrenamiento
        
        Returns:
            Dict[str, Any]: Estadísticas detalladas
        """
        try:
            total_sessions = len(self.training_sessions)
            
            # Contar por estado
            status_counts = {}
            for session in self.training_sessions:
                status = session.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Estadísticas de rendimiento
            performance_metrics = []
            training_durations = []
            
            for session in self.training_sessions:
                if session.status == "completed" and session.performance_metrics:
                    performance_metrics.append(session.performance_metrics)
                    
                    if session.end_time and session.start_time:
                        duration = (session.end_time - session.start_time).total_seconds()
                        training_durations.append(duration)
            
            # Calcular promedios
            avg_accuracy = 0.0
            avg_loss = 0.0
            avg_duration = 0.0
            
            if performance_metrics:
                avg_accuracy = np.mean([m.get("final_accuracy", 0) for m in performance_metrics])
                avg_loss = np.mean([m.get("final_loss", 0) for m in performance_metrics])
            
            if training_durations:
                avg_duration = np.mean(training_durations)
            
            # Estadísticas del caché
            cache_stats = {
                "embedding_cache_size": len(self.embedding_cache),
                "cache_hit_ratio": getattr(self._get_cached_embedding, 'cache_info', lambda: None)()
            }
            
            return {
                "session_statistics": {
                    "total_sessions": total_sessions,
                    "status_breakdown": status_counts,
                    "completion_rate": status_counts.get("completed", 0) / total_sessions if total_sessions > 0 else 0.0
                },
                "performance_metrics": {
                    "average_accuracy": float(avg_accuracy),
                    "average_loss": float(avg_loss),
                    "average_training_duration": float(avg_duration),
                    "sessions_analyzed": len(performance_metrics)
                },
                "resource_usage": {
                    "total_prompts_generated": len(getattr(self.prompt_generator, 'generated_prompts', [])),
                    "embedding_cache_size": cache_stats["embedding_cache_size"],
                    "thread_pool_size": self.max_workers
                },
                "system_info": {
                    "cache_directory": str(self.cache_dir),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": f"Fallo al obtener estadísticas: {e}"}
    
    def export_training_data(self, 
                           session_id: str, 
                           filepath: Union[str, Path],
                           include_embeddings: bool = False) -> None:
        """
        Exporta datos de entrenamiento con opciones avanzadas
        
        Args:
            session_id: ID de la sesión
            filepath: Ruta del archivo de exportación
            include_embeddings: Si incluir embeddings en la exportación
            
        Raises:
            ValueError: Si los parámetros no son válidos
            RuntimeError: Si hay errores en la exportación
        """
        try:
            # Validar sesión
            session = self._get_session(session_id)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            # Convertir a Path para mejor manejo
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos de exportación
            export_data = {
                "metadata": {
                    "session_id": session.id,
                    "model_id": session.model_id,
                    "curriculum_path": session.curriculum_path,
                    "status": session.status,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "export_timestamp": datetime.now().isoformat(),
                    "export_version": "0.7.0"
                },
                "configuration": session.config,
                "performance_metrics": session.performance_metrics,
                "training_data": session.training_data
            }
            
            # Incluir embeddings si se solicita
            if include_embeddings and session.training_data:
                logger.info("Generando embeddings para exportación...")
                try:
                    X, _ = self.preprocess_training_data(session.training_data, use_cache=True)
                    export_data["embeddings"] = X.tolist()
                except Exception as e:
                    logger.warning(f"Error generando embeddings para exportación: {e}")
                    export_data["embeddings_error"] = str(e)
            
            # Exportar según la extensión del archivo
            if filepath.suffix.lower() == '.json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            elif filepath.suffix.lower() == '.pkl':
                with open(filepath, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                # Por defecto, usar JSON
                with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Datos de entrenamiento exportados a: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exportando datos de entrenamiento: {e}")
            raise RuntimeError(f"Fallo en exportación: {e}")
    
    def cleanup_resources(self) -> None:
        """
        Limpia recursos y guarda estado antes del cierre
        """
        try:
            logger.info("Iniciando limpieza de recursos...")
            
            # Guardar caché de embeddings
            self._save_embedding_cache()
            
            # Cerrar pool de threads
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Limpiar caché LRU
            if hasattr(self._get_cached_embedding, 'cache_clear'):
                self._get_cached_embedding.cache_clear()
            
            logger.info("Limpieza de recursos completada")
            
        except Exception as e:
            logger.error(f"Error durante limpieza de recursos: {e}")