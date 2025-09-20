"""
Entrenador de Aprendizaje Profundo para @red_neuronal
Versión: 0.6.0
Sistema de entrenamiento de IA usando prompts educativos de ciberseguridad
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .security_topics import SecurityTopics, SecurityTopic, SecurityLevel
from .prompt_generator import PromptGenerator, LearningPrompt, PromptType, DifficultyLevel
from .knowledge_base import KnowledgeBase, LearningSession
from .learning_curriculum import LearningCurriculum, LearningPath, LearningPhase
from ..neural_core import NeuralCore, NetworkConfig, TrainingConfig

logger = logging.getLogger('Deep_Learning_Trainer')

@dataclass
class TrainingSession:
    """Sesión de entrenamiento de IA"""
    id: str
    model_id: str
    curriculum_path: str
    prompts_used: List[str]
    training_data: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # "active", "completed", "failed"

@dataclass
class LearningProgress:
    """Progreso de aprendizaje de la IA"""
    topic_id: str
    mastery_level: float  # 0.0 - 1.0
    confidence_score: float  # 0.0 - 1.0
    prompts_completed: int
    correct_responses: int
    last_updated: datetime

class DeepLearningTrainer:
    """Entrenador de aprendizaje profundo para IA en ciberseguridad"""
    
    def __init__(self):
        self.security_topics = SecurityTopics()
        self.prompt_generator = PromptGenerator()
        self.knowledge_base = KnowledgeBase()
        self.curriculum = LearningCurriculum()
        self.neural_core = NeuralCore()
        self.training_sessions = []
        self.learning_progress = {}
        
        logger.info("Entrenador de Aprendizaje Profundo inicializado")
    
    def create_training_session(self, curriculum_path: str, model_config: Dict[str, Any] = None) -> TrainingSession:
        """Crea una nueva sesión de entrenamiento"""
        try:
            session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Configuración por defecto del modelo
            if model_config is None:
                model_config = {
                    "input_size": 512,  # Tamaño de entrada para embeddings de texto
                    "hidden_layers": [256, 128, 64],
                    "output_size": 10,  # Número de categorías de respuesta
                    "activation": "relu",
                    "output_activation": "softmax",
                    "learning_rate": 0.001,
                    "dropout_rate": 0.3
                }
            
            # Crear modelo de red neuronal
            network_config = NetworkConfig(
                input_size=model_config["input_size"],
                hidden_layers=model_config["hidden_layers"],
                output_size=model_config["output_size"],
                activation=model_config["activation"],
                output_activation=model_config["output_activation"],
                learning_rate=model_config["learning_rate"],
                dropout_rate=model_config["dropout_rate"]
            )
            
            network = self.neural_core.create_network(network_config)
            
            # Crear sesión de entrenamiento
            session = TrainingSession(
                id=session_id,
                model_id=f"model_{session_id}",
                curriculum_path=curriculum_path,
                prompts_used=[],
                training_data=[],
                performance_metrics={},
                start_time=datetime.now(),
                end_time=None,
                status="active"
            )
            
            self.training_sessions.append(session)
            
            logger.info(f"Sesión de entrenamiento creada: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creando sesión de entrenamiento: {e}")
            raise
    
    def generate_training_data(self, topic_ids: List[str], num_prompts_per_topic: int = 5) -> List[Dict[str, Any]]:
        """Genera datos de entrenamiento a partir de prompts educativos"""
        try:
            training_data = []
            
            for topic_id in topic_ids:
                topic = self.security_topics.get_topic(topic_id)
                if not topic:
                    continue
                
                # Generar diferentes tipos de prompts
                prompt_types = [PromptType.CONCEPTUAL, PromptType.PRACTICO, PromptType.CODIGO]
                
                for i in range(num_prompts_per_topic):
                    for prompt_type in prompt_types:
                        try:
                            # Generar prompt
                            prompt = self.prompt_generator.generate_prompt(
                                topic_id, prompt_type, DifficultyLevel.MEDIO
                            )
                            
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
                                "created_at": prompt.created_at.isoformat()
                            }
                            
                            training_data.append(training_sample)
                            
                        except Exception as e:
                            logger.warning(f"Error generando prompt para {topic_id}: {e}")
                            continue
            
            logger.info(f"Generados {len(training_data)} ejemplos de entrenamiento")
            return training_data
            
        except Exception as e:
            logger.error(f"Error generando datos de entrenamiento: {e}")
            raise
    
    def preprocess_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesa los datos de entrenamiento para la red neuronal"""
        try:
            # Convertir texto a embeddings (simplificado)
            X = []
            y = []
            
            # Mapeo de tipos de prompt a categorías numéricas
            prompt_type_mapping = {
                "conceptual": 0,
                "practico": 1,
                "codigo": 2,
                "caso_estudio": 3,
                "evaluacion": 4,
                "simulacion": 5
            }
            
            # Mapeo de temas a categorías numéricas
            topic_mapping = {}
            all_topics = self.security_topics.get_all_topics()
            for i, topic in enumerate(all_topics):
                topic_mapping[topic.id] = i
            
            for sample in training_data:
                # Crear embedding simple basado en características del texto
                input_text = sample["input_text"]
                prompt_type = sample["prompt_type"]
                topic_id = sample["topic_id"]
                keywords = sample["keywords"]
                
                # Embedding básico (en una implementación real usaríamos un modelo de embeddings)
                embedding = self._create_simple_embedding(input_text, prompt_type, topic_id, keywords)
                X.append(embedding)
                
                # Categoría de salida basada en el tipo de prompt
                output_category = prompt_type_mapping.get(prompt_type, 0)
                y.append(output_category)
            
            X = np.array(X)
            y = np.array(y)
            
            # Convertir a one-hot encoding
            num_classes = len(prompt_type_mapping)
            y_onehot = np.eye(num_classes)[y]
            
            logger.info(f"Datos preprocesados: {X.shape[0]} muestras, {X.shape[1]} características")
            return X, y_onehot
            
        except Exception as e:
            logger.error(f"Error preprocesando datos de entrenamiento: {e}")
            raise
    
    def _create_simple_embedding(self, text: str, prompt_type: str, topic_id: str, keywords: List[str]) -> np.ndarray:
        """Crea un embedding simple para el texto (placeholder para implementación real)"""
        # En una implementación real, usaríamos un modelo de embeddings como Word2Vec, FastText, o BERT
        
        # Embedding básico basado en características del texto
        embedding_size = 512
        
        # Características básicas del texto
        text_length = len(text)
        word_count = len(text.split())
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        
        # Características del tipo de prompt
        prompt_type_features = {
            "conceptual": [1, 0, 0, 0, 0, 0],
            "practico": [0, 1, 0, 0, 0, 0],
            "codigo": [0, 0, 1, 0, 0, 0],
            "caso_estudio": [0, 0, 0, 1, 0, 0],
            "evaluacion": [0, 0, 0, 0, 1, 0],
            "simulacion": [0, 0, 0, 0, 0, 1]
        }
        
        # Características del tema
        topic_features = [0] * len(self.security_topics.get_all_topics())
        all_topics = self.security_topics.get_all_topics()
        for i, topic in enumerate(all_topics):
            if topic.id == topic_id:
                topic_features[i] = 1
                break
        
        # Combinar características
        features = [
            text_length / 1000.0,  # Normalizar longitud
            word_count / 100.0,    # Normalizar conteo de palabras
            keyword_count / 10.0,  # Normalizar conteo de keywords
        ]
        
        # Agregar características del tipo de prompt
        features.extend(prompt_type_features.get(prompt_type, [0] * 6))
        
        # Agregar características del tema
        features.extend(topic_features)
        
        # Rellenar hasta el tamaño deseado
        while len(features) < embedding_size:
            features.append(0.0)
        
        # Truncar si es muy largo
        features = features[:embedding_size]
        
        return np.array(features, dtype=np.float32)
    
    def train_model(self, session_id: str, training_data: List[Dict[str, Any]], 
                   epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Entrena el modelo con los datos proporcionados"""
        try:
            # Encontrar sesión
            session = next((s for s in self.training_sessions if s.id == session_id), None)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            # Preprocesar datos
            X, y = self.preprocess_training_data(training_data)
            
            # Configurar entrenamiento
            training_config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.001,
                validation_split=0.2,
                early_stopping=True,
                patience=10,
                verbose=1
            )
            
            # Entrenar modelo
            logger.info(f"Iniciando entrenamiento para sesión {session_id}")
            history = self.neural_core.train(X, y, config=training_config)
            
            # Actualizar sesión
            session.training_data = training_data
            session.performance_metrics = {
                "final_loss": history[-1].train_loss if history else 0.0,
                "final_accuracy": history[-1].train_accuracy if history else 0.0,
                "epochs_completed": len(history),
                "total_samples": len(training_data)
            }
            session.end_time = datetime.now()
            session.status = "completed"
            
            logger.info(f"Entrenamiento completado para sesión {session_id}")
            
            return {
                "session_id": session_id,
                "status": "completed",
                "performance_metrics": session.performance_metrics,
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
            logger.error(f"Error entrenando modelo: {e}")
            # Marcar sesión como fallida
            session = next((s for s in self.training_sessions if s.id == session_id), None)
            if session:
                session.status = "failed"
                session.end_time = datetime.now()
            raise
    
    def evaluate_model(self, session_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evalúa el modelo entrenado"""
        try:
            # Encontrar sesión
            session = next((s for s in self.training_sessions if s.id == session_id), None)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            if session.status != "completed":
                raise ValueError("El modelo debe estar entrenado antes de la evaluación")
            
            # Preprocesar datos de prueba
            X_test, y_test = self.preprocess_training_data(test_data)
            
            # Evaluar modelo
            metrics = self.neural_core.evaluate(X_test, y_test)
            
            # Realizar predicciones
            predictions = self.neural_core.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # Calcular métricas adicionales
            accuracy = np.mean(predicted_classes == true_classes)
            precision = self._calculate_precision(true_classes, predicted_classes)
            recall = self._calculate_recall(true_classes, predicted_classes)
            f1_score = self._calculate_f1_score(precision, recall)
            
            evaluation_results = {
                "session_id": session_id,
                "test_samples": len(test_data),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "loss": metrics["loss"],
                "detailed_metrics": metrics
            }
            
            logger.info(f"Evaluación completada para sesión {session_id}: Accuracy={accuracy:.4f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            raise
    
    def _calculate_precision(self, true_classes: np.ndarray, predicted_classes: np.ndarray) -> float:
        """Calcula la precisión promedio"""
        try:
            unique_classes = np.unique(true_classes)
            precisions = []
            
            for cls in unique_classes:
                true_positives = np.sum((predicted_classes == cls) & (true_classes == cls))
                false_positives = np.sum((predicted_classes == cls) & (true_classes != cls))
                
                if true_positives + false_positives > 0:
                    precision = true_positives / (true_positives + false_positives)
                    precisions.append(precision)
            
            return np.mean(precisions) if precisions else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando precisión: {e}")
            return 0.0
    
    def _calculate_recall(self, true_classes: np.ndarray, predicted_classes: np.ndarray) -> float:
        """Calcula el recall promedio"""
        try:
            unique_classes = np.unique(true_classes)
            recalls = []
            
            for cls in unique_classes:
                true_positives = np.sum((predicted_classes == cls) & (true_classes == cls))
                false_negatives = np.sum((predicted_classes != cls) & (true_classes == cls))
                
                if true_positives + false_negatives > 0:
                    recall = true_positives / (true_positives + false_negatives)
                    recalls.append(recall)
            
            return np.mean(recalls) if recalls else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando recall: {e}")
            return 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calcula el F1 score"""
        try:
            if precision + recall > 0:
                return 2 * (precision * recall) / (precision + recall)
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculando F1 score: {e}")
            return 0.0
    
    def generate_learning_curriculum(self, path: LearningPath, 
                                   num_prompts_per_topic: int = 10) -> Dict[str, Any]:
        """Genera un currículum completo de aprendizaje"""
        try:
            # Obtener plan de aprendizaje
            learning_plan = self.curriculum.generate_learning_plan(path)
            
            # Generar datos de entrenamiento para todos los temas
            all_topics = []
            for module in self.curriculum.get_learning_path(path):
                all_topics.extend(module.topics)
            
            # Eliminar duplicados
            unique_topics = list(set(all_topics))
            
            # Generar datos de entrenamiento
            training_data = self.generate_training_data(unique_topics, num_prompts_per_topic)
            
            # Crear sesión de entrenamiento
            session = self.create_training_session(path.value)
            
            return {
                "learning_plan": learning_plan,
                "training_data": training_data,
                "session_id": session.id,
                "total_prompts": len(training_data),
                "topics_covered": unique_topics
            }
            
        except Exception as e:
            logger.error(f"Error generando currículum de aprendizaje: {e}")
            raise
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de entrenamiento"""
        try:
            total_sessions = len(self.training_sessions)
            completed_sessions = sum(1 for s in self.training_sessions if s.status == "completed")
            failed_sessions = sum(1 for s in self.training_sessions if s.status == "failed")
            active_sessions = sum(1 for s in self.training_sessions if s.status == "active")
            
            # Estadísticas de rendimiento
            performance_stats = []
            for session in self.training_sessions:
                if session.status == "completed" and session.performance_metrics:
                    performance_stats.append(session.performance_metrics)
            
            avg_accuracy = 0.0
            avg_loss = 0.0
            if performance_stats:
                avg_accuracy = np.mean([p.get("final_accuracy", 0) for p in performance_stats])
                avg_loss = np.mean([p.get("final_loss", 0) for p in performance_stats])
            
            return {
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "failed_sessions": failed_sessions,
                "active_sessions": active_sessions,
                "completion_rate": completed_sessions / total_sessions if total_sessions > 0 else 0,
                "average_accuracy": avg_accuracy,
                "average_loss": avg_loss,
                "total_prompts_generated": len(self.prompt_generator.generated_prompts)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas de entrenamiento: {e}")
            return {}
    
    def export_training_data(self, session_id: str, filepath: str) -> None:
        """Exporta los datos de entrenamiento de una sesión"""
        try:
            session = next((s for s in self.training_sessions if s.id == session_id), None)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            
            export_data = {
                "session_id": session.id,
                "model_id": session.model_id,
                "curriculum_path": session.curriculum_path,
                "status": session.status,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "performance_metrics": session.performance_metrics,
                "training_data": session.training_data,
                "export_date": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Datos de entrenamiento exportados a: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exportando datos de entrenamiento: {e}")
            raise
