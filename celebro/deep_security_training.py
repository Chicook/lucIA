#!/usr/bin/env python3
"""
Entrenamiento Profundo de Seguridad con Gemini + TensorFlow - LucIA
Versión: 0.6.0
Sistema de aprendizaje automático que se entrena con Gemini sobre temas de seguridad
"""

import os
import json
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pickle

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Gemini integration
from .red_neuronal.gemini_integration import GeminiIntegration
from .tensorflow_integration import TensorFlowCelebroIntegration, ModelType, TrainingStatus

logger = logging.getLogger('Deep_Security_Training')

@dataclass
class SecurityTrainingData:
    """Datos de entrenamiento de seguridad"""
    topic: str
    question: str
    gemini_response: str
    security_category: str
    complexity_level: int  # 1-5
    confidence_score: float  # 0-1
    timestamp: datetime
    training_quality: str  # high, medium, low

@dataclass
class TrainingSession:
    """Sesión de entrenamiento"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    topics_covered: List[str]
    questions_asked: int
    responses_generated: int
    models_trained: List[str]
    accuracy_improvement: float
    status: str  # active, completed, error

class DeepSecurityTrainer:
    """
    Entrenador profundo de seguridad que combina Gemini + TensorFlow
    """
    
    def __init__(self, models_dir: str = "celebro/security_models"):
        self.models_dir = models_dir
        self.gemini = GeminiIntegration()
        self.tensorflow = TensorFlowCelebroIntegration(models_dir)
        
        # Datos de entrenamiento
        self.training_data: List[SecurityTrainingData] = []
        self.training_sessions: List[TrainingSession] = []
        self.current_session: Optional[TrainingSession] = None
        
        # Configuración de entrenamiento
        self.security_topics = [
            "authentication", "encryption", "malware", "phishing", "firewall",
            "vulnerability_assessment", "secure_coding", "web_security", "gdpr",
            "incident_response", "network_security", "data_protection",
            "penetration_testing", "security_auditing", "compliance"
        ]
        
        self.complexity_questions = {
            1: "¿Qué es {topic}?",
            2: "¿Cómo funciona {topic}?",
            3: "¿Cuáles son las mejores prácticas de {topic}?",
            4: "¿Cómo implementar {topic} en un sistema empresarial?",
            5: "¿Cuáles son los desafíos avanzados y soluciones innovadoras de {topic}?"
        }
        
        # Crear directorio de modelos
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info("Entrenador profundo de seguridad inicializado")
    
    async def start_training_session(self, session_name: str = None) -> str:
        """
        Inicia una nueva sesión de entrenamiento
        
        Args:
            session_name: Nombre de la sesión (opcional)
            
        Returns:
            ID de la sesión
        """
        try:
            session_id = f"security_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = TrainingSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                topics_covered=[],
                questions_asked=0,
                responses_generated=0,
                models_trained=[],
                accuracy_improvement=0.0,
                status="active"
            )
            
            self.training_sessions.append(self.current_session)
            
            logger.info(f"Sesión de entrenamiento iniciada: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error iniciando sesión de entrenamiento: {e}")
            raise
    
    async def generate_security_questions(self, topic: str, complexity: int = 3, 
                                        num_questions: int = 5) -> List[str]:
        """
        Genera preguntas de seguridad usando Gemini
        
        Args:
            topic: Tema de seguridad
            complexity: Nivel de complejidad (1-5)
            num_questions: Número de preguntas a generar
            
        Returns:
            Lista de preguntas generadas
        """
        try:
            # Crear prompt para generar preguntas
            prompt = f"""
            Eres un experto en ciberseguridad. Genera {num_questions} preguntas específicas 
            sobre {topic} con nivel de complejidad {complexity}/5.
            
            Las preguntas deben ser:
            - Técnicamente precisas
            - Prácticas y aplicables
            - Progresivas en dificultad
            - Enfocadas en implementación real
            
            Formato: Lista numerada de preguntas.
            """
            
            # Obtener respuesta de Gemini
            response = self.gemini.generate_text(prompt)
            
            if response and 'text' in response:
                # Extraer preguntas del texto
                questions = self._extract_questions_from_text(response['text'])
                return questions[:num_questions]
            else:
                # Fallback: usar preguntas predefinidas
                base_question = self.complexity_questions.get(complexity, self.complexity_questions[3])
                return [base_question.format(topic=topic) for _ in range(num_questions)]
                
        except Exception as e:
            logger.error(f"Error generando preguntas de seguridad: {e}")
            # Fallback a preguntas predefinidas
            base_question = self.complexity_questions.get(complexity, self.complexity_questions[3])
            return [base_question.format(topic=topic) for _ in range(num_questions)]
    
    async def get_gemini_security_response(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Obtiene respuesta de seguridad de Gemini
        
        Args:
            question: Pregunta de seguridad
            context: Contexto adicional
            
        Returns:
            Respuesta estructurada de Gemini
        """
        try:
            # Crear prompt especializado en seguridad
            security_prompt = f"""
            Eres un experto en ciberseguridad con 20+ años de experiencia.
            
            Pregunta: {question}
            
            {f"Contexto: {context}" if context else ""}
            
            Proporciona una respuesta:
            1. Técnicamente precisa y actualizada
            2. Con ejemplos prácticos de código cuando sea relevante
            3. Mencionando las mejores prácticas de la industria
            4. Incluyendo consideraciones de seguridad específicas
            5. Con referencias a estándares (ISO 27001, NIST, OWASP, etc.)
            
            Formato de respuesta:
            - Resumen ejecutivo
            - Explicación técnica detallada
            - Ejemplos de implementación
            - Mejores prácticas
            - Consideraciones de seguridad
            """
            
            response = self.gemini.generate_text(security_prompt)
            
            if response and 'text' in response:
                # Analizar la respuesta para extraer información estructurada
                structured_response = self._analyze_security_response(response['text'], question)
                return structured_response
            else:
                return {
                    'text': 'Error obteniendo respuesta de Gemini',
                    'security_category': 'unknown',
                    'complexity_level': 1,
                    'confidence_score': 0.0,
                    'quality': 'low'
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo respuesta de seguridad: {e}")
            return {
                'text': f'Error: {str(e)}',
                'security_category': 'unknown',
                'complexity_level': 1,
                'confidence_score': 0.0,
                'quality': 'low'
            }
    
    def _extract_questions_from_text(self, text: str) -> List[str]:
        """Extrae preguntas de un texto"""
        questions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith(('¿', 'What', 'How', 'Why', 'When', 'Where'))):
                # Limpiar numeración y caracteres especiales
                question = line
                if line[0].isdigit() and '. ' in line:
                    question = line.split('. ', 1)[1]
                elif line.startswith('- '):
                    question = line[2:]
                elif line.startswith('* '):
                    question = line[2:]
                
                if question and len(question) > 10:  # Filtrar preguntas muy cortas
                    questions.append(question)
        
        return questions
    
    def _analyze_security_response(self, response_text: str, original_question: str) -> Dict[str, Any]:
        """Analiza una respuesta de seguridad para extraer información estructurada"""
        try:
            # Detectar categoría de seguridad
            security_category = self._detect_security_category(response_text, original_question)
            
            # Calcular nivel de complejidad
            complexity_level = self._calculate_complexity_level(response_text)
            
            # Calcular puntuación de confianza
            confidence_score = self._calculate_confidence_score(response_text)
            
            # Determinar calidad de la respuesta
            quality = self._determine_response_quality(response_text, complexity_level, confidence_score)
            
            return {
                'text': response_text,
                'security_category': security_category,
                'complexity_level': complexity_level,
                'confidence_score': confidence_score,
                'quality': quality,
                'original_question': original_question,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analizando respuesta de seguridad: {e}")
            return {
                'text': response_text,
                'security_category': 'unknown',
                'complexity_level': 1,
                'confidence_score': 0.5,
                'quality': 'medium',
                'original_question': original_question,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _detect_security_category(self, text: str, question: str) -> str:
        """Detecta la categoría de seguridad del texto"""
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Palabras clave por categoría
        category_keywords = {
            'authentication': ['autenticación', 'login', 'password', 'mfa', '2fa', 'biometric'],
            'encryption': ['encriptación', 'cifrado', 'aes', 'rsa', 'ssl', 'tls', 'crypto'],
            'malware': ['malware', 'virus', 'trojan', 'ransomware', 'antivirus', 'detection'],
            'phishing': ['phishing', 'social engineering', 'email', 'spoofing', 'fraud'],
            'firewall': ['firewall', 'network', 'packet', 'filtering', 'iptables'],
            'vulnerability': ['vulnerabilidad', 'exploit', 'cve', 'patch', 'update'],
            'web_security': ['web', 'http', 'https', 'xss', 'csrf', 'sql injection'],
            'compliance': ['compliance', 'gdpr', 'iso', 'nist', 'regulation', 'audit']
        }
        
        # Contar coincidencias por categoría
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower or keyword in question_lower)
            category_scores[category] = score
        
        # Retornar la categoría con mayor puntuación
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general_security'
    
    def _calculate_complexity_level(self, text: str) -> int:
        """Calcula el nivel de complejidad del texto (1-5)"""
        try:
            # Factores de complejidad
            word_count = len(text.split())
            technical_terms = len([word for word in text.split() if len(word) > 8])
            code_blocks = text.count('```') + text.count('`')
            bullet_points = text.count('•') + text.count('-') + text.count('*')
            
            # Calcular puntuación de complejidad
            complexity_score = 0
            
            if word_count > 500:
                complexity_score += 2
            elif word_count > 200:
                complexity_score += 1
            
            if technical_terms > 10:
                complexity_score += 2
            elif technical_terms > 5:
                complexity_score += 1
            
            if code_blocks > 3:
                complexity_score += 2
            elif code_blocks > 1:
                complexity_score += 1
            
            if bullet_points > 5:
                complexity_score += 1
            
            # Normalizar a escala 1-5
            complexity_level = min(5, max(1, complexity_score))
            return complexity_level
            
        except Exception as e:
            logger.error(f"Error calculando complejidad: {e}")
            return 3  # Nivel medio por defecto
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calcula la puntuación de confianza del texto (0-1)"""
        try:
            # Factores de confianza
            confidence_indicators = [
                'expert', 'professional', 'industry standard', 'best practice',
                'recommended', 'proven', 'tested', 'validated', 'secure',
                'robust', 'reliable', 'effective', 'efficient'
            ]
            
            uncertainty_indicators = [
                'maybe', 'perhaps', 'might', 'could', 'possibly', 'unclear',
                'unknown', 'uncertain', 'doubt', 'question', 'issue'
            ]
            
            text_lower = text.lower()
            
            confidence_count = sum(1 for indicator in confidence_indicators if indicator in text_lower)
            uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in text_lower)
            
            # Calcular puntuación base
            base_score = 0.5
            
            # Ajustar por indicadores
            confidence_boost = min(0.3, confidence_count * 0.05)
            uncertainty_penalty = min(0.2, uncertainty_count * 0.03)
            
            final_score = base_score + confidence_boost - uncertainty_penalty
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.5  # Puntuación media por defecto
    
    def _determine_response_quality(self, text: str, complexity: int, confidence: float) -> str:
        """Determina la calidad de la respuesta"""
        try:
            # Criterios de calidad
            has_code = '```' in text or '`' in text
            has_examples = 'example' in text.lower() or 'ejemplo' in text.lower()
            has_best_practices = any(term in text.lower() for term in 
                                   ['best practice', 'mejor práctica', 'recommended', 'recomendado'])
            is_detailed = len(text.split()) > 100
            
            quality_score = 0
            
            if has_code:
                quality_score += 2
            if has_examples:
                quality_score += 1
            if has_best_practices:
                quality_score += 1
            if is_detailed:
                quality_score += 1
            if complexity >= 4:
                quality_score += 1
            if confidence >= 0.7:
                quality_score += 1
            
            # Determinar calidad
            if quality_score >= 6:
                return 'high'
            elif quality_score >= 3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error determinando calidad: {e}")
            return 'medium'
    
    async def train_on_security_topic(self, topic: str, complexity_levels: List[int] = [1, 2, 3, 4, 5],
                                    questions_per_level: int = 3) -> Dict[str, Any]:
        """
        Entrena modelos en un tema específico de seguridad
        
        Args:
            topic: Tema de seguridad
            complexity_levels: Niveles de complejidad a entrenar
            questions_per_level: Preguntas por nivel de complejidad
            
        Returns:
            Resultados del entrenamiento
        """
        try:
            if not self.current_session:
                await self.start_training_session()
            
            logger.info(f"Iniciando entrenamiento en tema: {topic}")
            
            all_questions = []
            all_responses = []
            all_categories = []
            all_complexities = []
            
            # Generar preguntas para cada nivel de complejidad
            for complexity in complexity_levels:
                logger.info(f"Generando preguntas nivel {complexity} para {topic}")
                
                questions = await self.generate_security_questions(topic, complexity, questions_per_level)
                all_questions.extend(questions)
                
                # Obtener respuestas de Gemini para cada pregunta
                for question in questions:
                    logger.info(f"Obteniendo respuesta de Gemini para: {question[:50]}...")
                    
                    response = await self.get_gemini_security_response(question, f"Tema: {topic}")
                    
                    all_responses.append(response['text'])
                    all_categories.append(response['security_category'])
                    all_complexities.append(response['complexity_level'])
                    
                    # Guardar datos de entrenamiento
                    training_data = SecurityTrainingData(
                        topic=topic,
                        question=question,
                        gemini_response=response['text'],
                        security_category=response['security_category'],
                        complexity_level=response['complexity_level'],
                        confidence_score=response['confidence_score'],
                        timestamp=datetime.now(),
                        training_quality=response['quality']
                    )
                    
                    self.training_data.append(training_data)
                    
                    # Actualizar sesión actual
                    if self.current_session:
                        self.current_session.questions_asked += 1
                        self.current_session.responses_generated += 1
                        if topic not in self.current_session.topics_covered:
                            self.current_session.topics_covered.append(topic)
            
            # Crear y entrenar modelos
            logger.info(f"Creando modelos para {topic}")
            
            # Modelo de clasificación de categorías
            category_model_id = self.tensorflow.create_text_classification_model(
                f"Security_Category_{topic}",
                "text_classification",
                num_classes=len(set(all_categories))
            )
            
            # Modelo de análisis de sentimientos para respuestas de seguridad
            sentiment_model_id = self.tensorflow.create_sentiment_analysis_model(
                f"Security_Sentiment_{topic}"
            )
            
            # Modelo de análisis de complejidad
            complexity_model_id = self.tensorflow.create_text_classification_model(
                f"Security_Complexity_{topic}",
                "text_classification",
                num_classes=5  # 5 niveles de complejidad
            )
            
            # Entrenar modelos
            logger.info("Entrenando modelos...")
            
            # Entrenar modelo de categorías
            category_metrics = self.tensorflow.train_model(
                category_model_id,
                all_responses,
                all_categories
            )
            
            # Entrenar modelo de sentimientos
            sentiment_labels = ['positive' if 'good' in resp.lower() or 'excellent' in resp.lower() 
                              else 'negative' if 'bad' in resp.lower() or 'poor' in resp.lower() 
                              else 'neutral' for resp in all_responses]
            
            sentiment_metrics = self.tensorflow.train_model(
                sentiment_model_id,
                all_responses,
                sentiment_labels
            )
            
            # Entrenar modelo de complejidad
            complexity_metrics = self.tensorflow.train_model(
                complexity_model_id,
                all_responses,
                all_complexities
            )
            
            # Actualizar sesión
            if self.current_session:
                self.current_session.models_trained.extend([
                    category_model_id, sentiment_model_id, complexity_model_id
                ])
            
            # Calcular mejora promedio
            avg_accuracy = (
                category_metrics.accuracy + 
                sentiment_metrics.accuracy + 
                complexity_metrics.accuracy
            ) / 3
            
            if self.current_session:
                self.current_session.accuracy_improvement = avg_accuracy
            
            logger.info(f"Entrenamiento completado para {topic}. Precisión promedio: {avg_accuracy:.3f}")
            
            return {
                'topic': topic,
                'questions_generated': len(all_questions),
                'responses_obtained': len(all_responses),
                'models_created': 3,
                'category_model_id': category_model_id,
                'sentiment_model_id': sentiment_model_id,
                'complexity_model_id': complexity_model_id,
                'category_accuracy': category_metrics.accuracy,
                'sentiment_accuracy': sentiment_metrics.accuracy,
                'complexity_accuracy': complexity_metrics.accuracy,
                'average_accuracy': avg_accuracy,
                'training_quality': 'high' if avg_accuracy > 0.8 else 'medium' if avg_accuracy > 0.6 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error entrenando en tema {topic}: {e}")
            return {'error': str(e), 'topic': topic}
    
    async def comprehensive_security_training(self, topics: List[str] = None) -> Dict[str, Any]:
        """
        Entrenamiento comprensivo en todos los temas de seguridad
        
        Args:
            topics: Lista de temas (usa todos si no se especifica)
            
        Returns:
            Resultados del entrenamiento comprensivo
        """
        try:
            if not topics:
                topics = self.security_topics
            
            logger.info(f"Iniciando entrenamiento comprensivo en {len(topics)} temas")
            
            # Iniciar sesión de entrenamiento
            session_id = await self.start_training_session("Comprehensive Security Training")
            
            results = []
            total_accuracy = 0
            successful_topics = 0
            
            for i, topic in enumerate(topics, 1):
                logger.info(f"Procesando tema {i}/{len(topics)}: {topic}")
                
                try:
                    result = await self.train_on_security_topic(topic)
                    
                    if 'error' not in result:
                        results.append(result)
                        total_accuracy += result['average_accuracy']
                        successful_topics += 1
                        
                        logger.info(f"✅ {topic}: Precisión {result['average_accuracy']:.3f}")
                    else:
                        logger.error(f"❌ {topic}: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error en {topic}: {e}")
            
            # Finalizar sesión
            if self.current_session:
                self.current_session.end_time = datetime.now()
                self.current_session.status = "completed"
                self.current_session.accuracy_improvement = total_accuracy / max(successful_topics, 1)
            
            # Calcular estadísticas finales
            avg_accuracy = total_accuracy / max(successful_topics, 1)
            
            logger.info(f"Entrenamiento comprensivo completado. Precisión promedio: {avg_accuracy:.3f}")
            
            return {
                'session_id': session_id,
                'topics_processed': len(topics),
                'successful_topics': successful_topics,
                'failed_topics': len(topics) - successful_topics,
                'total_questions': sum(r.get('questions_generated', 0) for r in results),
                'total_responses': sum(r.get('responses_obtained', 0) for r in results),
                'total_models': sum(r.get('models_created', 0) for r in results),
                'average_accuracy': avg_accuracy,
                'training_quality': 'excellent' if avg_accuracy > 0.9 else 'good' if avg_accuracy > 0.7 else 'fair',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento comprensivo: {e}")
            return {'error': str(e)}
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del entrenamiento"""
        try:
            return {
                'current_session': {
                    'id': self.current_session.session_id if self.current_session else None,
                    'status': self.current_session.status if self.current_session else 'none',
                    'topics_covered': self.current_session.topics_covered if self.current_session else [],
                    'questions_asked': self.current_session.questions_asked if self.current_session else 0,
                    'models_trained': self.current_session.models_trained if self.current_session else [],
                    'accuracy_improvement': self.current_session.accuracy_improvement if self.current_session else 0.0
                },
                'total_training_data': len(self.training_data),
                'total_sessions': len(self.training_sessions),
                'tensorflow_models': len(self.tensorflow.models),
                'gemini_connected': self.gemini.test_connection(),
                'available_topics': self.security_topics
            }
        except Exception as e:
            logger.error(f"Error obteniendo estado de entrenamiento: {e}")
            return {'error': str(e)}
    
    def save_training_data(self, filepath: str = None) -> str:
        """Guarda los datos de entrenamiento"""
        try:
            if not filepath:
                filepath = os.path.join(self.models_dir, f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            
            data_to_save = {
                'training_data': self.training_data,
                'training_sessions': self.training_sessions,
                'security_topics': self.security_topics,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            logger.info(f"Datos de entrenamiento guardados en: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error guardando datos de entrenamiento: {e}")
            raise
    
    def load_training_data(self, filepath: str):
        """Carga datos de entrenamiento guardados"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.training_data = data.get('training_data', [])
            self.training_sessions = data.get('training_sessions', [])
            self.security_topics = data.get('security_topics', self.security_topics)
            
            logger.info(f"Datos de entrenamiento cargados desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando datos de entrenamiento: {e}")
            raise

# Instancia global del entrenador
deep_security_trainer = DeepSecurityTrainer()
