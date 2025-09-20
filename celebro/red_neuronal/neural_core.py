"""
Núcleo de Red Neuronal
Versión: 0.6.0
Sistema central que coordina todos los componentes de la red neuronal
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import pickle
import os

from .neural_network import NeuralNetwork, NetworkConfig
from .layers import DenseLayer, DropoutLayer, BatchNormLayer, ConvLayer, PoolingLayer, FlattenLayer
from .optimizers import Adam, SGD, RMSprop, LearningRateScheduler, GradientClipping
from .loss_functions import CrossEntropy, MSE, BinaryCrossEntropy, Huber
from .gemini_integration import GeminiIntegration
from .query_analyzer import DeepQueryAnalyzer
from .deep_learning_engine import DeepLearningEngine

logger = logging.getLogger('Neural_Core')

@dataclass
class TrainingConfig:
    """Configuración de entrenamiento"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    verbose: int = 1
    shuffle: bool = True
    callbacks: List[Any] = None

@dataclass
class ModelMetrics:
    """Métricas del modelo"""
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch: int
    timestamp: datetime

class NeuralCore:
    """
    Núcleo central del sistema de redes neuronales.
    Coordina todos los componentes y proporciona una interfaz unificada.
    """
    
    def __init__(self):
        self.network = None
        self.training_config = None
        self.training_history = []
        self.best_weights = None
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.is_training = False
        
        # Directorios
        self.models_dir = "celebro/red_neuronal/models"
        self.logs_dir = "celebro/red_neuronal/logs"
        self.data_dir = "celebro/red_neuronal/data"
        
        # Crear directorios si no existen
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Nuevos componentes de aprendizaje profundo
        self.gemini_integration = GeminiIntegration()
        self.query_analyzer = DeepQueryAnalyzer()
        self.deep_learning_engine = DeepLearningEngine()
        
        logger.info("Núcleo de red neuronal inicializado")
    
    def create_network(self, config: NetworkConfig) -> NeuralNetwork:
        """Crea una nueva red neuronal"""
        try:
            self.network = NeuralNetwork(config)
            
            # Construir arquitectura básica
            self._build_architecture(config)
            
            logger.info(f"Red neuronal creada: {config}")
            return self.network
            
        except Exception as e:
            logger.error(f"Error creando red neuronal: {e}")
            raise
    
    def _build_architecture(self, config: NetworkConfig):
        """Construye la arquitectura de la red"""
        try:
            # Capa de entrada (implícita)
            current_units = config.input_size
            
            # Capas ocultas
            for i, hidden_units in enumerate(config.hidden_layers):
                # Capa densa
                dense_layer = DenseLayer(
                    units=hidden_units,
                    activation=config.activation,
                    kernel_initializer='he_normal',
                    name=f'dense_{i+1}'
                )
                self.network.add_layer(dense_layer)
                
                # Batch normalization si está habilitado
                if config.batch_normalization:
                    bn_layer = BatchNormLayer(name=f'batch_norm_{i+1}')
                    self.network.add_layer(bn_layer)
                
                # Dropout si está habilitado
                if config.dropout_rate > 0:
                    dropout_layer = DropoutLayer(rate=config.dropout_rate, name=f'dropout_{i+1}')
                    self.network.add_layer(dropout_layer)
            
            # Capa de salida
            output_layer = DenseLayer(
                units=config.output_size,
                activation=config.output_activation,
                kernel_initializer='glorot_uniform',
                name='output'
            )
            self.network.add_layer(output_layer)
            
            # Construir la red
            self.network.build()
            
            # Compilar con optimizador y función de pérdida
            self.network.compile(
                optimizer=config.optimizer,
                loss=config.loss_function
            )
            
            logger.info("Arquitectura construida exitosamente")
            
        except Exception as e:
            logger.error(f"Error construyendo arquitectura: {e}")
            raise
    
    def create_convolutional_network(self, input_shape: Tuple[int, int, int], 
                                   num_classes: int, config: Dict[str, Any] = None) -> NeuralNetwork:
        """Crea una red neuronal convolucional"""
        try:
            # Configuración por defecto
            default_config = {
                'filters': [32, 64, 128],
                'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
                'pool_sizes': [(2, 2), (2, 2), (2, 2)],
                'dense_units': [512, 256],
                'dropout_rate': 0.5,
                'activation': 'relu',
                'output_activation': 'softmax'
            }
            
            if config:
                default_config.update(config)
            
            # Crear red
            network_config = NetworkConfig(
                input_size=input_shape[0] * input_shape[1] * input_shape[2],
                hidden_layers=default_config['dense_units'],
                output_size=num_classes,
                activation=default_config['activation'],
                output_activation=default_config['output_activation'],
                dropout_rate=default_config['dropout_rate']
            )
            
            self.network = NeuralNetwork(network_config)
            
            # Construir arquitectura convolucional
            self._build_convolutional_architecture(input_shape, default_config)
            
            logger.info("Red convolucional creada exitosamente")
            return self.network
            
        except Exception as e:
            logger.error(f"Error creando red convolucional: {e}")
            raise
    
    def _build_convolutional_architecture(self, input_shape: Tuple[int, int, int], config: Dict[str, Any]):
        """Construye la arquitectura convolucional"""
        try:
            # Capas convolucionales
            for i, (filters, kernel_size, pool_size) in enumerate(zip(
                config['filters'], config['kernel_sizes'], config['pool_sizes']
            )):
                # Capa convolucional
                conv_layer = ConvLayer(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation=config['activation'],
                    name=f'conv_{i+1}'
                )
                self.network.add_layer(conv_layer)
                
                # Capa de pooling
                pool_layer = PoolingLayer(
                    pool_size=pool_size,
                    pool_type='max',
                    name=f'pool_{i+1}'
                )
                self.network.add_layer(pool_layer)
            
            # Aplanar para capas densas
            flatten_layer = FlattenLayer(name='flatten')
            self.network.add_layer(flatten_layer)
            
            # Capas densas
            for i, units in enumerate(config['dense_units']):
                dense_layer = DenseLayer(
                    units=units,
                    activation=config['activation'],
                    name=f'dense_{i+1}'
                )
                self.network.add_layer(dense_layer)
                
                # Dropout
                if config['dropout_rate'] > 0:
                    dropout_layer = DropoutLayer(
                        rate=config['dropout_rate'],
                        name=f'dropout_{i+1}'
                    )
                    self.network.add_layer(dropout_layer)
            
            # Capa de salida
            output_layer = DenseLayer(
                units=1 if config['output_activation'] == 'sigmoid' else len(config['dense_units']),
                activation=config['output_activation'],
                name='output'
            )
            self.network.add_layer(output_layer)
            
            # Construir la red
            self.network.build()
            
            # Compilar
            self.network.compile(
                optimizer='adam',
                loss='categorical_crossentropy' if config['output_activation'] == 'softmax' else 'binary_crossentropy'
            )
            
        except Exception as e:
            logger.error(f"Error construyendo arquitectura convolucional: {e}")
            raise
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray = None, y_val: np.ndarray = None,
              config: TrainingConfig = None) -> List[ModelMetrics]:
        """Entrena la red neuronal"""
        try:
            if self.network is None:
                raise ValueError("No hay red neuronal creada")
            
            self.training_config = config or TrainingConfig()
            self.is_training = True
            self.training_history = []
            self.early_stopping_counter = 0
            
            # Preparar datos de validación
            if x_val is None or y_val is None:
                x_val, y_val = self._split_validation_data(x_train, y_train)
            
            # Entrenar por épocas
            for epoch in range(self.training_config.epochs):
                # Entrenar una época
                train_metrics = self._train_epoch(x_train, y_train)
                
                # Evaluar en validación
                val_metrics = self._evaluate_epoch(x_val, y_val)
                
                # Crear métricas combinadas
                metrics = ModelMetrics(
                    train_loss=train_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    val_loss=val_metrics['loss'],
                    val_accuracy=val_metrics['accuracy'],
                    epoch=epoch + 1,
                    timestamp=datetime.now()
                )
                
                self.training_history.append(metrics)
                
                # Log de progreso
                if self.training_config.verbose > 0:
                    self._log_epoch(metrics)
                
                # Early stopping
                if self.training_config.early_stopping:
                    if self._check_early_stopping(metrics):
                        logger.info(f"Early stopping en época {epoch + 1}")
                        break
            
            self.is_training = False
            logger.info("Entrenamiento completado")
            return self.training_history
            
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            self.is_training = False
            raise
    
    def _split_validation_data(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Divide los datos de entrenamiento en entrenamiento y validación"""
        try:
            n_samples = x_train.shape[0]
            n_val = int(n_samples * self.training_config.validation_split)
            
            # Mezclar datos
            if self.training_config.shuffle:
                indices = np.random.permutation(n_samples)
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            # Dividir
            x_val = x_train[:n_val]
            y_val = y_train[:n_val]
            x_train = x_train[n_val:]
            y_train = y_train[n_val:]
            
            return x_val, y_val
            
        except Exception as e:
            logger.error(f"Error dividiendo datos de validación: {e}")
            raise
    
    def _train_epoch(self, x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Entrena una época"""
        try:
            n_samples = x_train.shape[0]
            batch_size = self.training_config.batch_size
            n_batches = n_samples // batch_size
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for batch in range(n_batches):
                # Obtener lote
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Forward pass
                predictions = self.network.forward(x_batch)
                
                # Calcular pérdida
                loss = self.network.loss_function.compute(y_batch, predictions)
                
                # Calcular gradientes
                loss_gradients = self.network.loss_function.gradient(y_batch, predictions)
                
                # Backward pass
                self.network.backward(loss_gradients)
                
                # Actualizar parámetros
                self._update_parameters()
                
                # Acumular métricas
                epoch_loss += loss
                
                # Calcular precisión
                if self.network.config.output_activation == 'softmax':
                    predicted_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predicted_classes == true_classes)
                else:
                    accuracy = 1.0 - (loss / np.var(y_batch))
                
                epoch_accuracy += accuracy
            
            return {
                'loss': epoch_loss / n_batches,
                'accuracy': epoch_accuracy / n_batches
            }
            
        except Exception as e:
            logger.error(f"Error entrenando época: {e}")
            raise
    
    def _evaluate_epoch(self, x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evalúa una época en datos de validación"""
        try:
            # Establecer modo de inferencia
            for layer in self.network.layers:
                layer.set_training(False)
            
            # Evaluar
            metrics = self.network.evaluate(x_val, y_val)
            
            # Restaurar modo de entrenamiento
            for layer in self.network.layers:
                layer.set_training(True)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluando época: {e}")
            raise
    
    def _update_parameters(self):
        """Actualiza los parámetros de la red"""
        try:
            # Recopilar gradientes de todas las capas
            all_gradients = {}
            all_parameters = {}
            
            for layer in self.network.layers:
                layer_gradients = layer.get_gradients()
                layer_parameters = layer.get_parameters()
                
                for key, grad in layer_gradients.items():
                    full_key = f"{layer.name}_{key}"
                    all_gradients[full_key] = grad
                    all_parameters[full_key] = layer_parameters[key]
            
            # Aplicar recorte de gradientes si está configurado
            if hasattr(self.network.optimizer, 'gradient_clipping'):
                all_gradients = self.network.optimizer.gradient_clipping.clip_gradients(all_gradients)
            
            # Actualizar parámetros
            updated_parameters = self.network.optimizer.update(all_parameters, all_gradients)
            
            # Distribuir parámetros actualizados a las capas
            for layer in self.network.layers:
                layer_parameters = {}
                for key, param in layer.get_parameters().items():
                    full_key = f"{layer.name}_{key}"
                    if full_key in updated_parameters:
                        layer_parameters[key] = updated_parameters[full_key]
                    else:
                        layer_parameters[key] = param
                
                layer.set_parameters(layer_parameters)
            
        except Exception as e:
            logger.error(f"Error actualizando parámetros: {e}")
            raise
    
    def _check_early_stopping(self, metrics: ModelMetrics) -> bool:
        """Verifica si se debe aplicar early stopping"""
        try:
            if len(self.training_history) < 2:
                return False
            
            # Verificar si la pérdida de validación mejoró
            if metrics.val_loss < self.training_history[-2].val_loss - self.training_config.min_delta:
                self.early_stopping_counter = 0
                # Guardar mejores pesos
                self.best_weights = self._get_current_weights()
                self.best_epoch = metrics.epoch
            else:
                self.early_stopping_counter += 1
            
            # Aplicar early stopping si se alcanzó la paciencia
            if self.early_stopping_counter >= self.training_config.patience:
                if self.training_config.restore_best_weights and self.best_weights:
                    self._set_weights(self.best_weights)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verificando early stopping: {e}")
            return False
    
    def _get_current_weights(self) -> Dict[str, Any]:
        """Obtiene los pesos actuales de la red"""
        weights = {}
        for layer in self.network.layers:
            weights[layer.name] = layer.get_parameters()
        return weights
    
    def _set_weights(self, weights: Dict[str, Any]):
        """Establece los pesos de la red"""
        for layer in self.network.layers:
            if layer.name in weights:
                layer.set_parameters(weights[layer.name])
    
    def _log_epoch(self, metrics: ModelMetrics):
        """Registra el progreso de una época"""
        print(f"Época {metrics.epoch:3d} - "
              f"Pérdida: {metrics.train_loss:.4f} - "
              f"Precisión: {metrics.train_accuracy:.4f} - "
              f"Val. Pérdida: {metrics.val_loss:.4f} - "
              f"Val. Precisión: {metrics.val_accuracy:.4f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Realiza predicciones"""
        try:
            if self.network is None:
                raise ValueError("No hay red neuronal creada")
            
            return self.network.predict(x)
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evalúa la red en datos de prueba"""
        try:
            if self.network is None:
                raise ValueError("No hay red neuronal creada")
            
            return self.network.evaluate(x_test, y_test)
            
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Guarda el modelo completo"""
        try:
            if self.network is None:
                raise ValueError("No hay red neuronal para guardar")
            
            model_data = {
                'network': self.network,
                'training_history': self.training_history,
                'best_weights': self.best_weights,
                'best_epoch': self.best_epoch,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Carga un modelo guardado"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.network = model_data['network']
            self.training_history = model_data.get('training_history', [])
            self.best_weights = model_data.get('best_weights', None)
            self.best_epoch = model_data.get('best_epoch', 0)
            
            logger.info(f"Modelo cargado desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def get_training_summary(self) -> str:
        """Obtiene un resumen del entrenamiento"""
        try:
            if not self.training_history:
                return "No hay historial de entrenamiento"
            
            summary = "=" * 60 + "\n"
            summary += "RESUMEN DE ENTRENAMIENTO\n"
            summary += "=" * 60 + "\n"
            summary += f"Épocas entrenadas: {len(self.training_history)}\n"
            summary += f"Mejor época: {self.best_epoch}\n"
            summary += f"Última pérdida de entrenamiento: {self.training_history[-1].train_loss:.4f}\n"
            summary += f"Última precisión de entrenamiento: {self.training_history[-1].train_accuracy:.4f}\n"
            summary += f"Última pérdida de validación: {self.training_history[-1].val_loss:.4f}\n"
            summary += f"Última precisión de validación: {self.training_history[-1].val_accuracy:.4f}\n"
            summary += "=" * 60
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generando resumen: {e}")
            return "Error generando resumen"
    
    def get_network_summary(self) -> str:
        """Obtiene un resumen de la red neuronal"""
        try:
            if self.network is None:
                return "No hay red neuronal creada"
            
            return self.network.get_summary()
            
        except Exception as e:
            logger.error(f"Error generando resumen de red: {e}")
            return "Error generando resumen de red"
    
    # ===== MÉTODOS DE APRENDIZAJE PROFUNDO =====
    
    async def analyze_query_deep(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analiza una consulta de forma profunda usando el analizador avanzado
        
        Args:
            query: Consulta a analizar
            context: Contexto adicional
            
        Returns:
            Análisis detallado de la consulta
        """
        try:
            analysis = self.query_analyzer.analyze_query(query, context)
            
            return {
                'query_id': analysis.query_id,
                'complexity': analysis.complexity.value,
                'category': analysis.category.value,
                'keywords': analysis.keywords,
                'entities': analysis.entities,
                'intent': analysis.intent,
                'learning_potential': analysis.learning_potential,
                'suggested_prompts': analysis.suggested_prompts,
                'metadata': analysis.analysis_metadata
            }
            
        except Exception as e:
            logger.error(f"Error en análisis profundo de consulta: {e}")
            return {'error': str(e)}
    
    async def start_deep_learning_session(self, initial_query: str, 
                                        learning_objectives: List[str] = None) -> str:
        """
        Inicia una sesión de aprendizaje profundo
        
        Args:
            initial_query: Consulta inicial
            learning_objectives: Objetivos de aprendizaje
            
        Returns:
            ID de la sesión
        """
        try:
            session_id = await self.deep_learning_engine.start_learning_session(
                initial_query, learning_objectives
            )
            
            logger.info(f"Sesión de aprendizaje profundo iniciada: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error iniciando sesión de aprendizaje profundo: {e}")
            raise
    
    async def process_deep_learning_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        Procesa una consulta en una sesión de aprendizaje profundo
        
        Args:
            session_id: ID de la sesión
            query: Consulta a procesar
            
        Returns:
            Resultado del procesamiento
        """
        try:
            result = await self.deep_learning_engine.process_query(session_id, query)
            
            logger.info(f"Consulta procesada en sesión {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando consulta en sesión {session_id}: {e}")
            raise
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """
        Obtiene insights de aprendizaje del sistema
        
        Returns:
            Insights de aprendizaje
        """
        try:
            # Insights del analizador de consultas
            query_insights = self.query_analyzer.get_learning_insights()
            
            # Insights del motor de aprendizaje profundo
            deep_learning_analytics = self.deep_learning_engine.get_learning_analytics()
            
            return {
                'query_analyzer_insights': query_insights,
                'deep_learning_analytics': deep_learning_analytics,
                'combined_insights': {
                    'total_queries_analyzed': query_insights.get('total_queries', 0),
                    'active_learning_sessions': deep_learning_analytics.get('active_sessions', 0),
                    'learning_efficiency': deep_learning_analytics.get('performance_metrics', {}).get('learning_efficiency', 0),
                    'insights_generated': deep_learning_analytics.get('total_insights', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo insights de aprendizaje: {e}")
            return {'error': str(e)}
    
    async def generate_adaptive_prompt(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Genera un prompt adaptativo basado en el análisis profundo de la consulta
        
        Args:
            query: Consulta original
            context: Contexto adicional
            
        Returns:
            Prompt adaptativo optimizado
        """
        try:
            # Analizar la consulta
            analysis = self.query_analyzer.analyze_query(query, context)
            
            # Generar prompt base
            base_prompt = f"""Eres LucIA, un asistente especializado en ciberseguridad con capacidades de aprendizaje profundo.

ANÁLISIS DE LA CONSULTA:
- Complejidad: {analysis.complexity.value}
- Categoría: {analysis.category.value}
- Palabras clave: {', '.join(analysis.keywords[:10])}
- Entidades técnicas: {', '.join(analysis.entities[:5])}
- Intención: {analysis.intent}
- Potencial de aprendizaje: {analysis.learning_potential:.2f}

CONTEXTO DE APRENDIZAJE:
- Nivel de experiencia estimado: {self.query_analyzer._estimate_user_level()}
- Temas relacionados: {', '.join(self.query_analyzer._find_related_topics(query))}
- Progresión de complejidad: {self.query_analyzer._calculate_complexity_trend()}

CONSULTA: {query}

INSTRUCCIONES ADAPTATIVAS:
"""
            
            # Agregar instrucciones específicas basadas en el análisis
            if analysis.complexity.value == 'expert':
                base_prompt += """
- Proporciona análisis forense y técnico profundo
- Incluye técnicas avanzadas y metodologías especializadas
- Menciona herramientas de análisis profesional
- Proporciona ejemplos de casos reales complejos
"""
            elif analysis.complexity.value == 'advanced':
                base_prompt += """
- Profundiza en aspectos técnicos avanzados
- Incluye análisis de arquitectura y diseño
- Menciona consideraciones de rendimiento y escalabilidad
- Proporciona ejemplos de código y configuraciones
"""
            elif analysis.complexity.value == 'intermediate':
                base_prompt += """
- Proporciona información técnica pero accesible
- Incluye ejemplos de implementación práctica
- Menciona mejores prácticas y estándares
- Explica el "por qué" detrás de las recomendaciones
"""
            else:  # basic
                base_prompt += """
- Explica de manera simple y clara
- Usa ejemplos prácticos y cotidianos
- Evita jerga técnica innecesaria
- Proporciona pasos claros y accesibles
"""
            
            # Agregar instrucciones específicas por categoría
            category_instructions = {
                'authentication': "Enfócate en mecanismos de autenticación seguros, OAuth 2.0, SAML, JWT, 2FA, MFA y vulnerabilidades comunes.",
                'encryption': "Profundiza en algoritmos de cifrado (AES, RSA, ECC), protocolos SSL/TLS, HTTPS y gestión de claves.",
                'malware': "Analiza técnicas de detección y prevención, sandboxing, análisis de comportamiento y herramientas forenses.",
                'phishing': "Explica técnicas de detección y prevención, SPF, DKIM, DMARC y educación del usuario.",
                'firewall': "Profundiza en configuración y reglas, NGFW, WAF, segmentación de red y monitoreo.",
                'vulnerability': "Analiza gestión de vulnerabilidades, CVE, CVSS, parcheo y herramientas de escaneo."
            }
            
            category_instruction = category_instructions.get(analysis.category.value, "")
            if category_instruction:
                base_prompt += f"\n- {category_instruction}"
            
            # Agregar instrucciones finales
            base_prompt += f"""

RESPUESTA REQUERIDA:
- Responde en español de manera técnica pero accesible
- Adapta el nivel técnico a la complejidad identificada ({analysis.complexity.value})
- Enfócate en la categoría de ciberseguridad: {analysis.category.value}
- Incluye ejemplos prácticos y código cuando sea relevante
- Proporciona información accionable y específica
- Menciona herramientas específicas y recursos adicionales

CONSULTA: {query}"""
            
            return base_prompt
            
        except Exception as e:
            logger.error(f"Error generando prompt adaptativo: {e}")
            return f"Analiza esta consulta de ciberseguridad: {query}"
    
    async def save_learning_data(self):
        """Guarda todos los datos de aprendizaje"""
        try:
            # Guardar datos del analizador de consultas
            self.query_analyzer.save_analysis_data()
            
            # Guardar datos del motor de aprendizaje profundo
            self.deep_learning_engine.save_learning_data()
            
            logger.info("Datos de aprendizaje guardados correctamente")
            
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")
    
    async def load_learning_data(self):
        """Carga todos los datos de aprendizaje"""
        try:
            # Cargar datos del analizador de consultas
            self.query_analyzer.load_analysis_data()
            
            logger.info("Datos de aprendizaje cargados correctamente")
            
        except Exception as e:
            logger.error(f"Error cargando datos de aprendizaje: {e}")
