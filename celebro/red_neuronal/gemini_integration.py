"""
Integración con Gemini API para @red_neuronal
Versión: 0.6.0
Integración con Google Gemini para análisis y generación de contenido
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time

from .config_simple import get_gemini_api_key, get_gemini_config

logger = logging.getLogger('Neural_Gemini')

class GeminiIntegration:
    """Integración con Google Gemini API"""
    
    def __init__(self):
        self.api_key = get_gemini_api_key()
        self.config = get_gemini_config()
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'LucIA-NeuralNetwork/0.6.0'
        })
        
        logger.info("Integración con Gemini API inicializada")
    
    def generate_text(self, prompt: str, max_tokens: int = None, 
                     temperature: float = None, **kwargs) -> Dict[str, Any]:
        """
        Genera texto usando Gemini API
        
        Args:
            prompt: Texto de entrada
            max_tokens: Máximo número de tokens
            temperature: Temperatura de generación
            **kwargs: Parámetros adicionales
        
        Returns:
            Respuesta de la API
        """
        try:
            # Configurar parámetros
            max_tokens = max_tokens or self.config['max_tokens']
            temperature = temperature or self.config['temperature']
            
            # Preparar payload
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "topP": kwargs.get('top_p', 0.8),
                    "topK": kwargs.get('top_k', 40)
                }
            }
            
            # Realizar petición
            url = f"{self.base_url}/models/{self.config['model']}:generateContent"
            params = {'key': self.api_key}
            
            response = self.session.post(
                url, 
                json=payload, 
                params=params,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'text': result['candidates'][0]['content']['parts'][0]['text'],
                    'usage': result.get('usageMetadata', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"Error {response.status_code}: {response.text}",
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error generando texto con Gemini: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def unified_analysis(self, analysis_type: str, data: Dict[str, Any], 
                        custom_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Función unificada para todos los tipos de análisis y generación de contenido
        
        Args:
            analysis_type: Tipo de análisis ('network', 'architecture', 'results', 'advice', 'custom')
            data: Datos específicos para el análisis
            custom_prompt: Prompt personalizado (opcional)
            **kwargs: Parámetros adicionales
        
        Returns:
            Resultado del análisis
        """
        try:
            # Configuración base
            base_config = {
                'max_tokens': kwargs.get('max_tokens', 1024),
                'temperature': kwargs.get('temperature', 0.3),
                'top_p': kwargs.get('top_p', 0.8),
                'top_k': kwargs.get('top_k', 40)
            }
            
            # Generar prompt según el tipo de análisis
            if custom_prompt:
                prompt = custom_prompt
            elif analysis_type == 'network':
                prompt = self._generate_network_analysis_prompt(data)
            elif analysis_type == 'architecture':
                prompt = self._generate_architecture_prompt(data)
            elif analysis_type == 'results':
                prompt = self._generate_results_prompt(data)
            elif analysis_type == 'advice':
                prompt = self._generate_advice_prompt(data)
            elif analysis_type == 'custom':
                prompt = data.get('prompt', '')
            else:
                raise ValueError(f"Tipo de análisis no válido: {analysis_type}")
            
            # Generar respuesta
            result = self.generate_text(
                prompt, 
                max_tokens=base_config['max_tokens'],
                temperature=base_config['temperature'],
                top_p=base_config['top_p'],
                top_k=base_config['top_k']
            )
            
            if result['success']:
                # Preparar respuesta unificada
                response = {
                    'success': True,
                    'analysis_type': analysis_type,
                    'timestamp': datetime.now().isoformat(),
                    'usage': result.get('usage', {}),
                    'response_text': result['text']
                }
                
                # Agregar datos específicos según el tipo
                if analysis_type == 'network':
                    response.update({
                        'analysis': result['text'],
                        'network_config': data
                    })
                elif analysis_type == 'architecture':
                    response.update({
                        'suggestion': result['text'],
                        'problem_type': data.get('problem_type'),
                        'input_size': data.get('input_size'),
                        'output_size': data.get('output_size'),
                        'data_size': data.get('data_size')
                    })
                elif analysis_type == 'results':
                    response.update({
                        'explanation': result['text'],
                        'metrics': data
                    })
                elif analysis_type == 'advice':
                    response.update({
                        'advice': result['text'],
                        'problem_description': data.get('problem_description', '')
                    })
                else:
                    response['content'] = result['text']
                
                return response
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error en análisis unificado: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_network_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Genera prompt para análisis de red neuronal"""
        return f"""
        Analiza la siguiente configuración de red neuronal y proporciona recomendaciones:
        
        Configuración:
        - Capas ocultas: {data.get('hidden_layers', [])}
        - Función de activación: {data.get('activation', 'relu')}
        - Tasa de aprendizaje: {data.get('learning_rate', 0.001)}
        - Tamaño de lote: {data.get('batch_size', 32)}
        - Épocas: {data.get('epochs', 100)}
        - Dropout: {data.get('dropout_rate', 0.0)}
        
        Por favor proporciona:
        1. Evaluación de la arquitectura
        2. Recomendaciones de mejora
        3. Posibles problemas
        4. Sugerencias de optimización
        """
    
    def _generate_architecture_prompt(self, data: Dict[str, Any]) -> str:
        """Genera prompt para sugerencia de arquitectura"""
        return f"""
        Sugiere una arquitectura de red neuronal para:
        - Tipo de problema: {data.get('problem_type', 'classification')}
        - Tamaño de entrada: {data.get('input_size', 0)}
        - Tamaño de salida: {data.get('output_size', 0)}
        - Tamaño del dataset: {data.get('data_size', 0)}
        
        Proporciona:
        1. Número de capas ocultas recomendadas
        2. Tamaño de cada capa
        3. Funciones de activación
        4. Tasa de aprendizaje
        5. Tamaño de lote
        6. Técnicas de regularización
        7. Estrategia de entrenamiento
        """
    
    def _generate_results_prompt(self, data: Dict[str, Any]) -> str:
        """Genera prompt para explicación de resultados"""
        return f"""
        Explica los siguientes resultados de entrenamiento de una red neuronal:
        
        Métricas:
        - Pérdida de entrenamiento: {data.get('train_loss', 'N/A')}
        - Precisión de entrenamiento: {data.get('train_accuracy', 'N/A')}
        - Pérdida de validación: {data.get('val_loss', 'N/A')}
        - Precisión de validación: {data.get('val_accuracy', 'N/A')}
        - Épocas entrenadas: {data.get('epochs', 'N/A')}
        
        Proporciona:
        1. Interpretación de las métricas
        2. Análisis de overfitting/underfitting
        3. Recomendaciones de mejora
        4. Próximos pasos sugeridos
        """
    
    def _generate_advice_prompt(self, data: Dict[str, Any]) -> str:
        """Genera prompt para consejos de entrenamiento"""
        return f"""
        Proporciona consejos de entrenamiento para el siguiente problema de machine learning:
        
        Descripción: {data.get('problem_description', '')}
        
        Incluye:
        1. Estrategia de preprocesamiento de datos
        2. Arquitectura de red recomendada
        3. Hiperparámetros sugeridos
        4. Técnicas de regularización
        5. Estrategia de validación
        6. Métricas a monitorear
        7. Señales de overfitting/underfitting
        8. Consejos de optimización
        """
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba la conexión con Gemini API
        
        Returns:
            Estado de la conexión
        """
        try:
            # Prueba simple
            result = self.generate_text("Hola, ¿puedes responder con 'Conexión exitosa'?", 
                                      max_tokens=50, temperature=0.1)
            
            if result['success']:
                return {
                    'success': True,
                    'status': 'Conectado',
                    'response': result['text'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'status': 'Error de conexión',
                    'error': result['error'],
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error probando conexión: {e}")
            return {
                'success': False,
                'status': 'Error de conexión',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso de la API
        
        Returns:
            Estadísticas de uso
        """
        try:
            # Esta funcionalidad dependería de la API específica de Gemini
            # Por ahora retornamos información básica
            return {
                'success': True,
                'api_key_configured': bool(self.api_key),
                'model': self.config['model'],
                'max_tokens': self.config['max_tokens'],
                'temperature': self.config['temperature'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Instancia global de integración
gemini_integration = GeminiIntegration()

# Funciones de conveniencia unificadas
def unified_analysis(analysis_type: str, data: Dict[str, Any], 
                    custom_prompt: str = None, **kwargs) -> Dict[str, Any]:
    """
    Función unificada para todos los tipos de análisis
    
    Args:
        analysis_type: 'network', 'architecture', 'results', 'advice', 'custom'
        data: Datos específicos para el análisis
        custom_prompt: Prompt personalizado (opcional)
        **kwargs: Parámetros adicionales (max_tokens, temperature, etc.)
    
    Returns:
        Resultado del análisis
    """
    return gemini_integration.unified_analysis(analysis_type, data, custom_prompt, **kwargs)

def analyze_network(network_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analiza una configuración de red neuronal (función de conveniencia)"""
    return gemini_integration.unified_analysis('network', network_config)

def suggest_architecture(problem_type: str, input_size: int, 
                        output_size: int, data_size: int) -> Dict[str, Any]:
    """Sugiere una arquitectura de red neuronal (función de conveniencia)"""
    data = {
        'problem_type': problem_type,
        'input_size': input_size,
        'output_size': output_size,
        'data_size': data_size
    }
    return gemini_integration.unified_analysis('architecture', data)

def explain_results(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Explica los resultados de entrenamiento (función de conveniencia)"""
    return gemini_integration.unified_analysis('results', metrics)

def generate_training_advice(problem_description: str) -> Dict[str, Any]:
    """Genera consejos de entrenamiento (función de conveniencia)"""
    data = {'problem_description': problem_description}
    return gemini_integration.unified_analysis('advice', data)

def custom_analysis(prompt: str, **kwargs) -> Dict[str, Any]:
    """Análisis personalizado con prompt específico"""
    data = {'prompt': prompt}
    return gemini_integration.unified_analysis('custom', data, **kwargs)

def test_gemini_connection() -> Dict[str, Any]:
    """Prueba la conexión con Gemini"""
    return gemini_integration.test_connection()

def get_gemini_usage() -> Dict[str, Any]:
    """Obtiene estadísticas de uso"""
    return gemini_integration.get_usage_stats()
