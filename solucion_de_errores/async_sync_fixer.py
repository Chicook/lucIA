"""
Solucionador de Errores de Sincronización Asíncrona
Corrige problemas de sincronización entre operaciones asíncronas y síncronas.
"""

import asyncio
import logging
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import traceback

logger = logging.getLogger('AsyncSyncFixer')

class AsyncSyncFixer:
    """
    Solucionador de errores de sincronización asíncrona.
    Maneja la conversión entre funciones asíncronas y síncronas de forma segura.
    """
    
    def __init__(self):
        self.fixed_functions = {}
        self.error_log = []
        self.performance_metrics = {
            'async_to_sync_conversions': 0,
            'sync_to_async_conversions': 0,
            'errors_fixed': 0,
            'performance_improvements': 0
        }
        
    def fix_async_sync_issues(self, func: Callable, is_async: bool = True) -> Callable:
        """
        Convierte una función para manejar correctamente la sincronización asíncrona.
        
        Args:
            func: Función a convertir
            is_async: Si la función es asíncrona (True) o síncrona (False)
            
        Returns:
            Función convertida y optimizada
        """
        try:
            if is_async:
                return self._make_async_safe(func)
            else:
                return self._make_sync_safe(func)
        except Exception as e:
            logger.error(f"Error convirtiendo función: {e}")
            return func
    
    def _make_async_safe(self, func: Callable) -> Callable:
        """Convierte una función asíncrona para manejo seguro de errores"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Verificar si estamos en un loop de eventos
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No hay loop, crear uno nuevo
                    return await asyncio.run(func(*args, **kwargs))
                
                # Ejecutar en el loop actual
                result = await func(*args, **kwargs)
                self.performance_metrics['async_to_sync_conversions'] += 1
                return result
                
            except Exception as e:
                self._log_error(func.__name__, e, 'async_safe')
                # Intentar ejecución síncrona como fallback
                try:
                    if asyncio.iscoroutinefunction(func):
                        # Si es corrutina, ejecutar en nuevo loop
                        return await asyncio.run(func(*args, **kwargs))
                    else:
                        return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Error en fallback síncrono: {fallback_error}")
                    return None
                    
        return async_wrapper
    
    def _make_sync_safe(self, func: Callable) -> Callable:
        """Convierte una función síncrona para ejecución asíncrona segura"""
        @functools.wraps(func)
        async def sync_wrapper(*args, **kwargs):
            try:
                # Ejecutar función síncrona en thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
                self.performance_metrics['sync_to_async_conversions'] += 1
                return result
                
            except Exception as e:
                self._log_error(func.__name__, e, 'sync_safe')
                # Intentar ejecución directa como fallback
                try:
                    return func(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Error en fallback directo: {fallback_error}")
                    return None
                    
        return sync_wrapper
    
    def fix_gemini_integration(self, gemini_instance) -> Any:
        """
        Corrige problemas específicos de integración con Gemini.
        
        Args:
            gemini_instance: Instancia de GeminiIntegration
            
        Returns:
            Instancia corregida
        """
        try:
            # Verificar si los métodos son asíncronos o síncronos
            if hasattr(gemini_instance, 'generate_text'):
                original_generate_text = gemini_instance.generate_text
                
                # Hacer el método compatible con ambos tipos de llamada
                def fixed_generate_text(prompt: str, **kwargs):
                    try:
                        # Intentar ejecución síncrona primero
                        if asyncio.iscoroutinefunction(original_generate_text):
                            # Si es asíncrona, ejecutar en loop
                            try:
                                loop = asyncio.get_running_loop()
                                return loop.run_until_complete(original_generate_text(prompt, **kwargs))
                            except RuntimeError:
                                # No hay loop, crear uno nuevo
                                return asyncio.run(original_generate_text(prompt, **kwargs))
                        else:
                            # Si es síncrona, ejecutar directamente
                            return original_generate_text(prompt, **kwargs)
                    except Exception as e:
                        logger.error(f"Error en generate_text: {e}")
                        return {'error': str(e), 'text': ''}
                
                gemini_instance.generate_text = fixed_generate_text
                self.performance_metrics['errors_fixed'] += 1
                
            if hasattr(gemini_instance, 'test_connection'):
                original_test_connection = gemini_instance.test_connection
                
                def fixed_test_connection():
                    try:
                        if asyncio.iscoroutinefunction(original_test_connection):
                            try:
                                loop = asyncio.get_running_loop()
                                return loop.run_until_complete(original_test_connection())
                            except RuntimeError:
                                return asyncio.run(original_test_connection())
                        else:
                            return original_test_connection()
                    except Exception as e:
                        logger.error(f"Error en test_connection: {e}")
                        return False
                
                gemini_instance.test_connection = fixed_test_connection
                self.performance_metrics['errors_fixed'] += 1
                
            logger.info("Integración de Gemini corregida exitosamente")
            return gemini_instance
            
        except Exception as e:
            logger.error(f"Error corrigiendo integración de Gemini: {e}")
            return gemini_instance
    
    def fix_tensorflow_async_issues(self, tensorflow_instance) -> Any:
        """
        Corrige problemas de sincronización en operaciones de TensorFlow.
        
        Args:
            tensorflow_instance: Instancia de TensorFlowCelebroIntegration
            
        Returns:
            Instancia corregida
        """
        try:
            # Corregir métodos de entrenamiento
            if hasattr(tensorflow_instance, 'train_model'):
                original_train_model = tensorflow_instance.train_model
                
                def fixed_train_model(model_id: str, training_data, labels, validation_data=None):
                    try:
                        # Ejecutar entrenamiento en thread separado para evitar bloqueos
                        import concurrent.futures
                        
                        def sync_train():
                            return original_train_model(model_id, training_data, labels, validation_data)
                        
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(sync_train)
                            return future.result(timeout=300)  # 5 minutos timeout
                            
                    except Exception as e:
                        logger.error(f"Error en entrenamiento TensorFlow: {e}")
                        return None
                
                tensorflow_instance.train_model = fixed_train_model
                self.performance_metrics['errors_fixed'] += 1
            
            # Corregir métodos de predicción
            if hasattr(tensorflow_instance, 'predict'):
                original_predict = tensorflow_instance.predict
                
                def fixed_predict(model_id: str, text: str):
                    try:
                        # Ejecutar predicción de forma síncrona pero segura
                        return original_predict(model_id, text)
                    except Exception as e:
                        logger.error(f"Error en predicción TensorFlow: {e}")
                        return {'error': str(e)}
                
                tensorflow_instance.predict = fixed_predict
                self.performance_metrics['errors_fixed'] += 1
                
            logger.info("Operaciones de TensorFlow corregidas exitosamente")
            return tensorflow_instance
            
        except Exception as e:
            logger.error(f"Error corrigiendo TensorFlow: {e}")
            return tensorflow_instance
    
    def fix_celebro_async_issues(self, celebro_instance) -> Any:
        """
        Corrige problemas de sincronización en el sistema @celebro.
        
        Args:
            celebro_instance: Instancia de CelebroCore
            
        Returns:
            Instancia corregida
        """
        try:
            # Corregir métodos de análisis
            if hasattr(celebro_instance, 'analyze_response'):
                original_analyze_response = celebro_instance.analyze_response
                
                def fixed_analyze_response(response: str, context: Dict = None):
                    try:
                        if asyncio.iscoroutinefunction(original_analyze_response):
                            try:
                                loop = asyncio.get_running_loop()
                                return loop.run_until_complete(original_analyze_response(response, context))
                            except RuntimeError:
                                return asyncio.run(original_analyze_response(response, context))
                        else:
                            return original_analyze_response(response, context)
                    except Exception as e:
                        logger.error(f"Error en analyze_response: {e}")
                        return {'error': str(e)}
                
                celebro_instance.analyze_response = fixed_analyze_response
                self.performance_metrics['errors_fixed'] += 1
            
            logger.info("Sistema @celebro corregido exitosamente")
            return celebro_instance
            
        except Exception as e:
            logger.error(f"Error corrigiendo @celebro: {e}")
            return celebro_instance
    
    def _log_error(self, function_name: str, error: Exception, fix_type: str):
        """Registra errores para análisis posterior"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': function_name,
            'error': str(error),
            'fix_type': fix_type,
            'traceback': traceback.format_exc()
        }
        self.error_log.append(error_entry)
        logger.error(f"Error en {function_name} ({fix_type}): {error}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento del solucionador"""
        return {
            **self.performance_metrics,
            'total_errors_logged': len(self.error_log),
            'fix_success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calcula la tasa de éxito de las correcciones"""
        total_attempts = (self.performance_metrics['async_to_sync_conversions'] + 
                         self.performance_metrics['sync_to_async_conversions'])
        if total_attempts == 0:
            return 0.0
        return (self.performance_metrics['errors_fixed'] / total_attempts) * 100
    
    def get_error_report(self) -> Dict[str, Any]:
        """Genera reporte de errores encontrados y corregidos"""
        return {
            'total_errors': len(self.error_log),
            'recent_errors': self.error_log[-10:] if self.error_log else [],
            'performance_metrics': self.get_performance_metrics(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en los errores encontrados"""
        recommendations = []
        
        if self.performance_metrics['async_to_sync_conversions'] > 10:
            recommendations.append("Considerar usar más operaciones asíncronas nativas")
        
        if self.performance_metrics['sync_to_async_conversions'] > 10:
            recommendations.append("Optimizar funciones síncronas para mejor rendimiento")
        
        if len(self.error_log) > 5:
            recommendations.append("Implementar más validaciones de entrada")
        
        return recommendations
