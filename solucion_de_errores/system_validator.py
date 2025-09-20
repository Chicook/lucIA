"""
Validador de Sistemas
Valida la integridad y funcionamiento de todos los sistemas integrados.
"""

import logging
import asyncio
import importlib
import inspect
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import traceback
import json
import os

logger = logging.getLogger('SystemValidator')

class SystemValidator:
    """
    Validador que verifica la integridad y funcionamiento de todos los sistemas.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.validation_metrics = {
            'modules_validated': 0,
            'systems_checked': 0,
            'errors_found': 0,
            'warnings_found': 0,
            'fixes_applied': 0,
            'validation_time': 0
        }
        
        self.validation_criteria = {
            'module_initialization': True,
            'method_availability': True,
            'dependency_check': True,
            'configuration_validation': True,
            'performance_check': True,
            'integration_test': True
        }
    
    async def validate_all_systems(self, lucia_core) -> Dict[str, Any]:
        """
        Valida todos los sistemas integrados en LucIA.
        
        Args:
            lucia_core: Instancia del core de LucIA
            
        Returns:
            Reporte completo de validación
        """
        start_time = datetime.now()
        
        try:
            logger.info("Iniciando validación completa de sistemas...")
            
            # Validar módulos principales
            modules_validation = await self._validate_modules(lucia_core)
            
            # Validar sistemas especializados
            specialized_systems = await self._validate_specialized_systems(lucia_core)
            
            # Validar integraciones externas
            external_integrations = await self._validate_external_integrations(lucia_core)
            
            # Validar configuración
            config_validation = await self._validate_configuration(lucia_core)
            
            # Validar rendimiento
            performance_validation = await self._validate_performance(lucia_core)
            
            # Compilar resultados
            end_time = datetime.now()
            validation_time = (end_time - start_time).total_seconds()
            self.validation_metrics['validation_time'] = validation_time
            
            validation_report = {
                'timestamp': start_time.isoformat(),
                'validation_time': validation_time,
                'modules_validation': modules_validation,
                'specialized_systems': specialized_systems,
                'external_integrations': external_integrations,
                'configuration_validation': config_validation,
                'performance_validation': performance_validation,
                'overall_status': self._calculate_overall_status(),
                'metrics': self.validation_metrics,
                'recommendations': self._generate_validation_recommendations()
            }
            
            self.validation_results = validation_report
            logger.info(f"Validación completada en {validation_time:.2f} segundos")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error en validación de sistemas: {e}")
            return {
                'error': str(e),
                'timestamp': start_time.isoformat(),
                'status': 'FAILED'
            }
    
    async def _validate_modules(self, lucia_core) -> Dict[str, Any]:
        """Valida todos los módulos principales"""
        try:
            modules_status = {}
            expected_modules = [
                'memory', 'learning', 'communication', 'training',
                'reasoning', 'perception', 'action', 'evaluation',
                'optimization', 'security', 'monitoring', 'integration',
                'advanced_integration', 'infrastructure'
            ]
            
            for module_name in expected_modules:
                try:
                    if module_name in lucia_core.modules:
                        module = lucia_core.modules[module_name]
                        
                        # Verificar inicialización
                        is_initialized = getattr(module, 'is_initialized', False)
                        
                        # Verificar métodos críticos
                        critical_methods = self._get_critical_methods(module_name)
                        available_methods = []
                        
                        for method_name in critical_methods:
                            if hasattr(module, method_name):
                                available_methods.append(method_name)
                        
                        # Verificar dependencias
                        dependencies_ok = await self._check_module_dependencies(module)
                        
                        modules_status[module_name] = {
                            'status': 'ACTIVE' if is_initialized else 'INACTIVE',
                            'initialized': is_initialized,
                            'critical_methods': available_methods,
                            'missing_methods': [m for m in critical_methods if m not in available_methods],
                            'dependencies_ok': dependencies_ok,
                            'health_score': self._calculate_module_health_score(
                                is_initialized, available_methods, dependencies_ok
                            )
                        }
                        
                        self.validation_metrics['modules_validated'] += 1
                        
                    else:
                        modules_status[module_name] = {
                            'status': 'NOT_LOADED',
                            'initialized': False,
                            'critical_methods': [],
                            'missing_methods': [],
                            'dependencies_ok': False,
                            'health_score': 0.0
                        }
                        self.validation_metrics['warnings_found'] += 1
                        
                except Exception as e:
                    logger.error(f"Error validando módulo {module_name}: {e}")
                    modules_status[module_name] = {
                        'status': 'ERROR',
                        'error': str(e),
                        'health_score': 0.0
                    }
                    self.validation_metrics['errors_found'] += 1
            
            return modules_status
            
        except Exception as e:
            logger.error(f"Error en validación de módulos: {e}")
            return {'error': str(e)}
    
    def _get_critical_methods(self, module_name: str) -> List[str]:
        """Obtiene métodos críticos para cada módulo"""
        critical_methods = {
            'memory': ['store', 'retrieve', 'search'],
            'learning': ['learn', 'adapt', 'process_learning_cycle'],
            'communication': ['send_message', 'receive_message'],
            'training': ['train', 'external_training'],
            'reasoning': ['reason', 'deduce', 'induce'],
            'perception': ['perceive', 'process_input'],
            'action': ['execute', 'perform_action'],
            'evaluation': ['evaluate', 'assess'],
            'optimization': ['optimize', 'improve'],
            'security': ['authenticate', 'authorize'],
            'monitoring': ['monitor', 'get_metrics'],
            'integration': ['integrate', 'connect'],
            'advanced_integration': ['get_system_status', 'celebro_core', 'red_neuronal_core'],
            'infrastructure': ['get_system_status', 'optimize_systems']
        }
        
        return critical_methods.get(module_name, [])
    
    async def _check_module_dependencies(self, module) -> bool:
        """Verifica dependencias de un módulo"""
        try:
            # Verificar imports críticos
            critical_imports = [
                'numpy', 'pandas', 'tensorflow', 'sklearn',
                'asyncio', 'logging', 'json', 'datetime'
            ]
            
            missing_imports = []
            for import_name in critical_imports:
                try:
                    importlib.import_module(import_name)
                except ImportError:
                    missing_imports.append(import_name)
            
            if missing_imports:
                logger.warning(f"Dependencias faltantes: {missing_imports}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando dependencias: {e}")
            return False
    
    def _calculate_module_health_score(self, initialized: bool, 
                                     available_methods: List[str], 
                                     dependencies_ok: bool) -> float:
        """Calcula puntaje de salud de un módulo"""
        try:
            score = 0.0
            
            # Puntaje por inicialización
            if initialized:
                score += 0.4
            
            # Puntaje por métodos disponibles
            if available_methods:
                method_score = len(available_methods) / max(len(available_methods) + 2, 1)
                score += method_score * 0.4
            
            # Puntaje por dependencias
            if dependencies_ok:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculando puntaje de salud: {e}")
            return 0.0
    
    async def _validate_specialized_systems(self, lucia_core) -> Dict[str, Any]:
        """Valida sistemas especializados (@celebro, @red_neuronal, etc.)"""
        try:
            specialized_status = {}
            
            # Validar @celebro
            if 'advanced_integration' in lucia_core.modules:
                celebro_status = await self._validate_celebro_system(lucia_core)
                specialized_status['celebro'] = celebro_status
            
            # Validar @red_neuronal
            if 'advanced_integration' in lucia_core.modules:
                neural_status = await self._validate_neural_system(lucia_core)
                specialized_status['red_neuronal'] = neural_status
            
            # Validar TensorFlow
            tensorflow_status = await self._validate_tensorflow_system(lucia_core)
            specialized_status['tensorflow'] = tensorflow_status
            
            # Validar Gemini
            gemini_status = await self._validate_gemini_system(lucia_core)
            specialized_status['gemini'] = gemini_status
            
            self.validation_metrics['systems_checked'] += len(specialized_status)
            
            return specialized_status
            
        except Exception as e:
            logger.error(f"Error validando sistemas especializados: {e}")
            return {'error': str(e)}
    
    async def _validate_celebro_system(self, lucia_core) -> Dict[str, Any]:
        """Valida el sistema @celebro"""
        try:
            if not hasattr(lucia_core.modules['advanced_integration'], 'celebro_core'):
                return {'status': 'NOT_AVAILABLE', 'error': 'Celebro core no encontrado'}
            
            celebro_core = lucia_core.modules['advanced_integration'].celebro_core
            
            # Verificar métodos críticos
            critical_methods = ['analyze_response', 'generate_alternative_response', 'synthesize_knowledge']
            available_methods = [m for m in critical_methods if hasattr(celebro_core, m)]
            
            # Verificar estado del sistema
            system_status = celebro_core.get_system_status() if hasattr(celebro_core, 'get_system_status') else {}
            
            return {
                'status': 'ACTIVE' if available_methods else 'INACTIVE',
                'available_methods': available_methods,
                'missing_methods': [m for m in critical_methods if m not in available_methods],
                'system_status': system_status,
                'health_score': len(available_methods) / len(critical_methods)
            }
            
        except Exception as e:
            logger.error(f"Error validando @celebro: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _validate_neural_system(self, lucia_core) -> Dict[str, Any]:
        """Valida el sistema @red_neuronal"""
        try:
            if not hasattr(lucia_core.modules['advanced_integration'], 'red_neuronal_core'):
                return {'status': 'NOT_AVAILABLE', 'error': 'Neural core no encontrado'}
            
            neural_core = lucia_core.modules['advanced_integration'].red_neuronal_core
            
            # Verificar métodos críticos
            critical_methods = ['analyze_query_deep', 'generate_adaptive_prompt', 'get_learning_insights']
            available_methods = [m for m in critical_methods if hasattr(neural_core, m)]
            
            return {
                'status': 'ACTIVE' if available_methods else 'INACTIVE',
                'available_methods': available_methods,
                'missing_methods': [m for m in critical_methods if m not in available_methods],
                'health_score': len(available_methods) / len(critical_methods)
            }
            
        except Exception as e:
            logger.error(f"Error validando @red_neuronal: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _validate_tensorflow_system(self, lucia_core) -> Dict[str, Any]:
        """Valida el sistema TensorFlow"""
        try:
            # Verificar disponibilidad de TensorFlow
            try:
                import tensorflow as tf
                tf_version = tf.__version__
                tf_available = True
            except ImportError:
                tf_version = None
                tf_available = False
            
            # Verificar integración con @celebro
            celebro_tf_available = False
            if ('advanced_integration' in lucia_core.modules and 
                hasattr(lucia_core.modules['advanced_integration'], 'celebro_core')):
                celebro_core = lucia_core.modules['advanced_integration'].celebro_core
                if hasattr(celebro_core, 'tensorflow_integration'):
                    celebro_tf_available = True
            
            return {
                'status': 'ACTIVE' if tf_available and celebro_tf_available else 'INACTIVE',
                'tensorflow_available': tf_available,
                'tensorflow_version': tf_version,
                'celebro_integration': celebro_tf_available,
                'health_score': 1.0 if tf_available and celebro_tf_available else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error validando TensorFlow: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _validate_gemini_system(self, lucia_core) -> Dict[str, Any]:
        """Valida el sistema Gemini"""
        try:
            # Verificar integración de Gemini
            gemini_available = False
            gemini_connected = False
            
            if hasattr(lucia_core, 'gemini_integration') and lucia_core.gemini_integration:
                gemini_available = True
                try:
                    gemini_connected = lucia_core.gemini_integration.test_connection()
                except Exception as e:
                    logger.warning(f"Error probando conexión Gemini: {e}")
            
            return {
                'status': 'ACTIVE' if gemini_available and gemini_connected else 'INACTIVE',
                'gemini_available': gemini_available,
                'gemini_connected': gemini_connected,
                'health_score': 1.0 if gemini_available and gemini_connected else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error validando Gemini: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _validate_external_integrations(self, lucia_core) -> Dict[str, Any]:
        """Valida integraciones externas"""
        try:
            integrations_status = {}
            
            # Verificar APIs externas
            apis_to_check = ['openai', 'claude', 'gemini']
            
            for api_name in apis_to_check:
                try:
                    # Verificar si hay configuración para la API
                    api_available = False
                    if api_name == 'gemini' and hasattr(lucia_core, 'gemini_integration'):
                        api_available = lucia_core.gemini_integration is not None
                    
                    integrations_status[api_name] = {
                        'available': api_available,
                        'configured': api_available
                    }
                    
                except Exception as e:
                    integrations_status[api_name] = {
                        'available': False,
                        'configured': False,
                        'error': str(e)
                    }
            
            return integrations_status
            
        except Exception as e:
            logger.error(f"Error validando integraciones externas: {e}")
            return {'error': str(e)}
    
    async def _validate_configuration(self, lucia_core) -> Dict[str, Any]:
        """Valida la configuración del sistema"""
        try:
            config_status = {
                'config_loaded': bool(lucia_core.config),
                'modules_configured': len(lucia_core.config.get('modules', {})),
                'training_enabled': lucia_core.config.get('training', {}).get('auto_learning', False),
                'security_enabled': lucia_core.config.get('security', {}).get('encryption_enabled', False)
            }
            
            # Verificar archivos de configuración
            config_files = ['config/ai_config.json']
            for config_file in config_files:
                config_status[f'{config_file}_exists'] = os.path.exists(config_file)
            
            return config_status
            
        except Exception as e:
            logger.error(f"Error validando configuración: {e}")
            return {'error': str(e)}
    
    async def _validate_performance(self, lucia_core) -> Dict[str, Any]:
        """Valida el rendimiento del sistema"""
        try:
            performance_status = {
                'memory_usage': getattr(lucia_core, '_get_memory_usage', lambda: 0.0)(),
                'cpu_usage': getattr(lucia_core, '_get_cpu_usage', lambda: 0.0)(),
                'modules_loaded': len(lucia_core.modules),
                'is_running': lucia_core.is_running
            }
            
            # Verificar métricas de rendimiento
            if hasattr(lucia_core, 'performance_metrics'):
                performance_status['performance_metrics'] = lucia_core.performance_metrics
            
            return performance_status
            
        except Exception as e:
            logger.error(f"Error validando rendimiento: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_status(self) -> str:
        """Calcula el estado general del sistema"""
        try:
            if self.validation_metrics['errors_found'] > 0:
                return 'ERROR'
            elif self.validation_metrics['warnings_found'] > 0:
                return 'WARNING'
            elif self.validation_metrics['modules_validated'] > 0:
                return 'HEALTHY'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            logger.error(f"Error calculando estado general: {e}")
            return 'ERROR'
    
    def _generate_validation_recommendations(self) -> List[str]:
        """Genera recomendaciones basadas en la validación"""
        recommendations = []
        
        if self.validation_metrics['errors_found'] > 0:
            recommendations.append("Corregir errores críticos encontrados")
        
        if self.validation_metrics['warnings_found'] > 0:
            recommendations.append("Revisar advertencias del sistema")
        
        if self.validation_metrics['modules_validated'] < 10:
            recommendations.append("Verificar inicialización de módulos faltantes")
        
        if self.validation_metrics['validation_time'] > 30:
            recommendations.append("Optimizar tiempo de validación")
        
        return recommendations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la validación"""
        return {
            'overall_status': self._calculate_overall_status(),
            'metrics': self.validation_metrics,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_validation_recommendations()
        }
