"""
Integración del Sistema de Resolución de Errores con LucIA
Conecta el sistema de resolución de errores con el motor principal.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .error_resolution_system import ErrorResolutionSystem

logger = logging.getLogger('ErrorResolutionIntegration')

class ErrorResolutionIntegration:
    """
    Integración que conecta el sistema de resolución de errores con LucIA.
    """
    
    def __init__(self):
        self.error_resolution_system = ErrorResolutionSystem()
        self.is_initialized = False
        self.auto_fix_enabled = True
        self.auto_validation_enabled = True
        
    async def initialize(self, lucia_core) -> bool:
        """
        Inicializa la integración con el core de LucIA.
        
        Args:
            lucia_core: Instancia del core de LucIA
            
        Returns:
            True si se inicializó correctamente
        """
        try:
            logger.info("Inicializando integración de resolución de errores...")
            
            # Inicializar sistema de resolución
            success = await self.error_resolution_system.initialize()
            if not success:
                logger.error("Error inicializando sistema de resolución")
                return False
            
            # Aplicar correcciones automáticas si está habilitado
            if self.auto_fix_enabled:
                await self.apply_automatic_fixes(lucia_core)
            
            # Ejecutar validación automática si está habilitado
            if self.auto_validation_enabled:
                await self.run_automatic_validation(lucia_core)
            
            self.is_initialized = True
            logger.info("Integración de resolución de errores inicializada correctamente")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando integración: {e}")
            return False
    
    async def apply_automatic_fixes(self, lucia_core) -> Dict[str, Any]:
        """
        Aplica correcciones automáticas a LucIA.
        
        Args:
            lucia_core: Instancia del core de LucIA
            
        Returns:
            Reporte de correcciones aplicadas
        """
        try:
            logger.info("Aplicando correcciones automáticas...")
            
            # Ejecutar corrección de errores críticos
            fix_report = await self.error_resolution_system.fix_critical_errors(lucia_core)
            
            if fix_report.get('success', False):
                logger.info(f"Correcciones automáticas aplicadas: {len(fix_report.get('fixes_applied', []))} correcciones")
            else:
                logger.warning("Algunas correcciones automáticas fallaron")
            
            return fix_report
            
        except Exception as e:
            logger.error(f"Error aplicando correcciones automáticas: {e}")
            return {'error': str(e), 'success': False}
    
    async def run_automatic_validation(self, lucia_core) -> Dict[str, Any]:
        """
        Ejecuta validación automática del sistema.
        
        Args:
            lucia_core: Instancia del core de LucIA
            
        Returns:
            Resultados de la validación
        """
        try:
            logger.info("Ejecutando validación automática...")
            
            # Ejecutar validación completa
            validation_results = await self.error_resolution_system.system_validator.validate_all_systems(lucia_core)
            
            # Verificar si hay problemas críticos
            overall_status = validation_results.get('overall_status', 'UNKNOWN')
            if overall_status == 'ERROR':
                logger.warning("Se detectaron errores críticos en la validación")
                # Aplicar correcciones adicionales
                await self.apply_automatic_fixes(lucia_core)
            elif overall_status == 'WARNING':
                logger.info("Validación completada con advertencias")
            else:
                logger.info("Validación completada exitosamente")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error ejecutando validación automática: {e}")
            return {'error': str(e)}
    
    async def fix_specific_error(self, lucia_core, error_type: str, error_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Corrige un error específico.
        
        Args:
            lucia_core: Instancia del core de LucIA
            error_type: Tipo de error a corregir
            error_context: Contexto del error
            
        Returns:
            Resultado de la corrección
        """
        try:
            logger.info(f"Corrigiendo error específico: {error_type}")
            
            if error_type == 'async_sync':
                fixes = await self.error_resolution_system._fix_async_sync_issues(lucia_core)
                return {'type': 'async_sync', 'fixes': fixes, 'success': len(fixes) > 0}
            
            elif error_type == 'tensorflow':
                optimizations = await self.error_resolution_system._optimize_tensorflow_systems(lucia_core)
                return {'type': 'tensorflow', 'optimizations': optimizations, 'success': len(optimizations) > 0}
            
            elif error_type == 'predictions':
                enhancements = await self.error_resolution_system._enhance_predictions(lucia_core)
                return {'type': 'predictions', 'enhancements': enhancements, 'success': len(enhancements) > 0}
            
            elif error_type == 'validation':
                validation = await self.error_resolution_system._validate_all_systems(lucia_core)
                return {'type': 'validation', 'results': validation, 'success': True}
            
            else:
                return {'type': error_type, 'error': 'Tipo de error no soportado', 'success': False}
                
        except Exception as e:
            logger.error(f"Error corrigiendo error específico {error_type}: {e}")
            return {'type': error_type, 'error': str(e), 'success': False}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema de resolución de errores"""
        try:
            return {
                'is_initialized': self.is_initialized,
                'auto_fix_enabled': self.auto_fix_enabled,
                'auto_validation_enabled': self.auto_validation_enabled,
                'resolution_status': self.error_resolution_system.get_resolution_status(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {'error': str(e)}
    
    def enable_auto_fix(self, enabled: bool = True):
        """Habilita o deshabilita las correcciones automáticas"""
        self.auto_fix_enabled = enabled
        logger.info(f"Correcciones automáticas {'habilitadas' if enabled else 'deshabilitadas'}")
    
    def enable_auto_validation(self, enabled: bool = True):
        """Habilita o deshabilita la validación automática"""
        self.auto_validation_enabled = enabled
        logger.info(f"Validación automática {'habilitada' if enabled else 'deshabilitada'}")
    
    async def shutdown(self):
        """Apaga el sistema de integración"""
        try:
            await self.error_resolution_system.shutdown()
            self.is_initialized = False
            logger.info("Integración de resolución de errores apagada")
            
        except Exception as e:
            logger.error(f"Error apagando integración: {e}")

# Función de conveniencia para integración rápida
async def integrate_error_resolution(lucia_core) -> ErrorResolutionIntegration:
    """
    Función de conveniencia para integrar rápidamente el sistema de resolución de errores.
    
    Args:
        lucia_core: Instancia del core de LucIA
        
    Returns:
        Instancia de la integración
    """
    integration = ErrorResolutionIntegration()
    await integration.initialize(lucia_core)
    return integration
