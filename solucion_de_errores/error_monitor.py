"""
Monitor de Errores
Monitorea, detecta y reporta errores en tiempo real en todos los sistemas.
"""

import logging
import asyncio
import traceback
import json
import os
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time

logger = logging.getLogger('ErrorMonitor')

class ErrorMonitor:
    """
    Monitor de errores que detecta, clasifica y reporta problemas en tiempo real.
    """
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history = deque(maxlen=max_error_history)
        self.error_counts = defaultdict(int)
        self.error_categories = defaultdict(list)
        self.alert_thresholds = {
            'critical': 5,      # Errores críticos en 1 hora
            'warning': 20,      # Advertencias en 1 hora
            'info': 100         # Informaciones en 1 hora
        }
        
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.error_patterns = {
            'async_sync': ['await', 'async', 'sync', 'coroutine'],
            'tensorflow': ['tensorflow', 'keras', 'model', 'training'],
            'gemini': ['gemini', 'api', 'connection', 'response'],
            'celebro': ['celebro', 'analysis', 'response'],
            'database': ['sqlite', 'database', 'query', 'connection'],
            'memory': ['memory', 'out of memory', 'allocation'],
            'network': ['network', 'connection', 'timeout', 'refused']
        }
        
        self.performance_metrics = {
            'errors_detected': 0,
            'errors_resolved': 0,
            'alerts_sent': 0,
            'monitoring_uptime': 0,
            'average_response_time': 0
        }
    
    def start_monitoring(self):
        """Inicia el monitoreo de errores en tiempo real"""
        try:
            if self.monitoring_active:
                logger.warning("El monitoreo ya está activo")
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Monitor de errores iniciado")
            
        except Exception as e:
            logger.error(f"Error iniciando monitor de errores: {e}")
    
    def stop_monitoring(self):
        """Detiene el monitoreo de errores"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Monitor de errores detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo monitor de errores: {e}")
    
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                # Actualizar métricas de rendimiento
                self.performance_metrics['monitoring_uptime'] = time.time() - start_time
                
                # Verificar umbrales de alerta
                self._check_alert_thresholds()
                
                # Limpiar errores antiguos
                self._cleanup_old_errors()
                
                # Esperar antes de la siguiente verificación
                time.sleep(10)  # Verificar cada 10 segundos
                
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(30)  # Esperar más tiempo si hay error
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, 
                  severity: str = 'error', category: str = 'general') -> str:
        """
        Registra un error en el sistema de monitoreo.
        
        Args:
            error: Excepción o error
            context: Contexto adicional del error
            severity: Severidad (critical, error, warning, info)
            category: Categoría del error
            
        Returns:
            ID único del error registrado
        """
        try:
            error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.error_history)}"
            
            error_entry = {
                'id': error_id,
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'category': category,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {},
                'resolved': False,
                'resolution_time': None
            }
            
            # Agregar a historial
            self.error_history.append(error_entry)
            
            # Actualizar contadores
            self.error_counts[severity] += 1
            self.error_categories[category].append(error_id)
            
            # Detectar patrón de error
            detected_pattern = self._detect_error_pattern(error_entry)
            if detected_pattern:
                error_entry['detected_pattern'] = detected_pattern
            
            # Verificar si requiere alerta inmediata
            if severity == 'critical':
                self._send_immediate_alert(error_entry)
            
            self.performance_metrics['errors_detected'] += 1
            
            logger.error(f"Error registrado: {error_id} - {severity.upper()}: {str(error)}")
            
            return error_id
            
        except Exception as e:
            logger.error(f"Error registrando error: {e}")
            return f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _detect_error_pattern(self, error_entry: Dict[str, Any]) -> Optional[str]:
        """Detecta patrones de error basados en el contenido"""
        try:
            error_text = f"{error_entry['error_message']} {error_entry['traceback']}".lower()
            
            for pattern_name, keywords in self.error_patterns.items():
                if any(keyword in error_text for keyword in keywords):
                    return pattern_name
            
            return None
            
        except Exception as e:
            logger.error(f"Error detectando patrón: {e}")
            return None
    
    def _check_alert_thresholds(self):
        """Verifica si se han alcanzado umbrales de alerta"""
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            
            # Contar errores en la última hora
            recent_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > one_hour_ago
            ]
            
            severity_counts = defaultdict(int)
            for error in recent_errors:
                severity_counts[error['severity']] += 1
            
            # Verificar umbrales
            for severity, threshold in self.alert_thresholds.items():
                if severity_counts[severity] >= threshold:
                    self._send_threshold_alert(severity, severity_counts[severity], threshold)
            
        except Exception as e:
            logger.error(f"Error verificando umbrales: {e}")
    
    def _send_immediate_alert(self, error_entry: Dict[str, Any]):
        """Envía alerta inmediata para errores críticos"""
        try:
            alert = {
                'type': 'IMMEDIATE_ALERT',
                'timestamp': datetime.now().isoformat(),
                'error_id': error_entry['id'],
                'severity': error_entry['severity'],
                'message': f"Error crítico detectado: {error_entry['error_message']}",
                'context': error_entry['context']
            }
            
            # Aquí se podría integrar con sistemas de notificación
            logger.critical(f"ALERTA INMEDIATA: {alert['message']}")
            
            self.performance_metrics['alerts_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error enviando alerta inmediata: {e}")
    
    def _send_threshold_alert(self, severity: str, count: int, threshold: int):
        """Envía alerta por umbral alcanzado"""
        try:
            alert = {
                'type': 'THRESHOLD_ALERT',
                'timestamp': datetime.now().isoformat(),
                'severity': severity,
                'count': count,
                'threshold': threshold,
                'message': f"Umbral de {severity} alcanzado: {count}/{threshold}"
            }
            
            logger.warning(f"ALERTA DE UMBRAL: {alert['message']}")
            
            self.performance_metrics['alerts_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error enviando alerta de umbral: {e}")
    
    def _cleanup_old_errors(self):
        """Limpia errores antiguos del historial"""
        try:
            # Mantener solo errores de las últimas 24 horas
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # El deque ya maneja el tamaño máximo, pero podemos limpiar por tiempo
            while (self.error_history and 
                   datetime.fromisoformat(self.error_history[0]['timestamp']) < cutoff_time):
                self.error_history.popleft()
            
        except Exception as e:
            logger.error(f"Error limpiando errores antiguos: {e}")
    
    def resolve_error(self, error_id: str, resolution_notes: str = "") -> bool:
        """
        Marca un error como resuelto.
        
        Args:
            error_id: ID del error a resolver
            resolution_notes: Notas sobre la resolución
            
        Returns:
            True si se resolvió exitosamente
        """
        try:
            for error in self.error_history:
                if error['id'] == error_id:
                    error['resolved'] = True
                    error['resolution_time'] = datetime.now().isoformat()
                    error['resolution_notes'] = resolution_notes
                    
                    self.performance_metrics['errors_resolved'] += 1
                    logger.info(f"Error resuelto: {error_id}")
                    return True
            
            logger.warning(f"Error no encontrado para resolver: {error_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error resolviendo error: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de errores"""
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            one_day_ago = current_time - timedelta(days=1)
            
            # Errores recientes
            recent_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > one_hour_ago
            ]
            
            # Errores del día
            daily_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > one_day_ago
            ]
            
            # Estadísticas por severidad
            severity_stats = defaultdict(int)
            for error in daily_errors:
                severity_stats[error['severity']] += 1
            
            # Estadísticas por categoría
            category_stats = defaultdict(int)
            for error in daily_errors:
                category_stats[error['category']] += 1
            
            # Errores no resueltos
            unresolved_errors = [error for error in daily_errors if not error['resolved']]
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'daily_errors': len(daily_errors),
                'unresolved_errors': len(unresolved_errors),
                'severity_distribution': dict(severity_stats),
                'category_distribution': dict(category_stats),
                'resolution_rate': self._calculate_resolution_rate(),
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {'error': str(e)}
    
    def _calculate_resolution_rate(self) -> float:
        """Calcula la tasa de resolución de errores"""
        try:
            if not self.error_history:
                return 0.0
            
            resolved_count = sum(1 for error in self.error_history if error['resolved'])
            total_count = len(self.error_history)
            
            return (resolved_count / total_count) * 100 if total_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando tasa de resolución: {e}")
            return 0.0
    
    def get_error_report(self, hours: int = 24) -> Dict[str, Any]:
        """Genera reporte de errores para un período específico"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            period_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > cutoff_time
            ]
            
            # Agrupar por patrón detectado
            pattern_groups = defaultdict(list)
            for error in period_errors:
                pattern = error.get('detected_pattern', 'unknown')
                pattern_groups[pattern].append(error)
            
            # Errores más frecuentes
            error_frequency = defaultdict(int)
            for error in period_errors:
                error_key = f"{error['error_type']}: {error['error_message'][:50]}"
                error_frequency[error_key] += 1
            
            most_frequent = sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'period_hours': hours,
                'total_errors': len(period_errors),
                'pattern_analysis': dict(pattern_groups),
                'most_frequent_errors': most_frequent,
                'severity_breakdown': self._get_severity_breakdown(period_errors),
                'recommendations': self._generate_error_recommendations(period_errors)
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            return {'error': str(e)}
    
    def _get_severity_breakdown(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Obtiene desglose por severidad"""
        breakdown = defaultdict(int)
        for error in errors:
            breakdown[error['severity']] += 1
        return dict(breakdown)
    
    def _generate_error_recommendations(self, errors: List[Dict[str, Any]]) -> List[str]:
        """Genera recomendaciones basadas en los errores"""
        recommendations = []
        
        # Analizar patrones de error
        pattern_counts = defaultdict(int)
        for error in errors:
            pattern = error.get('detected_pattern', 'unknown')
            pattern_counts[pattern] += 1
        
        # Recomendaciones basadas en patrones
        if pattern_counts['async_sync'] > 5:
            recommendations.append("Revisar sincronización asíncrona - muchos errores de await/async")
        
        if pattern_counts['tensorflow'] > 3:
            recommendations.append("Optimizar operaciones de TensorFlow - errores frecuentes detectados")
        
        if pattern_counts['gemini'] > 2:
            recommendations.append("Verificar conexión y configuración de Gemini API")
        
        if pattern_counts['memory'] > 1:
            recommendations.append("Revisar uso de memoria - posibles memory leaks")
        
        # Recomendaciones generales
        if len(errors) > 50:
            recommendations.append("Alto volumen de errores - revisar configuración general")
        
        unresolved_count = sum(1 for error in errors if not error['resolved'])
        if unresolved_count > len(errors) * 0.5:
            recommendations.append("Muchos errores sin resolver - priorizar resolución")
        
        return recommendations
    
    def export_error_log(self, filepath: str) -> bool:
        """Exporta el log de errores a un archivo"""
        try:
            error_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_errors': len(self.error_history),
                'error_history': list(self.error_history),
                'statistics': self.get_error_statistics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Log de errores exportado a: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando log: {e}")
            return False
