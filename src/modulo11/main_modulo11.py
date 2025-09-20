"""
Módulo 11: Sistema de Monitoreo
Versión: 0.6.0
Funcionalidad: Monitoreo de rendimiento, alertas y métricas del sistema
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger('LucIA_Monitoring')

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Métrica del sistema"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False

class MonitoringSystem:
    """
    Sistema de monitoreo para LucIA.
    Gestiona métricas, alertas y rendimiento del sistema.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.metrics = {}
        self.alerts = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Configuración de alertas
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5.0,
            "error_rate": 0.1
        }
        
        # Estadísticas
        self.total_metrics_collected = 0
        self.total_alerts_generated = 0
        self.active_alerts = 0
        
        logger.info("Sistema de monitoreo inicializado")
    
    async def start_monitoring(self, interval: int = 30):
        """Inicia el monitoreo del sistema"""
        try:
            if self.monitoring_active:
                logger.warning("El monitoreo ya está activo")
                return
            
            self.monitoring_active = True
            
            # Iniciar hilo de monitoreo
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            
            logger.info(f"Monitoreo iniciado con intervalo de {interval} segundos")
            
        except Exception as e:
            logger.error(f"Error iniciando monitoreo: {e}")
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Detiene el monitoreo del sistema"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Monitoreo detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo monitoreo: {e}")
    
    def _monitoring_loop(self, interval: int):
        """Loop principal de monitoreo"""
        while self.monitoring_active:
            try:
                # Recopilar métricas
                asyncio.run(self._collect_system_metrics())
                
                # Verificar alertas
                asyncio.run(self._check_alerts())
                
                # Limpiar métricas antiguas
                asyncio.run(self._cleanup_old_metrics())
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Recopila métricas del sistema"""
        try:
            current_time = datetime.now()
            
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("cpu_usage", cpu_percent, "%", current_time)
            
            # Métricas de memoria
            memory = psutil.virtual_memory()
            await self._record_metric("memory_usage", memory.percent, "%", current_time)
            await self._record_metric("memory_available", memory.available / (1024**3), "GB", current_time)
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._record_metric("disk_usage", disk_percent, "%", current_time)
            await self._record_metric("disk_free", disk.free / (1024**3), "GB", current_time)
            
            # Métricas de red
            network = psutil.net_io_counters()
            await self._record_metric("network_bytes_sent", network.bytes_sent, "bytes", current_time)
            await self._record_metric("network_bytes_recv", network.bytes_recv, "bytes", current_time)
            
            # Métricas de procesos
            process_count = len(psutil.pids())
            await self._record_metric("process_count", process_count, "count", current_time)
            
            # Métricas específicas de LucIA
            await self._collect_lucia_metrics(current_time)
            
            self.total_metrics_collected += 1
            
        except Exception as e:
            logger.error(f"Error recopilando métricas: {e}")
    
    async def _collect_lucia_metrics(self, timestamp: datetime):
        """Recopila métricas específicas de LucIA"""
        try:
            if not self.core_engine:
                return
            
            # Métricas de módulos
            if hasattr(self.core_engine, 'modules'):
                active_modules = len([m for m in self.core_engine.modules.values() if m])
                await self._record_metric("active_modules", active_modules, "count", timestamp)
            
            # Métricas de memoria del sistema
            if hasattr(self.core_engine, 'memory_system'):
                memory_stats = await self.core_engine.memory_system.get_memory_stats()
                await self._record_metric("lucia_memories", memory_stats.get("total_memories", 0), "count", timestamp)
                await self._record_metric("lucia_memory_usage", memory_stats.get("memory_usage_bytes", 0), "bytes", timestamp)
            
            # Métricas de aprendizaje
            if hasattr(self.core_engine, 'learning_engine'):
                learning_stats = await self.core_engine.learning_engine.get_learning_stats()
                await self._record_metric("learning_cycles", learning_stats.get("learning_cycles", 0), "count", timestamp)
                await self._record_metric("total_models", learning_stats.get("total_models", 0), "count", timestamp)
            
            # Métricas de comunicación
            if hasattr(self.core_engine, 'communication_hub'):
                comm_stats = await self.core_engine.communication_hub.get_communication_stats()
                await self._record_metric("connected_ais", comm_stats.get("connected_ais", 0), "count", timestamp)
                await self._record_metric("messages_sent", comm_stats.get("messages_sent", 0), "count", timestamp)
            
        except Exception as e:
            logger.error(f"Error recopilando métricas de LucIA: {e}")
    
    async def _record_metric(self, name: str, value: float, unit: str, timestamp: datetime):
        """Registra una métrica"""
        try:
            metric_id = f"{name}_{int(timestamp.timestamp())}"
            metric = Metric(
                name=name,
                value=value,
                unit=unit,
                timestamp=timestamp,
                tags={"source": "monitoring_system"}
            )
            
            self.metrics[metric_id] = metric
            
        except Exception as e:
            logger.error(f"Error registrando métrica {name}: {e}")
    
    async def _check_alerts(self):
        """Verifica si se deben generar alertas"""
        try:
            current_time = datetime.now()
            
            # Verificar métricas recientes (últimos 5 minutos)
            recent_metrics = self._get_recent_metrics(minutes=5)
            
            for metric_name, threshold in self.alert_thresholds.items():
                metric_values = [m.value for m in recent_metrics if m.name == metric_name]
                
                if metric_values:
                    current_value = metric_values[-1]  # Valor más reciente
                    
                    if current_value >= threshold:
                        await self._generate_alert(
                            metric_name, threshold, current_value, current_time
                        )
            
        except Exception as e:
            logger.error(f"Error verificando alertas: {e}")
    
    def _get_recent_metrics(self, minutes: int = 5) -> List[Metric]:
        """Obtiene métricas recientes"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            return [
                metric for metric in self.metrics.values()
                if metric.timestamp >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error obteniendo métricas recientes: {e}")
            return []
    
    async def _generate_alert(self, metric_name: str, threshold: float, 
                            current_value: float, timestamp: datetime):
        """Genera una alerta"""
        try:
            # Verificar si ya existe una alerta activa para esta métrica
            existing_alert = None
            for alert in self.alerts.values():
                if (alert.metric_name == metric_name and 
                    not alert.resolved and 
                    alert.level in [AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]):
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Actualizar alerta existente
                existing_alert.current_value = current_value
                existing_alert.timestamp = timestamp
                return
            
            # Determinar nivel de alerta
            if current_value >= threshold * 1.5:
                level = AlertLevel.CRITICAL
            elif current_value >= threshold * 1.2:
                level = AlertLevel.ERROR
            elif current_value >= threshold:
                level = AlertLevel.WARNING
            else:
                return
            
            # Crear nueva alerta
            alert_id = f"alert_{int(timestamp.timestamp())}"
            alert = Alert(
                id=alert_id,
                level=level,
                message=f"{metric_name} excedió el umbral: {current_value:.2f} >= {threshold:.2f}",
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value,
                timestamp=timestamp
            )
            
            self.alerts[alert_id] = alert
            self.total_alerts_generated += 1
            self.active_alerts += 1
            
            logger.warning(f"Alerta generada: {alert.message}")
            
        except Exception as e:
            logger.error(f"Error generando alerta: {e}")
    
    async def _cleanup_old_metrics(self, hours: int = 24):
        """Limpia métricas antiguas"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            old_metrics = [
                metric_id for metric_id, metric in self.metrics.items()
                if metric.timestamp < cutoff_time
            ]
            
            for metric_id in old_metrics:
                del self.metrics[metric_id]
            
            if old_metrics:
                logger.debug(f"Limpiadas {len(old_metrics)} métricas antiguas")
                
        except Exception as e:
            logger.error(f"Error limpiando métricas antiguas: {e}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del sistema"""
        try:
            recent_metrics = self._get_recent_metrics(minutes=10)
            
            # Calcular promedios
            cpu_avg = self._calculate_average(recent_metrics, "cpu_usage")
            memory_avg = self._calculate_average(recent_metrics, "memory_usage")
            disk_avg = self._calculate_average(recent_metrics, "disk_usage")
            
            # Determinar estado general
            health_score = 100
            if cpu_avg > 80:
                health_score -= 20
            if memory_avg > 85:
                health_score -= 20
            if disk_avg > 90:
                health_score -= 20
            
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 60:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "status": status,
                "health_score": health_score,
                "cpu_usage": cpu_avg,
                "memory_usage": memory_avg,
                "disk_usage": disk_avg,
                "active_alerts": self.active_alerts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de salud: {e}")
            return {"status": "unknown", "error": str(e)}
    
    def _calculate_average(self, metrics: List[Metric], name: str) -> float:
        """Calcula el promedio de una métrica"""
        try:
            values = [m.value for m in metrics if m.name == name]
            return sum(values) / len(values) if values else 0.0
        except:
            return 0.0
    
    async def get_metrics(self, metric_name: str = None, 
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Obtiene métricas del sistema"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_metrics = [
                metric for metric in self.metrics.values()
                if metric.timestamp >= cutoff_time and
                (metric_name is None or metric.name == metric_name)
            ]
            
            # Ordenar por timestamp
            filtered_metrics.sort(key=lambda x: x.timestamp)
            
            return [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags
                }
                for metric in filtered_metrics
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return []
    
    async def get_alerts(self, level: AlertLevel = None, 
                        resolved: bool = None) -> List[Dict[str, Any]]:
        """Obtiene alertas del sistema"""
        try:
            filtered_alerts = list(self.alerts.values())
            
            if level:
                filtered_alerts = [a for a in filtered_alerts if a.level == level]
            
            if resolved is not None:
                filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
            
            # Ordenar por timestamp descendente
            filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in filtered_alerts
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo alertas: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resuelve una alerta"""
        try:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.active_alerts = max(0, self.active_alerts - 1)
                logger.info(f"Alerta resuelta: {alert_id}")
                return True
            else:
                logger.warning(f"Alerta no encontrada: {alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error resolviendo alerta: {e}")
            return False
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de monitoreo"""
        return {
            "monitoring_active": self.monitoring_active,
            "total_metrics_collected": self.total_metrics_collected,
            "total_alerts_generated": self.total_alerts_generated,
            "active_alerts": self.active_alerts,
            "total_metrics": len(self.metrics),
            "total_alerts": len(self.alerts),
            "alert_thresholds": self.alert_thresholds
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de monitoreo"""
        try:
            state = {
                "metrics": {
                    metric_id: {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "tags": metric.tags
                    }
                    for metric_id, metric in self.metrics.items()
                },
                "alerts": {
                    alert_id: {
                        "id": alert.id,
                        "level": alert.level.value,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "threshold": alert.threshold,
                        "current_value": alert.current_value,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved
                    }
                    for alert_id, alert in self.alerts.items()
                },
                "stats": {
                    "total_metrics_collected": self.total_metrics_collected,
                    "total_alerts_generated": self.total_alerts_generated,
                    "active_alerts": self.active_alerts
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/monitoring_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de monitoreo guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de monitoreo: {e}")

# Instancia global del sistema de monitoreo
monitoring_system = MonitoringSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de monitoreo"""
    global monitoring_system
    monitoring_system.core_engine = core_engine
    core_engine.monitoring_system = monitoring_system
    
    # Iniciar monitoreo automáticamente
    await monitoring_system.start_monitoring()
    
    logger.info("Módulo de monitoreo inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de monitoreo"""
    if isinstance(input_data, dict) and "get_health" in input_data:
        # Obtener estado de salud
        health = await monitoring_system.get_system_health()
        return health
    
    elif isinstance(input_data, dict) and "get_metrics" in input_data:
        # Obtener métricas
        metric_data = input_data["get_metrics"]
        metrics = await monitoring_system.get_metrics(
            metric_name=metric_data.get("name"),
            hours=metric_data.get("hours", 24)
        )
        return {"metrics": metrics}
    
    elif isinstance(input_data, dict) and "get_alerts" in input_data:
        # Obtener alertas
        alert_data = input_data["get_alerts"]
        alerts = await monitoring_system.get_alerts(
            level=AlertLevel(alert_data.get("level")) if alert_data.get("level") else None,
            resolved=alert_data.get("resolved")
        )
        return {"alerts": alerts}
    
    return input_data

def run_modulo11():
    """Función de compatibilidad con el sistema anterior"""
    print("📊 Módulo 11: Sistema de Monitoreo")
    print("   - Monitoreo de rendimiento")
    print("   - Alertas automáticas")
    print("   - Métricas del sistema")
    print("   - Estado de salud")
    print("   ✅ Módulo inicializado correctamente")