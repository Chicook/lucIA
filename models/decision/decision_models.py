#!/usr/bin/env python3
"""
Sistema de Modelos de Toma de Decisiones - LucIA
Versión: 0.6.0
Sistema para gestión de modelos de toma de decisiones y árboles de decisión
"""

import os
import json
import pickle
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger('Decision_Models')

class DecisionType(Enum):
    """Tipos de modelos de decisión"""
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    RULE_BASED = "rule_based"
    FUZZY_LOGIC = "fuzzy_logic"
    BAYESIAN = "bayesian"
    NEURAL_DECISION = "neural_decision"

class DecisionStatus(Enum):
    """Estados de los modelos de decisión"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    ERROR = "error"

@dataclass
class DecisionRule:
    """Regla de decisión"""
    rule_id: str
    condition: str
    action: str
    confidence: float
    priority: int
    metadata: Dict[str, Any]

@dataclass
class DecisionModelMetadata:
    """Metadatos de un modelo de decisión"""
    model_id: str
    name: str
    decision_type: DecisionType
    status: DecisionStatus
    created_at: datetime
    last_updated: datetime
    version: str
    description: str
    input_features: List[str]
    output_classes: List[str]
    rules: List[DecisionRule]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    file_path: str
    file_size: int
    checksum: str

class DecisionModelManager:
    """
    Gestor de modelos de toma de decisiones
    """
    
    def __init__(self, models_dir: str = "models/decision"):
        self.models_dir = models_dir
        self.models: Dict[str, DecisionModelMetadata] = {}
        self.metadata_file = os.path.join(models_dir, "decision_models_metadata.json")
        
        # Crear directorio si no existe
        os.makedirs(models_dir, exist_ok=True)
        
        # Cargar metadatos existentes
        self._load_metadata()
        
        logger.info("Sistema de modelos de decisión inicializado")
    
    def save_decision_model(self, model: Any, name: str, decision_type: DecisionType,
                           description: str = "", version: str = "1.0.0",
                           input_features: List[str] = None,
                           output_classes: List[str] = None,
                           rules: List[DecisionRule] = None,
                           parameters: Dict[str, Any] = None,
                           performance_metrics: Dict[str, float] = None) -> str:
        """
        Guarda un modelo de decisión
        
        Args:
            model: Modelo a guardar
            name: Nombre del modelo
            decision_type: Tipo de modelo de decisión
            description: Descripción del modelo
            version: Versión del modelo
            input_features: Características de entrada
            output_classes: Clases de salida
            rules: Reglas de decisión
            parameters: Parámetros del modelo
            performance_metrics: Métricas de rendimiento
            
        Returns:
            ID del modelo guardado
        """
        try:
            # Generar ID único
            model_id = self._generate_model_id(name, version)
            
            # Crear nombre de archivo
            filename = f"{model_id}.pkl"
            file_path = os.path.join(self.models_dir, filename)
            
            # Guardar modelo
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Obtener información del archivo
            file_size = os.path.getsize(file_path)
            checksum = self._calculate_checksum(file_path)
            
            # Crear metadatos
            metadata = DecisionModelMetadata(
                model_id=model_id,
                name=name,
                decision_type=decision_type,
                status=DecisionStatus.TRAINED,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                version=version,
                description=description,
                input_features=input_features or [],
                output_classes=output_classes or [],
                rules=rules or [],
                parameters=parameters or {},
                performance_metrics=performance_metrics or {},
                file_path=file_path,
                file_size=file_size,
                checksum=checksum
            )
            
            # Registrar modelo
            self.models[model_id] = metadata
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Modelo de decisión guardado: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error guardando modelo de decisión: {e}")
            raise
    
    def load_decision_model(self, model_id: str) -> Optional[Any]:
        """
        Carga un modelo de decisión
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Modelo cargado o None si no existe
        """
        try:
            if model_id not in self.models:
                logger.warning(f"Modelo de decisión no encontrado: {model_id}")
                return None
            
            metadata = self.models[model_id]
            
            # Verificar que el archivo existe
            if not os.path.exists(metadata.file_path):
                logger.error(f"Archivo del modelo no encontrado: {metadata.file_path}")
                return None
            
            # Verificar checksum
            current_checksum = self._calculate_checksum(metadata.file_path)
            if current_checksum != metadata.checksum:
                logger.warning(f"Checksum del modelo {model_id} no coincide")
            
            # Cargar modelo
            with open(metadata.file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Actualizar último acceso
            metadata.last_updated = datetime.now()
            self._save_metadata()
            
            logger.info(f"Modelo de decisión cargado: {metadata.name} ({model_id})")
            return model
            
        except Exception as e:
            logger.error(f"Error cargando modelo de decisión: {e}")
            return None
    
    def add_decision_rule(self, model_id: str, condition: str, action: str,
                         confidence: float = 1.0, priority: int = 0,
                         metadata: Dict[str, Any] = None) -> str:
        """
        Añade una regla de decisión a un modelo
        
        Args:
            model_id: ID del modelo
            condition: Condición de la regla
            action: Acción a tomar
            confidence: Confianza de la regla
            priority: Prioridad de la regla
            metadata: Metadatos adicionales
            
        Returns:
            ID de la regla creada
        """
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo no encontrado: {model_id}")
            
            # Generar ID de regla
            rule_id = self._generate_rule_id(model_id, condition)
            
            # Crear regla
            rule = DecisionRule(
                rule_id=rule_id,
                condition=condition,
                action=action,
                confidence=confidence,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Añadir regla al modelo
            self.models[model_id].rules.append(rule)
            self.models[model_id].last_updated = datetime.now()
            
            # Guardar metadatos
            self._save_metadata()
            
            logger.info(f"Regla añadida al modelo {model_id}: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error añadiendo regla de decisión: {e}")
            raise
    
    def evaluate_decision(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evalúa una decisión usando un modelo
        
        Args:
            model_id: ID del modelo
            input_data: Datos de entrada
            
        Returns:
            Resultado de la decisión
        """
        try:
            if model_id not in self.models:
                return {'error': 'Modelo no encontrado'}
            
            metadata = self.models[model_id]
            
            # Cargar modelo
            model = self.load_decision_model(model_id)
            if model is None:
                return {'error': 'No se pudo cargar el modelo'}
            
            # Evaluar reglas
            applicable_rules = []
            for rule in metadata.rules:
                if self._evaluate_condition(rule.condition, input_data):
                    applicable_rules.append(rule)
            
            # Ordenar por prioridad y confianza
            applicable_rules.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
            
            if not applicable_rules:
                return {
                    'decision': 'no_action',
                    'confidence': 0.0,
                    'reasoning': 'No hay reglas aplicables',
                    'applicable_rules': []
                }
            
            # Seleccionar mejor regla
            best_rule = applicable_rules[0]
            
            return {
                'decision': best_rule.action,
                'confidence': best_rule.confidence,
                'reasoning': f"Regla aplicada: {best_rule.condition}",
                'applicable_rules': [
                    {
                        'rule_id': rule.rule_id,
                        'condition': rule.condition,
                        'action': rule.action,
                        'confidence': rule.confidence,
                        'priority': rule.priority
                    }
                    for rule in applicable_rules
                ]
            }
            
        except Exception as e:
            logger.error(f"Error evaluando decisión: {e}")
            return {'error': str(e)}
    
    def _evaluate_condition(self, condition: str, input_data: Dict[str, Any]) -> bool:
        """
        Evalúa una condición de regla
        
        Args:
            condition: Condición a evaluar
            input_data: Datos de entrada
            
        Returns:
            True si la condición se cumple
        """
        try:
            # Reemplazar variables en la condición
            evaluated_condition = condition
            for key, value in input_data.items():
                if isinstance(value, (int, float)):
                    evaluated_condition = evaluated_condition.replace(f"{{{key}}}", str(value))
                elif isinstance(value, str):
                    evaluated_condition = evaluated_condition.replace(f"{{{key}}}", f"'{value}'")
            
            # Evaluar la condición (simplificado)
            # En un sistema real, esto sería más sofisticado
            return eval(evaluated_condition)
            
        except Exception as e:
            logger.error(f"Error evaluando condición: {e}")
            return False
    
    def get_model_metadata(self, model_id: str) -> Optional[DecisionModelMetadata]:
        """Obtiene metadatos de un modelo de decisión"""
        return self.models.get(model_id)
    
    def list_models(self, decision_type: Optional[DecisionType] = None,
                   status: Optional[DecisionStatus] = None) -> List[DecisionModelMetadata]:
        """
        Lista modelos con filtros opcionales
        
        Args:
            decision_type: Filtrar por tipo de decisión
            status: Filtrar por estado
            
        Returns:
            Lista de metadatos de modelos
        """
        filtered_models = []
        
        for metadata in self.models.values():
            if decision_type and metadata.decision_type != decision_type:
                continue
            if status and metadata.status != status:
                continue
            filtered_models.append(metadata)
        
        # Ordenar por fecha de creación (más recientes primero)
        filtered_models.sort(key=lambda x: x.created_at, reverse=True)
        
        return filtered_models
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los modelos de decisión"""
        try:
            total_models = len(self.models)
            
            # Estadísticas por tipo
            type_counts = {}
            for decision_type in DecisionType:
                type_counts[decision_type.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.decision_type == decision_type
                )
            
            # Estadísticas por estado
            status_counts = {}
            for status in DecisionStatus:
                status_counts[status.value] = sum(
                    1 for metadata in self.models.values()
                    if metadata.status == status
                )
            
            # Tamaño total
            total_size = sum(metadata.file_size for metadata in self.models.values())
            
            # Total de reglas
            total_rules = sum(len(metadata.rules) for metadata in self.models.values())
            
            return {
                'total_models': total_models,
                'total_size': total_size,
                'total_rules': total_rules,
                'type_distribution': type_counts,
                'status_distribution': status_counts,
                'average_rules_per_model': total_rules / total_models if total_models > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Genera un ID único para el modelo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{version}_{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def _generate_rule_id(self, model_id: str, condition: str) -> str:
        """Genera un ID único para una regla"""
        hash_input = f"{model_id}_{condition}_{datetime.now()}".encode()
        return hashlib.md5(hash_input).hexdigest()[:8]
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcula el checksum de un archivo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _save_metadata(self):
        """Guarda metadatos de modelos de decisión"""
        try:
            metadata = {
                'models': {
                    model_id: {
                        'model_id': metadata.model_id,
                        'name': metadata.name,
                        'decision_type': metadata.decision_type.value,
                        'status': metadata.status.value,
                        'created_at': metadata.created_at.isoformat(),
                        'last_updated': metadata.last_updated.isoformat(),
                        'version': metadata.version,
                        'description': metadata.description,
                        'input_features': metadata.input_features,
                        'output_classes': metadata.output_classes,
                        'rules': [
                            {
                                'rule_id': rule.rule_id,
                                'condition': rule.condition,
                                'action': rule.action,
                                'confidence': rule.confidence,
                                'priority': rule.priority,
                                'metadata': rule.metadata
                            }
                            for rule in metadata.rules
                        ],
                        'parameters': metadata.parameters,
                        'performance_metrics': metadata.performance_metrics,
                        'file_path': metadata.file_path,
                        'file_size': metadata.file_size,
                        'checksum': metadata.checksum
                    }
                    for model_id, metadata in self.models.items()
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _load_metadata(self):
        """Carga metadatos de modelos de decisión"""
        try:
            if not os.path.exists(self.metadata_file):
                return
            
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            # Cargar modelos
            for model_id, model_data in data.get('models', {}).items():
                # Cargar reglas
                rules = []
                for rule_data in model_data.get('rules', []):
                    rule = DecisionRule(
                        rule_id=rule_data['rule_id'],
                        condition=rule_data['condition'],
                        action=rule_data['action'],
                        confidence=rule_data['confidence'],
                        priority=rule_data['priority'],
                        metadata=rule_data['metadata']
                    )
                    rules.append(rule)
                
                metadata = DecisionModelMetadata(
                    model_id=model_data['model_id'],
                    name=model_data['name'],
                    decision_type=DecisionType(model_data['decision_type']),
                    status=DecisionStatus(model_data['status']),
                    created_at=datetime.fromisoformat(model_data['created_at']),
                    last_updated=datetime.fromisoformat(model_data['last_updated']),
                    version=model_data['version'],
                    description=model_data['description'],
                    input_features=model_data['input_features'],
                    output_classes=model_data['output_classes'],
                    rules=rules,
                    parameters=model_data['parameters'],
                    performance_metrics=model_data['performance_metrics'],
                    file_path=model_data['file_path'],
                    file_size=model_data['file_size'],
                    checksum=model_data['checksum']
                )
                
                # Solo cargar si el archivo existe
                if os.path.exists(metadata.file_path):
                    self.models[model_id] = metadata
            
            logger.info(f"Cargados {len(self.models)} modelos de decisión")
            
        except Exception as e:
            logger.error(f"Error cargando metadatos: {e}")

# Instancia global del gestor de modelos de decisión
decision_model_manager = DecisionModelManager()
