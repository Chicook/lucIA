"""
Módulo 6: Sistema de Percepción
Versión: 0.6.0
Funcionalidad: Procesamiento de entrada, análisis de datos y extracción de características
"""

import asyncio
import json
import logging
import re
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import io

logger = logging.getLogger('LucIA_Perception')

class InputType(Enum):
    """Tipos de entrada soportados"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED_DATA = "structured_data"
    MULTIMODAL = "multimodal"

class ProcessingStage(Enum):
    """Etapas de procesamiento"""
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    FEATURES_EXTRACTED = "features_extracted"
    ANALYZED = "analyzed"

@dataclass
class PerceptionResult:
    """Resultado del procesamiento de percepción"""
    input_id: str
    input_type: InputType
    stage: ProcessingStage
    raw_data: Any
    processed_data: Any
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime

class PerceptionSystem:
    """
    Sistema de percepción para LucIA.
    Procesa diferentes tipos de entrada y extrae características relevantes.
    """
    
    def __init__(self, core_engine=None):
        self.core_engine = core_engine
        self.input_processors = {}
        self.feature_extractors = {}
        self.preprocessing_pipelines = {}
        self.perception_results = {}
        
        # Configuración
        self.max_input_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = {
            "text": ["txt", "md", "json", "csv"],
            "image": ["jpg", "jpeg", "png", "gif", "bmp"],
            "audio": ["wav", "mp3", "ogg", "flac"]
        }
        
        # Estadísticas
        self.total_inputs_processed = 0
        self.successful_processing = 0
        self.failed_processing = 0
        
        # Inicializar procesadores
        self._initialize_processors()
        
        logger.info("Sistema de percepción inicializado")
    
    def _initialize_processors(self):
        """Inicializa los procesadores de entrada"""
        # Procesador de texto
        self.input_processors[InputType.TEXT] = self._process_text_input
        self.feature_extractors[InputType.TEXT] = self._extract_text_features
        
        # Procesador de imagen
        self.input_processors[InputType.IMAGE] = self._process_image_input
        self.feature_extractors[InputType.IMAGE] = self._extract_image_features
        
        # Procesador de audio
        self.input_processors[InputType.AUDIO] = self._process_audio_input
        self.feature_extractors[InputType.AUDIO] = self._extract_audio_features
        
        # Procesador de datos estructurados
        self.input_processors[InputType.STRUCTURED_DATA] = self._process_structured_input
        self.feature_extractors[InputType.STRUCTURED_DATA] = self._extract_structured_features
        
        # Pipeline de preprocesamiento
        self.preprocessing_pipelines = {
            InputType.TEXT: self._preprocess_text,
            InputType.IMAGE: self._preprocess_image,
            InputType.AUDIO: self._preprocess_audio,
            InputType.STRUCTURED_DATA: self._preprocess_structured
        }
    
    async def process_input(self, input_data: Any, input_type: InputType, 
                          context: Dict[str, Any] = None) -> PerceptionResult:
        """
        Procesa entrada de cualquier tipo
        
        Args:
            input_data: Datos de entrada
            input_type: Tipo de entrada
            context: Contexto adicional
        
        Returns:
            Resultado del procesamiento
        """
        try:
            self.total_inputs_processed += 1
            input_id = f"input_{int(datetime.now().timestamp())}"
            
            # Crear resultado inicial
            result = PerceptionResult(
                input_id=input_id,
                input_type=input_type,
                stage=ProcessingStage.RAW,
                raw_data=input_data,
                processed_data=None,
                features={},
                metadata={},
                confidence=0.0,
                timestamp=datetime.now()
            )
            
            # Validar entrada
            if not self._validate_input(input_data, input_type):
                result.metadata["error"] = "Entrada no válida"
                self.failed_processing += 1
                return result
            
            # Procesar entrada
            if input_type in self.input_processors:
                processed_data = await self.input_processors[input_type](input_data, context or {})
                result.processed_data = processed_data
                result.stage = ProcessingStage.PREPROCESSED
                
                # Extraer características
                if input_type in self.feature_extractors:
                    features = await self.feature_extractors[input_type](processed_data, context or {})
                    result.features = features
                    result.stage = ProcessingStage.FEATURES_EXTRACTED
                    
                    # Calcular confianza
                    result.confidence = self._calculate_confidence(features, input_type)
                    result.stage = ProcessingStage.ANALYZED
                
                self.successful_processing += 1
            else:
                result.metadata["error"] = f"Tipo de entrada no soportado: {input_type}"
                self.failed_processing += 1
            
            # Guardar resultado
            self.perception_results[input_id] = result
            
            logger.info(f"Entrada procesada: {input_id} ({input_type.value})")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando entrada: {e}")
            self.failed_processing += 1
            
            return PerceptionResult(
                input_id="error",
                input_type=input_type,
                stage=ProcessingStage.RAW,
                raw_data=input_data,
                processed_data=None,
                features={},
                metadata={"error": str(e)},
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _validate_input(self, input_data: Any, input_type: InputType) -> bool:
        """Valida la entrada según el tipo"""
        try:
            if input_type == InputType.TEXT:
                return isinstance(input_data, str) and len(input_data) > 0
            elif input_type == InputType.IMAGE:
                return isinstance(input_data, (str, bytes)) and len(input_data) > 0
            elif input_type == InputType.AUDIO:
                return isinstance(input_data, (str, bytes)) and len(input_data) > 0
            elif input_type == InputType.STRUCTURED_DATA:
                return isinstance(input_data, (dict, list)) and len(str(input_data)) > 0
            else:
                return False
        except:
            return False
    
    async def _process_text_input(self, input_data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada de texto"""
        try:
            # Preprocesar texto
            preprocessed = await self._preprocess_text(input_data, context)
            
            # Análisis básico
            analysis = {
                "length": len(preprocessed),
                "word_count": len(preprocessed.split()),
                "sentence_count": len(re.split(r'[.!?]+', preprocessed)),
                "language": self._detect_language(preprocessed),
                "sentiment": self._analyze_sentiment(preprocessed),
                "entities": self._extract_entities(preprocessed),
                "keywords": self._extract_keywords(preprocessed)
            }
            
            return {
                "original": input_data,
                "preprocessed": preprocessed,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error procesando texto: {e}")
            return {"error": str(e)}
    
    async def _process_image_input(self, input_data: Union[str, bytes], context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada de imagen"""
        try:
            # Decodificar imagen
            if isinstance(input_data, str):
                # Asumir que es base64
                image_data = base64.b64decode(input_data)
            else:
                image_data = input_data
            
            # Cargar imagen
            image = Image.open(io.BytesIO(image_data))
            
            # Análisis básico
            analysis = {
                "size": image.size,
                "mode": image.mode,
                "format": image.format,
                "has_transparency": "transparency" in image.info,
                "color_count": len(image.getcolors(maxcolors=256*256*256)) if image.mode == "RGB" else 0
            }
            
            return {
                "image_data": image_data,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return {"error": str(e)}
    
    async def _process_audio_input(self, input_data: Union[str, bytes], context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada de audio"""
        try:
            # Decodificar audio
            if isinstance(input_data, str):
                audio_data = base64.b64decode(input_data)
            else:
                audio_data = input_data
            
            # Análisis básico (simulado)
            analysis = {
                "size_bytes": len(audio_data),
                "estimated_duration": len(audio_data) / 16000,  # Estimación simple
                "format": "unknown",
                "sample_rate": 16000  # Asumido
            }
            
            return {
                "audio_data": audio_data,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error procesando audio: {e}")
            return {"error": str(e)}
    
    async def _process_structured_input(self, input_data: Union[Dict, List], context: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa entrada de datos estructurados"""
        try:
            # Análisis de estructura
            analysis = {
                "type": type(input_data).__name__,
                "size": len(str(input_data)),
                "keys": list(input_data.keys()) if isinstance(input_data, dict) else None,
                "depth": self._calculate_depth(input_data),
                "data_types": self._analyze_data_types(input_data)
            }
            
            return {
                "data": input_data,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error procesando datos estructurados: {e}")
            return {"error": str(e)}
    
    async def _preprocess_text(self, text: str, context: Dict[str, Any]) -> str:
        """Preprocesa texto"""
        try:
            # Limpiar texto
            cleaned = re.sub(r'\s+', ' ', text.strip())
            
            # Normalizar
            normalized = cleaned.lower()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error preprocesando texto: {e}")
            return text
    
    async def _preprocess_image(self, image_data: bytes, context: Dict[str, Any]) -> bytes:
        """Preprocesa imagen"""
        try:
            # Redimensionar si es muy grande
            image = Image.open(io.BytesIO(image_data))
            max_size = (1024, 1024)
            
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Guardar procesada
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            return image_data
    
    async def _preprocess_audio(self, audio_data: bytes, context: Dict[str, Any]) -> bytes:
        """Preprocesa audio"""
        # Por ahora, retornar sin cambios
        return audio_data
    
    async def _preprocess_structured(self, data: Union[Dict, List], context: Dict[str, Any]) -> Union[Dict, List]:
        """Preprocesa datos estructurados"""
        # Por ahora, retornar sin cambios
        return data
    
    async def _extract_text_features(self, processed_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características de texto"""
        try:
            text = processed_data.get("preprocessed", "")
            analysis = processed_data.get("analysis", {})
            
            features = {
                "text_length": analysis.get("length", 0),
                "word_count": analysis.get("word_count", 0),
                "sentence_count": analysis.get("sentence_count", 0),
                "language": analysis.get("language", "unknown"),
                "sentiment_score": analysis.get("sentiment", 0.0),
                "entity_count": len(analysis.get("entities", [])),
                "keyword_count": len(analysis.get("keywords", [])),
                "complexity_score": self._calculate_text_complexity(text),
                "readability_score": self._calculate_readability(text)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de texto: {e}")
            return {}
    
    async def _extract_image_features(self, processed_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características de imagen"""
        try:
            analysis = processed_data.get("analysis", {})
            
            features = {
                "width": analysis.get("size", (0, 0))[0],
                "height": analysis.get("size", (0, 0))[1],
                "aspect_ratio": self._calculate_aspect_ratio(analysis.get("size", (0, 0))),
                "mode": analysis.get("mode", "unknown"),
                "has_transparency": analysis.get("has_transparency", False),
                "color_count": analysis.get("color_count", 0),
                "brightness_score": 0.5,  # Simulado
                "contrast_score": 0.5,    # Simulado
                "edge_density": 0.3       # Simulado
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de imagen: {e}")
            return {}
    
    async def _extract_audio_features(self, processed_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características de audio"""
        try:
            analysis = processed_data.get("analysis", {})
            
            features = {
                "duration": analysis.get("estimated_duration", 0.0),
                "sample_rate": analysis.get("sample_rate", 0),
                "size_bytes": analysis.get("size_bytes", 0),
                "format": analysis.get("format", "unknown"),
                "volume_level": 0.5,      # Simulado
                "frequency_center": 1000,  # Simulado
                "spectral_centroid": 0.5   # Simulado
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de audio: {e}")
            return {}
    
    async def _extract_structured_features(self, processed_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae características de datos estructurados"""
        try:
            analysis = processed_data.get("analysis", {})
            
            features = {
                "data_type": analysis.get("type", "unknown"),
                "size": analysis.get("size", 0),
                "depth": analysis.get("depth", 0),
                "key_count": len(analysis.get("keys", [])) if analysis.get("keys") else 0,
                "data_type_diversity": len(set(analysis.get("data_types", []))),
                "complexity_score": self._calculate_structured_complexity(processed_data.get("data"))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características estructuradas: {e}")
            return {}
    
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto"""
        # Implementación simple basada en patrones
        spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"]
        english_words = ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that"]
        
        text_lower = text.lower()
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if spanish_count > english_count:
            return "spanish"
        elif english_count > spanish_count:
            return "english"
        else:
            return "unknown"
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analiza el sentimiento del texto"""
        # Implementación simple
        positive_words = ["bueno", "excelente", "genial", "fantástico", "perfecto", "good", "excellent", "great", "amazing"]
        negative_words = ["malo", "terrible", "horrible", "pésimo", "awful", "bad", "terrible", "horrible", "awful"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extrae entidades del texto"""
        # Implementación simple
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave del texto"""
        # Implementación simple
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Solo palabras de más de 3 caracteres
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Retornar las 10 palabras más frecuentes
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calcula la complejidad del texto"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Normalizar (valores arbitrarios)
        complexity = (avg_words_per_sentence * 0.1 + avg_word_length * 0.1) / 2
        return min(1.0, max(0.0, complexity))
    
    def _calculate_readability(self, text: str) -> float:
        """Calcula la legibilidad del texto"""
        # Implementación simple del índice de Flesch
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Fórmula simplificada
        readability = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        
        # Normalizar a 0-1
        return max(0.0, min(1.0, readability / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Cuenta las sílabas de una palabra"""
        vowels = "aeiouy"
        word = word.lower()
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _calculate_aspect_ratio(self, size: Tuple[int, int]) -> float:
        """Calcula la relación de aspecto"""
        if size[1] == 0:
            return 0.0
        return size[0] / size[1]
    
    def _calculate_depth(self, data: Union[Dict, List], current_depth: int = 0) -> int:
        """Calcula la profundidad de datos estructurados"""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_depth(value, current_depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth
    
    def _analyze_data_types(self, data: Union[Dict, List]) -> List[str]:
        """Analiza los tipos de datos en estructuras"""
        types = set()
        
        if isinstance(data, dict):
            for value in data.values():
                types.add(type(value).__name__)
                if isinstance(value, (dict, list)):
                    types.update(self._analyze_data_types(value))
        elif isinstance(data, list):
            for item in data:
                types.add(type(item).__name__)
                if isinstance(item, (dict, list)):
                    types.update(self._analyze_data_types(item))
        
        return list(types)
    
    def _calculate_structured_complexity(self, data: Union[Dict, List]) -> float:
        """Calcula la complejidad de datos estructurados"""
        if isinstance(data, dict):
            return len(data) * 0.1
        elif isinstance(data, list):
            return len(data) * 0.05
        else:
            return 0.0
    
    def _calculate_confidence(self, features: Dict[str, Any], input_type: InputType) -> float:
        """Calcula la confianza del procesamiento"""
        try:
            if input_type == InputType.TEXT:
                # Basado en longitud y complejidad
                length_score = min(1.0, features.get("text_length", 0) / 1000)
                complexity_score = features.get("complexity_score", 0.5)
                return (length_score + complexity_score) / 2
            
            elif input_type == InputType.IMAGE:
                # Basado en tamaño y características
                size_score = min(1.0, (features.get("width", 0) * features.get("height", 0)) / (1024 * 1024))
                return size_score
            
            elif input_type == InputType.AUDIO:
                # Basado en duración
                duration = features.get("duration", 0)
                duration_score = min(1.0, duration / 60)  # Normalizar a 1 minuto
                return duration_score
            
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculando confianza: {e}")
            return 0.0
    
    async def get_perception_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de percepción"""
        return {
            "total_inputs_processed": self.total_inputs_processed,
            "successful_processing": self.successful_processing,
            "failed_processing": self.failed_processing,
            "success_rate": (self.successful_processing / max(self.total_inputs_processed, 1)) * 100,
            "supported_input_types": [t.value for t in InputType],
            "total_results": len(self.perception_results)
        }
    
    async def save_state(self):
        """Guarda el estado del sistema de percepción"""
        try:
            state = {
                "perception_results": {
                    result_id: {
                        "input_id": result.input_id,
                        "input_type": result.input_type.value,
                        "stage": result.stage.value,
                        "features": result.features,
                        "metadata": result.metadata,
                        "confidence": result.confidence,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result_id, result in self.perception_results.items()
                },
                "stats": {
                    "total_inputs_processed": self.total_inputs_processed,
                    "successful_processing": self.successful_processing,
                    "failed_processing": self.failed_processing
                },
                "timestamp": datetime.now().isoformat()
            }
            
            with open("data/perception_state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info("Estado del sistema de percepción guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado de percepción: {e}")

# Instancia global del sistema de percepción
perception_system = PerceptionSystem()

async def initialize_module(core_engine):
    """Inicializa el módulo de percepción"""
    global perception_system
    perception_system.core_engine = core_engine
    core_engine.perception_system = perception_system
    logger.info("Módulo de percepción inicializado")

async def process(input_data, context):
    """Procesa entrada a través del sistema de percepción"""
    if isinstance(input_data, str):
        # Procesar como texto
        result = await perception_system.process_input(input_data, InputType.TEXT, context)
        return {
            "input_type": "text",
            "features": result.features,
            "confidence": result.confidence,
            "metadata": result.metadata
        }
    elif isinstance(input_data, dict):
        # Procesar como datos estructurados
        result = await perception_system.process_input(input_data, InputType.STRUCTURED_DATA, context)
        return {
            "input_type": "structured",
            "features": result.features,
            "confidence": result.confidence,
            "metadata": result.metadata
        }
    
    return input_data

def run_modulo6():
    """Función de compatibilidad con el sistema anterior"""
    print("👁️ Módulo 6: Sistema de Percepción")
    print("   - Procesamiento de texto, imagen y audio")
    print("   - Extracción de características")
    print("   - Análisis de datos estructurados")
    print("   - Preprocesamiento inteligente")
    print("   ✅ Módulo inicializado correctamente")