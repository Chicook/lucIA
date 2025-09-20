"""
Módulo 13: Integración de Sistemas Avanzados
Versión: 0.6.0
Integra @celebro, @red_neuronal y @conocimientos con el motor principal
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Agregar rutas para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger('Modulo13_Integracion')

class SistemaIntegracion:
    """Sistema de integración de todos los módulos avanzados"""
    
    def __init__(self, core_engine):
        self.core_engine = core_engine
        self.celebro_core = None
        self.red_neuronal_core = None
        self.conocimientos_system = None
        self.is_initialized = False
        
        logger.info("Sistema de Integración inicializado")
    
    async def initialize_module(self, core_engine):
        """Inicializa el módulo de integración"""
        try:
            self.core_engine = core_engine
            
            # Importar y inicializar @celebro
            await self._initialize_celebro()
            
            # Importar y inicializar @red_neuronal
            await self._initialize_red_neuronal()
            
            # Importar y inicializar @conocimientos
            await self._initialize_conocimientos()
            
            self.is_initialized = True
            logger.info("Módulo 13 - Integración de Sistemas Avanzados inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando módulo de integración: {e}")
            raise
    
    async def _initialize_celebro(self):
        """Inicializa el sistema @celebro"""
        try:
            from celebro.celebro_core import CelebroCore
            
            self.celebro_core = CelebroCore()
            await self.celebro_core.initialize()
            
            logger.info("Sistema @celebro integrado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando @celebro: {e}")
            # Continuar sin @celebro si hay error
    
    async def _initialize_red_neuronal(self):
        """Inicializa el sistema @red_neuronal"""
        try:
            from celebro.red_neuronal.neural_core import NeuralCore
            from celebro.red_neuronal.gemini_integration import GeminiIntegration
            
            self.red_neuronal_core = NeuralCore()
            self.gemini_integration = GeminiIntegration()
            
            logger.info("Sistema @red_neuronal integrado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando @red_neuronal: {e}")
            # Continuar sin @red_neuronal si hay error
    
    async def _initialize_conocimientos(self):
        """Inicializa el sistema @conocimientos"""
        try:
            from celebro.red_neuronal.conocimientos import (
                SecurityTopics, PromptGenerator, KnowledgeBase,
                LearningCurriculum, DeepLearningTrainer
            )
            
            self.conocimientos_system = {
                'security_topics': SecurityTopics(),
                'prompt_generator': PromptGenerator(),
                'knowledge_base': KnowledgeBase(),
                'learning_curriculum': LearningCurriculum(),
                'deep_learning_trainer': DeepLearningTrainer()
            }
            
            logger.info("Sistema @conocimientos integrado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando @conocimientos: {e}")
            # Continuar sin @conocimientos si hay error
    
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Procesa entrada a través de todos los sistemas integrados"""
        try:
            if not self.is_initialized:
                return input_data
            
            result = input_data
            context = context or {}
            
            # Procesar con @celebro si está disponible
            if self.celebro_core:
                try:
                    result = await self._process_with_celebro(result, context)
                except Exception as e:
                    logger.warning(f"Error procesando con @celebro: {e}")
            
            # Procesar con @red_neuronal si está disponible
            if self.red_neuronal_core:
                try:
                    result = await self._process_with_red_neuronal(result, context)
                except Exception as e:
                    logger.warning(f"Error procesando con @red_neuronal: {e}")
            
            # Procesar con @conocimientos si está disponible
            if self.conocimientos_system:
                try:
                    result = await self._process_with_conocimientos(result, context)
                except Exception as e:
                    logger.warning(f"Error procesando con @conocimientos: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en procesamiento integrado: {e}")
            return input_data
    
    async def _process_with_celebro(self, input_data: Any, context: Dict) -> Any:
        """Procesa entrada con @celebro"""
        try:
            if isinstance(input_data, str):
                # Procesar respuesta externa
                processed_response = await self.celebro_core.process_external_response(input_data)
                context['celebro_analysis'] = processed_response
                return processed_response
            return input_data
        except Exception as e:
            logger.warning(f"Error en @celebro: {e}")
            return input_data
    
    async def _process_with_red_neuronal(self, input_data: Any, context: Dict) -> Any:
        """Procesa entrada con @red_neuronal"""
        try:
            if isinstance(input_data, str) and len(input_data) > 10:
                # Analizar con red neuronal
                analysis = await self.gemini_integration.analyze_text(input_data)
                context['neural_analysis'] = analysis
                return input_data
            return input_data
        except Exception as e:
            logger.warning(f"Error en @red_neuronal: {e}")
            return input_data
    
    async def _process_with_conocimientos(self, input_data: Any, context: Dict) -> Any:
        """Procesa entrada con @conocimientos"""
        try:
            if isinstance(input_data, str):
                # Buscar conocimiento relevante
                knowledge_items = self.conocimientos_system['knowledge_base'].search_knowledge(input_data)
                if knowledge_items:
                    context['security_knowledge'] = [item.content for item in knowledge_items[:3]]
                return input_data
            return input_data
        except Exception as e:
            logger.warning(f"Error en @conocimientos: {e}")
            return input_data
    
    async def generate_learning_prompts(self, topic: str, num_prompts: int = 5) -> List[Dict[str, Any]]:
        """Genera prompts de aprendizaje para un tema específico"""
        try:
            if not self.conocimientos_system:
                return []
            
            from celebro.red_neuronal.conocimientos.prompt_generator import PromptType, DifficultyLevel
            
            prompts = []
            for i in range(num_prompts):
                prompt = self.conocimientos_system['prompt_generator'].generate_prompt(
                    topic_id=topic,
                    prompt_type=PromptType.CONCEPTUAL,
                    difficulty=DifficultyLevel.MEDIO
                )
                prompts.append({
                    'id': prompt.id,
                    'title': prompt.title,
                    'content': prompt.content,
                    'expected_response': prompt.expected_response,
                    'learning_objectives': prompt.learning_objectives
                })
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error generando prompts de aprendizaje: {e}")
            return []

# Funciones de conveniencia para acceso directo
async def generate_learning_prompts(topic: str, num_prompts: int = 5) -> List[Dict[str, Any]]:
    """Función de conveniencia para generar prompts"""
    try:
        from celebro.red_neuronal.conocimientos.prompt_generator import PromptType, DifficultyLevel
        from celebro.red_neuronal.conocimientos import PromptGenerator
        
        prompt_generator = PromptGenerator()
        prompts = []
        
        for i in range(num_prompts):
            prompt = prompt_generator.generate_prompt(
                topic_id=topic,
                prompt_type=PromptType.CONCEPTUAL,
                difficulty=DifficultyLevel.MEDIO
            )
            prompts.append({
                'id': prompt.id,
                'title': prompt.title,
                'content': prompt.content,
                'expected_response': prompt.expected_response,
                'learning_objectives': prompt.learning_objectives
            })
        
        return prompts
        
    except Exception as e:
        logger.error(f"Error generando prompts de aprendizaje: {e}")
        return []
    
    async def train_with_security_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Entrena la IA con temas de seguridad específicos"""
        try:
            if not self.conocimientos_system:
                return {'error': 'Sistema de conocimientos no disponible'}
            
            # Generar datos de entrenamiento
            training_data = self.conocimientos_system['deep_learning_trainer'].generate_training_data(
                topic_ids=topics,
                num_prompts_per_topic=10
            )
            
            # Crear sesión de entrenamiento
            session = self.conocimientos_system['deep_learning_trainer'].create_training_session("security_training")
            
            return {
                'session_id': session.id,
                'training_data_count': len(training_data),
                'topics': topics,
                'status': 'ready_for_training'
            }
            
        except Exception as e:
            logger.error(f"Error en entrenamiento con temas de seguridad: {e}")
            return {'error': str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los sistemas integrados"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_initialized': self.is_initialized,
                'systems': {}
            }
            
            # Estado de @celebro
            if self.celebro_core:
                try:
                    celebro_stats = await self.celebro_core.get_stats()
                    status['systems']['celebro'] = {
                        'status': 'active',
                        'stats': celebro_stats
                    }
                except:
                    status['systems']['celebro'] = {'status': 'error'}
            else:
                status['systems']['celebro'] = {'status': 'not_available'}
            
            # Estado de @red_neuronal
            if self.red_neuronal_core:
                try:
                    status['systems']['red_neuronal'] = {
                        'status': 'active',
                        'networks_created': len(self.red_neuronal_core.networks) if hasattr(self.red_neuronal_core, 'networks') else 0
                    }
                except:
                    status['systems']['red_neuronal'] = {'status': 'error'}
            else:
                status['systems']['red_neuronal'] = {'status': 'not_available'}
            
            # Estado de @conocimientos
            if self.conocimientos_system:
                try:
                    knowledge_stats = self.conocimientos_system['knowledge_base'].get_learning_statistics()
                    curriculum_stats = self.conocimientos_system['learning_curriculum'].get_curriculum_statistics()
                    training_stats = self.conocimientos_system['deep_learning_trainer'].get_training_statistics()
                    
                    status['systems']['conocimientos'] = {
                        'status': 'active',
                        'knowledge_stats': knowledge_stats,
                        'curriculum_stats': curriculum_stats,
                        'training_stats': training_stats
                    }
                except:
                    status['systems']['conocimientos'] = {'status': 'error'}
            else:
                status['systems']['conocimientos'] = {'status': 'not_available'}
            
            return status
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {'error': str(e)}
    
    async def save_state(self):
        """Guarda el estado del módulo de integración"""
        try:
            # Guardar estado de @celebro
            if self.celebro_core and hasattr(self.celebro_core, 'save_state'):
                await self.celebro_core.save_state()
            
            # Guardar estado de @red_neuronal
            if self.red_neuronal_core and hasattr(self.red_neuronal_core, 'save_state'):
                await self.red_neuronal_core.save_state()
            
            # Guardar estado de @conocimientos
            if self.conocimientos_system and 'knowledge_base' in self.conocimientos_system:
                # El knowledge_base ya guarda automáticamente en SQLite
                pass
            
            logger.info("Estado del módulo de integración guardado")
            
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")

# Función de inicialización para el motor principal
async def initialize_module(core_engine):
    """Función de inicialización requerida por el motor principal"""
    integration_module = SistemaIntegracion(core_engine)
    await integration_module.initialize_module(core_engine)
    
    # Agregar el módulo al diccionario de módulos del core
    core_engine.modules["advanced_integration"] = integration_module
    
    return integration_module
