#!/usr/bin/env python3
"""
WoldVirtual3DlucIA - Motor Principal de Inteligencia Artificial Modular
Versión: 0.6.0
Autor: Sistema de IA Autónomo
Fecha: 2025-01-11

Motor principal que coordina todos los módulos de IA especializados.
Diseñado para ser entrenado por otras IAs existentes.
"""

import asyncio
import logging
import sys
import os
import subprocess
import importlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Importar entrenador de seguridad profunda
from celebro.deep_security_training import DeepSecurityTrainer
import threading
import time

# Importar sistema de resolución de errores críticos
from solucion_de_errores.integration import integrate_error_resolution

# Configuración de logging
# Crear directorio de logs si no existe
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lucIA_core.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('LucIA_Core')

class LucIACore:
    """
    Motor principal de la IA LucIA.
    Coordina todos los módulos especializados y gestiona el aprendizaje.
    """
    
    def __init__(self, config_path: str = "config/ai_config.json", auto_install: bool = True, auto_test: bool = True):
        self.config_path = config_path
        self.auto_install = auto_install
        self.auto_test = auto_test
        self.config = self._load_config()
        self.modules = {}
        self.memory_system = None
        self.learning_engine = None
        self.communication_hub = None
        self.training_interface = None
        self.is_running = False
        self.performance_metrics = {}
        self.gemini_integration = None
        self.security_trainer = DeepSecurityTrainer()
        self.error_resolution_integration = None
        
        # Inicializar directorios necesarios
        self._setup_directories()
        
        # Instalar dependencias automáticamente si está habilitado
        if self.auto_install:
            self._install_dependencies()
        
        logger.info("LucIA Core inicializado correctamente")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde archivo JSON"""
        default_config = {
            "ai_name": "LucIA",
            "version": "0.6.0",
            "max_memory_size": 1000000,
            "learning_rate": 0.01,
            "modules": {
                "memory": {"enabled": True, "priority": 1},
                "learning": {"enabled": True, "priority": 2},
                "communication": {"enabled": True, "priority": 3},
                "training": {"enabled": True, "priority": 4},
                "reasoning": {"enabled": True, "priority": 5},
                "perception": {"enabled": True, "priority": 6},
                "action": {"enabled": True, "priority": 7},
                "evaluation": {"enabled": True, "priority": 8},
                "optimization": {"enabled": True, "priority": 9},
                "security": {"enabled": True, "priority": 10},
                "monitoring": {"enabled": True, "priority": 11},
                "integration": {"enabled": True, "priority": 12},
                "advanced_integration": {"enabled": True, "priority": 13},
                "infrastructure": {"enabled": True, "priority": 14}
            },
            "training": {
                "auto_learning": True,
                "external_ai_support": True,
                "model_update_frequency": 3600
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"Configuración cargada desde {self.config_path}")
                return {**default_config, **config}
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return default_config
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return default_config
    
    def _setup_directories(self):
        """Crea los directorios necesarios para el funcionamiento"""
        directories = [
            "logs", "data", "models", "cache", "config", "temp",
            "data/memory", "data/learning", "data/communication",
            "models/neural", "models/decision", "models/optimization"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directorio creado/verificado: {directory}")
    
    def _install_dependencies(self):
        """Instala dependencias automáticamente"""
        try:
            print("🔧 Verificando e instalando dependencias...")
            
            # Lista de dependencias críticas
            critical_deps = [
                "numpy", "pandas", "scikit-learn", "matplotlib", 
                "requests", "aiohttp", "sqlalchemy", "cryptography",
                "PyJWT", "Pillow", "psutil", "python-dotenv"
            ]
            
            missing_deps = []
            
            for dep in critical_deps:
                try:
                    importlib.import_module(dep)
                    print(f"✅ {dep} - OK")
                except ImportError:
                    missing_deps.append(dep)
                    print(f"❌ {dep} - Faltante")
            
            if missing_deps:
                print(f"\n📦 Instalando {len(missing_deps)} dependencias faltantes...")
                for dep in missing_deps:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print(f"✅ {dep} instalado")
                    except subprocess.CalledProcessError:
                        print(f"❌ Error instalando {dep}")
            
            print("✅ Verificación de dependencias completada")
            
        except Exception as e:
            logger.warning(f"Error en instalación automática de dependencias: {e}")
    
    async def _run_auto_tests(self):
        """Ejecuta tests automáticos al inicializar"""
        try:
            if not self.auto_test:
                return
            
            print("\n🧪 Ejecutando tests automáticos...")
            
            # Importar y ejecutar tests
            from auto_tests import run_auto_tests
            test_report = await run_auto_tests()
            
            if test_report and test_report.get('status') == 'READY':
                print("✅ Tests automáticos - SISTEMA LISTO")
            elif test_report and test_report.get('status') == 'WARNING':
                print("⚠️ Tests automáticos - SISTEMA FUNCIONAL CON ADVERTENCIAS")
            else:
                print("❌ Tests automáticos - ERRORES DETECTADOS")
            
        except Exception as e:
            logger.warning(f"Error ejecutando tests automáticos: {e}")
    
    async def _initialize_error_resolution(self):
        """Inicializa el sistema de resolución de errores críticos"""
        try:
            print("🔧 Inicializando sistema de resolución de errores críticos...")
            
            # Integrar sistema de resolución de errores
            self.error_resolution_integration = await integrate_error_resolution(self)
            
            if self.error_resolution_integration and self.error_resolution_integration.is_initialized:
                print("✅ Sistema de resolución de errores críticos activado")
                logger.info("Sistema de resolución de errores críticos inicializado correctamente")
            else:
                print("⚠️ Sistema de resolución de errores críticos no pudo ser inicializado")
                logger.warning("Error inicializando sistema de resolución de errores críticos")
                
        except Exception as e:
            print(f"❌ Error inicializando sistema de resolución de errores: {e}")
            logger.error(f"Error inicializando sistema de resolución de errores: {e}")
    
    async def initialize_modules(self):
        """Inicializa todos los módulos de IA de forma asíncrona"""
        logger.info("Inicializando módulos de IA...")
        
        # Importar y inicializar módulos en orden de prioridad
        module_imports = {
            "memory": "src.modulo1.main_modulo1",
            "learning": "src.modulo2.main_modulo2", 
            "communication": "src.modulo3.main_modulo3",
            "training": "src.modulo4.main_modulo4",
            "reasoning": "src.modulo5.main_modulo5",
            "perception": "src.modulo6.main_modulo6",
            "action": "src.modulo7.main_modulo7",
            "evaluation": "src.modulo8.main_modulo8",
            "optimization": "src.modulo9.main_modulo9",
            "security": "src.modulo10.main_modulo10",
            "monitoring": "src.modulo11.main_modulo11",
            "integration": "src.modulo12.main_modulo12",
            "advanced_integration": "src.modulo13.main_modulo13",
            "infrastructure": "src.modulo14.main_modulo14"
        }
        
        for module_name, module_path in module_imports.items():
            if self.config["modules"][module_name]["enabled"]:
                try:
                    module = __import__(module_path, fromlist=[''])
                    if hasattr(module, 'initialize_module'):
                        await module.initialize_module(self)
                        self.modules[module_name] = module
                        logger.info(f"Módulo {module_name} inicializado correctamente")
                    else:
                        logger.warning(f"Módulo {module_name} no tiene función initialize_module")
                except Exception as e:
                    logger.error(f"Error inicializando módulo {module_name}: {e}")
        
        logger.info(f"Total de módulos inicializados: {len(self.modules)}")
        
        # Ejecutar tests automáticos después de inicializar módulos
        if self.auto_test:
            await self._run_auto_tests()
        
        # Inicializar sistema de resolución de errores críticos
        await self._initialize_error_resolution()
    
    async def start_ai_engine(self):
        """Inicia el motor principal de IA"""
        logger.info("Iniciando motor de IA LucIA...")
        self.is_running = True
        
        try:
            # Inicializar módulos
            await self.initialize_modules()
            
            # Iniciar procesos de aprendizaje continuo
            if self.config["training"]["auto_learning"]:
                asyncio.create_task(self._continuous_learning_loop())
            
            # Iniciar monitoreo de rendimiento
            asyncio.create_task(self._performance_monitoring_loop())
            
            logger.info("Motor de IA LucIA iniciado correctamente")
            
            # Iniciar chat interactivo automáticamente
            await self._start_interactive_chat()
                
        except KeyboardInterrupt:
            logger.info("Recibida señal de interrupción")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Error en el motor principal: {e}")
            await self.shutdown()
    
    async def _start_interactive_chat(self):
        """Inicia el chat interactivo automáticamente"""
        try:
            print("\n" + "=" * 80)
            print("🤖 LUCIA v0.6.0 - CHAT INTERACTIVO INICIADO")
            print("=" * 80)
            print("🔒 Enfocado en seguridad en internet y cómo combatir amenazas vía código")
            print("🧠 Sistemas integrados: @celebro, @red_neuronal, @conocimientos, TensorFlow")
            print("🤖 Entrenamiento profundo con Gemini + TensorFlow")
            print("=" * 80)
            print("💡 Comandos especiales:")
            print("   - 'entrenar' - Entrenamiento completo de seguridad")
            print("   - 'entrenar [tema]' - Entrenar en tema específico")
            print("   - 'preguntas [tema]' - Generar preguntas de seguridad")
            print("   - 'analizar [pregunta]' - Analizar pregunta con IA")
            print("   - 'estado' - Ver estado del sistema")
            print("   - 'corregir' - Aplicar correcciones automáticas")
            print("   - 'validar' - Ejecutar validación del sistema")
            print("   - 'errores' - Ver reporte de errores")
            print("=" * 80)
            
            # Pregunta inicial automática
            initial_question = "Hola! Soy LucIA, tu asistente de ciberseguridad. ¿En qué puedo ayudarte hoy? Puedes preguntarme sobre autenticación, encriptación, malware, phishing, o cualquier tema de seguridad."
            print(f"\n🤖 LucIA: {initial_question}")
            
            # Conectar con Gemini después de la primera pregunta
            await self._connect_to_gemini()
            
            # Loop de chat interactivo
            while self.is_running:
                try:
                    print(f"\n💬 Tú: ", end="")
                    user_input = input().strip()
                    
                    if user_input.lower() in ['salir', 'exit', 'quit', 'bye', 'adiós']:
                        print("\n🤖 LucIA: ¡Hasta luego! Fue un placer ayudarte con temas de ciberseguridad.")
                        break
                    
                    # Comandos especiales de entrenamiento
                    if user_input.lower() == 'entrenar':
                        print("\n🔒 Iniciando entrenamiento profundo de seguridad...")
                        training_result = await self.start_security_training()
                        if 'error' not in training_result:
                            print(f"✅ Entrenamiento completado: {training_result.get('average_accuracy', 0):.3f} precisión")
                        else:
                            print(f"❌ Error en entrenamiento: {training_result['error']}")
                        continue
                    
                    if user_input.lower().startswith('entrenar '):
                        topic = user_input[9:].strip()
                        print(f"\n🔒 Entrenando en tema: {topic}")
                        training_result = await self.train_on_security_topic(topic)
                        if 'error' not in training_result:
                            print(f"✅ Entrenamiento completado: {training_result.get('average_accuracy', 0):.3f} precisión")
                        else:
                            print(f"❌ Error en entrenamiento: {training_result['error']}")
                        continue
                    
                    if user_input.lower().startswith('preguntas '):
                        topic = user_input[10:].strip()
                        print(f"\n❓ Generando preguntas sobre: {topic}")
                        questions = await self.generate_security_questions(topic, complexity=3, num_questions=3)
                        for i, question in enumerate(questions, 1):
                            print(f"   {i}. {question}")
                        continue
                    
                    if user_input.lower().startswith('analizar '):
                        question = user_input[9:].strip()
                        print(f"\n🔍 Analizando pregunta: {question}")
                        analysis = await self.analyze_security_question(question)
                        if 'error' not in analysis:
                            print(f"   🔒 Categoría: {analysis.get('security_category', 'N/A')}")
                            print(f"   📊 Complejidad: {analysis.get('complexity_level', 'N/A')}/5")
                            print(f"   🎯 Confianza: {analysis.get('confidence_score', 0):.3f}")
                            print(f"   ⭐ Calidad: {analysis.get('quality', 'N/A')}")
                            print(f"   📝 Respuesta: {analysis.get('text', 'N/A')[:200]}...")
                        else:
                            print(f"❌ Error en análisis: {analysis['error']}")
                        continue
                    
                    if user_input.lower() == 'estado':
                        status = await self.get_advanced_system_status()
                        security_status = await self.get_security_training_status()
                        print(f"\n📊 Estado del sistema:")
                        print(f"   🧠 Módulos activos: {len([m for m in status.get('modules', {}).values() if m.get('status') == 'active'])}")
                        print(f"   🔒 Datos de seguridad: {security_status.get('total_training_data', 0)}")
                        print(f"   🤖 Modelos TensorFlow: {security_status.get('tensorflow_models', 0)}")
                        print(f"   🔗 Gemini conectado: {security_status.get('gemini_connected', False)}")
                        
                        # Mostrar estado del sistema de resolución de errores
                        if self.error_resolution_integration:
                            error_status = self.error_resolution_integration.get_system_status()
                            print(f"   🔧 Resolución de errores: {'Activo' if error_status.get('is_initialized') else 'Inactivo'}")
                            print(f"   ⚡ Auto-fix: {'Habilitado' if error_status.get('auto_fix_enabled') else 'Deshabilitado'}")
                        continue
                    
                    if user_input.lower() == 'corregir':
                        print("\n🔧 Aplicando correcciones automáticas...")
                        if self.error_resolution_integration:
                            fix_result = await self.error_resolution_integration.apply_automatic_fixes(self)
                            if fix_result.get('success', False):
                                print(f"✅ Correcciones aplicadas: {len(fix_result.get('fixes_applied', []))} correcciones")
                            else:
                                print(f"❌ Error en correcciones: {fix_result.get('error', 'Desconocido')}")
                        else:
                            print("❌ Sistema de resolución de errores no disponible")
                        continue
                    
                    if user_input.lower() == 'validar':
                        print("\n🔍 Ejecutando validación del sistema...")
                        if self.error_resolution_integration:
                            validation_result = await self.error_resolution_integration.run_automatic_validation(self)
                            overall_status = validation_result.get('overall_status', 'UNKNOWN')
                            print(f"📊 Estado de validación: {overall_status}")
                            if overall_status == 'HEALTHY':
                                print("✅ Sistema funcionando correctamente")
                            elif overall_status == 'WARNING':
                                print("⚠️ Sistema con advertencias")
                            else:
                                print("❌ Sistema con errores detectados")
                        else:
                            print("❌ Sistema de resolución de errores no disponible")
                        continue
                    
                    if user_input.lower() == 'errores':
                        print("\n📊 Reporte de errores del sistema...")
                        if self.error_resolution_integration:
                            error_status = self.error_resolution_integration.get_system_status()
                            resolution_status = error_status.get('resolution_status', {})
                            metrics = resolution_status.get('metrics', {})
                            print(f"   🔧 Errores corregidos: {metrics.get('errors_fixed', 0)}")
                            print(f"   ⚡ Sistemas optimizados: {metrics.get('systems_optimized', 0)}")
                            print(f"   🎯 Predicciones mejoradas: {metrics.get('predictions_enhanced', 0)}")
                            print(f"   ✅ Validaciones completadas: {metrics.get('validations_completed', 0)}")
                            print(f"   📊 Monitoreo activo: {resolution_status.get('monitoring_active', False)}")
                        else:
                            print("❌ Sistema de resolución de errores no disponible")
                        continue
                    
                    if user_input:
                        # Procesar entrada del usuario
                        response = await self._process_user_input(user_input)
                        print(f"\n🤖 LucIA: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\n🤖 LucIA: ¡Hasta luego! Fue un placer ayudarte.")
                    break
                except Exception as e:
                    print(f"\n❌ Error procesando entrada: {e}")
                    print("🤖 LucIA: Disculpa, hubo un error. ¿Puedes repetir tu pregunta?")
            
        except Exception as e:
            logger.error(f"Error en chat interactivo: {e}")
    
    async def _connect_to_gemini(self):
        """Conecta con Gemini API después de la primera pregunta"""
        try:
            print("\n🔗 Conectando con Gemini API...")
            
            # Probar conexión directa con Gemini
            try:
                from celebro.red_neuronal.gemini_integration import GeminiIntegration
                gemini = GeminiIntegration()
                
                # Probar la conexión
                test_result = gemini.generate_text("Hola, soy LucIA. ¿Puedes confirmar la conexión?")
                test_response = test_result.get('text', '') if isinstance(test_result, dict) else str(test_result)
                if test_response and len(test_response.strip()) > 0:
                    print("✅ Conexión con Gemini establecida correctamente")
                    print(f"🧠 Gemini: {test_response}")
                    
                    # Guardar la instancia de Gemini para uso posterior
                    self.gemini_integration = gemini
                    return True
                else:
                    print("⚠️ Conexión con Gemini establecida pero sin respuesta válida")
                    return False
                    
            except Exception as e:
                print(f"⚠️ Error conectando con Gemini: {e}")
                print("🔄 Continuando con funcionalidades locales...")
                return False
                
        except Exception as e:
            logger.warning(f"Error conectando con Gemini: {e}")
            print("🔄 Continuando con funcionalidades locales...")
            return False
    
    async def _process_user_input(self, user_input: str) -> str:
        """Procesa la entrada del usuario y genera respuesta"""
        try:
            # Si hay integración avanzada, usar el sistema de aprendizaje profundo
            if "advanced_integration" in self.modules:
                try:
                    integration_module = self.modules["advanced_integration"]
                    if hasattr(integration_module, 'red_neuronal_core') and integration_module.red_neuronal_core:
                        # Usar el sistema de aprendizaje profundo
                        neural_core = integration_module.red_neuronal_core
                        
                        # Generar prompt adaptativo
                        adaptive_prompt = await neural_core.generate_adaptive_prompt(user_input, {
                            'session_context': 'main_chat',
                            'user_level': 'interactive'
                        })
                        
                        # Obtener respuesta de Gemini con prompt adaptativo
                        if hasattr(self, 'gemini_integration') and self.gemini_integration:
                            gemini_result = self.gemini_integration.generate_text(adaptive_prompt)
                            gemini_response = gemini_result.get('text', '') if isinstance(gemini_result, dict) else str(gemini_result)
                            
                            if gemini_response and len(gemini_response.strip()) > 0:
                                # Analizar la consulta para insights
                                query_analysis = await neural_core.analyze_query_deep(user_input)
                                
                                return f"🧠 Gemini + LucIA (Análisis Profundo): {gemini_response}\n\n📊 Análisis: {query_analysis['complexity']} - {query_analysis['category']} - Potencial: {query_analysis['learning_potential']:.2f}"
                except Exception as e:
                    logger.warning(f"Error usando sistema de aprendizaje profundo: {e}")
            
            # Fallback: Si Gemini está disponible, usarlo como segunda opción
            elif hasattr(self, 'gemini_integration') and self.gemini_integration:
                try:
                    # Crear prompt especializado en ciberseguridad
                    security_prompt = f"""Eres LucIA, un asistente especializado en ciberseguridad. 
                    Responde de manera técnica pero accesible sobre temas de seguridad en internet.
                    Pregunta del usuario: {user_input}
                    
                    Enfócate en proporcionar información práctica sobre:
                    - Autenticación y autorización
                    - Encriptación y cifrado
                    - Detección y prevención de malware
                    - Protección contra phishing
                    - Configuración de firewalls
                    - Gestión de vulnerabilidades
                    - Desarrollo seguro de aplicaciones
                    
                    Responde en español y de forma concisa pero completa."""
                    
                    gemini_result = self.gemini_integration.generate_text(security_prompt)
                    gemini_response = gemini_result.get('text', '') if isinstance(gemini_result, dict) else str(gemini_result)
                    if gemini_response and len(gemini_response.strip()) > 0:
                        return f"🧠 Gemini + LucIA: {gemini_response}"
                except Exception as e:
                    logger.warning(f"Error usando Gemini: {e}")
            
            # Si hay integración avanzada, usar @celebro para mejorar la respuesta
            if "advanced_integration" in self.modules:
                try:
                    integration_module = self.modules["advanced_integration"]
                    if hasattr(integration_module, 'celebro_core') and integration_module.celebro_core:
                        enhanced_response = await integration_module._process_with_celebro(user_input, {})
                        if enhanced_response and enhanced_response != user_input:
                            return enhanced_response
                except Exception as e:
                    logger.warning(f"Error procesando con @celebro: {e}")
            
            # Procesar a través del sistema principal
            processed_input = await self.process_input(user_input)
            
            # Generar respuesta básica si no hay procesamiento avanzado
            if isinstance(processed_input, str) and processed_input != user_input:
                return processed_input
            else:
                # Respuesta por defecto basada en el tema de ciberseguridad
                return self._generate_security_response(user_input)
                
        except Exception as e:
            logger.error(f"Error procesando entrada del usuario: {e}")
            return "Disculpa, hubo un error procesando tu consulta. ¿Puedes reformular tu pregunta?"
    
    def _generate_security_response(self, user_input: str) -> str:
        """Genera una respuesta básica sobre ciberseguridad"""
        user_lower = user_input.lower()
        
        # Respuestas temáticas básicas
        if any(word in user_lower for word in ['autenticación', 'autenticacion', 'login', 'password']):
            return "La autenticación es fundamental en ciberseguridad. Te recomiendo implementar autenticación de dos factores (2FA), usar contraseñas fuertes y considerar sistemas como OAuth 2.0 o SAML para aplicaciones empresariales."
        
        elif any(word in user_lower for word in ['encriptación', 'encriptacion', 'cifrado', 'encryption']):
            return "La encriptación protege los datos en tránsito y en reposo. Para aplicaciones web, usa HTTPS (TLS 1.3), para bases de datos considera AES-256, y para comunicaciones implementa end-to-end encryption."
        
        elif any(word in user_lower for word in ['malware', 'virus', 'ransomware']):
            return "El malware es una amenaza constante. Implementa antivirus actualizados, sandboxing, análisis de comportamiento, y educa a los usuarios sobre phishing y descargas sospechosas."
        
        elif any(word in user_lower for word in ['phishing', 'estafa', 'fraude']):
            return "El phishing es muy común. Usa filtros de email, educa a los usuarios sobre señales de alerta, implementa SPF, DKIM y DMARC, y considera herramientas de detección de phishing."
        
        elif any(word in user_lower for word in ['firewall', 'cortafuegos']):
            return "Los firewalls son la primera línea de defensa. Configura reglas estrictas, usa firewalls de próxima generación (NGFW), implementa segmentación de red y monitorea el tráfico constantemente."
        
        elif any(word in user_lower for word in ['vulnerabilidad', 'vulnerabilidades', 'exploit']):
            return "La gestión de vulnerabilidades es crucial. Implementa escaneo regular de vulnerabilidades, parcheo automático, análisis de dependencias y un programa de bug bounty."
        
        else:
            return f"Interesante pregunta sobre '{user_input}'. Como especialista en ciberseguridad, puedo ayudarte con temas como autenticación, encriptación, malware, phishing, firewalls, vulnerabilidades, y desarrollo seguro. ¿Hay algún tema específico que te interese?"
    
    async def _continuous_learning_loop(self):
        """Loop continuo de aprendizaje automático"""
        while self.is_running:
            try:
                if "learning" in self.modules:
                    # Importar y llamar la función directamente
                    from src.modulo2.main_modulo2 import process_learning_cycle
                    await process_learning_cycle()
                
                await asyncio.sleep(self.config["training"]["model_update_frequency"])
            except Exception as e:
                logger.error(f"Error en loop de aprendizaje: {e}")
                await asyncio.sleep(60)  # Esperar antes de reintentar
    
    async def _performance_monitoring_loop(self):
        """Loop de monitoreo de rendimiento"""
        while self.is_running:
            try:
                self.performance_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "modules_active": len(self.modules),
                    "memory_usage": self._get_memory_usage(),
                    "cpu_usage": self._get_cpu_usage(),
                    "learning_cycles": self._get_learning_cycles()
                }
                
                # Guardar métricas
                with open("data/performance_metrics.json", "w") as f:
                    json.dump(self.performance_metrics, f, indent=2)
                
                await asyncio.sleep(300)  # Actualizar cada 5 minutos
            except Exception as e:
                logger.error(f"Error en monitoreo de rendimiento: {e}")
                await asyncio.sleep(60)
    
    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria actual"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Obtiene el uso de CPU actual"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_learning_cycles(self) -> int:
        """Obtiene el número de ciclos de aprendizaje completados"""
        if "learning" in self.modules and hasattr(self.modules["learning"], 'learning_cycles'):
            return self.modules["learning"].learning_cycles
        return 0
    
    async def process_input(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Procesa entrada de datos a través de todos los módulos"""
        logger.info(f"Procesando entrada: {type(input_data).__name__}")
        
        try:
            # Procesar a través de la cadena de módulos
            result = input_data
            context = context or {}
            
            # Ordenar módulos por prioridad
            sorted_modules = sorted(
                self.modules.items(),
                key=lambda x: self.config["modules"][x[0]]["priority"]
            )
            
            for module_name, module in sorted_modules:
                if hasattr(module, 'process'):
                    result = await module.process(result, context)
                    logger.debug(f"Módulo {module_name} procesó entrada")
            
            return result
            
        except Exception as e:
            logger.error(f"Error procesando entrada: {e}")
            return None
    
    async def train_with_external_ai(self, training_data: Any, ai_interface: str) -> bool:
        """Permite entrenamiento por IAs externas"""
        logger.info(f"Entrenamiento con IA externa: {ai_interface}")
        
        try:
            if "training" in self.modules:
                success = await self.modules["training"].external_training(
                    training_data, ai_interface
                )
                if success:
                    logger.info("Entrenamiento externo completado exitosamente")
                    return True
                else:
                    logger.warning("Entrenamiento externo falló")
                    return False
            else:
                logger.error("Módulo de entrenamiento no disponible")
                return False
                
        except Exception as e:
            logger.error(f"Error en entrenamiento externo: {e}")
            return False
    
    async def generate_security_prompts(self, topic: str, num_prompts: int = 5) -> List[Dict[str, Any]]:
        """Genera prompts de aprendizaje sobre temas de seguridad"""
        try:
            if "advanced_integration" in self.modules:
                return await self.modules["advanced_integration"].generate_learning_prompts(topic, num_prompts)
            else:
                logger.warning("Módulo de integración avanzada no disponible")
                return []
        except Exception as e:
            logger.error(f"Error generando prompts de seguridad: {e}")
            return []
    
    async def train_with_security_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Entrena la IA con temas específicos de ciberseguridad"""
        try:
            if "advanced_integration" in self.modules:
                return await self.modules["advanced_integration"].train_with_security_topics(topics)
            else:
                logger.warning("Módulo de integración avanzada no disponible")
                return {'error': 'Módulo no disponible'}
        except Exception as e:
            logger.error(f"Error en entrenamiento con temas de seguridad: {e}")
            return {'error': str(e)}
    
    async def get_advanced_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los sistemas avanzados integrados"""
        try:
            if "advanced_integration" in self.modules:
                return await self.modules["advanced_integration"].get_system_status()
            else:
                return {'error': 'Módulo de integración avanzada no disponible'}
        except Exception as e:
            logger.error(f"Error obteniendo estado de sistemas avanzados: {e}")
            return {'error': str(e)}
    
    async def process_with_celebro(self, input_text: str) -> str:
        """Procesa texto usando el sistema @celebro"""
        try:
            if "advanced_integration" in self.modules and self.modules["advanced_integration"].celebro_core:
                result = await self.modules["advanced_integration"]._process_with_celebro(input_text, {})
                return result if isinstance(result, str) else input_text
            else:
                logger.warning("Sistema @celebro no disponible")
                return input_text
        except Exception as e:
            logger.error(f"Error procesando con @celebro: {e}")
            return input_text
    
    # ===== MÉTODOS DE INFRAESTRUCTURA =====
    
    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Obtiene el estado de todos los sistemas de infraestructura"""
        try:
            if "infrastructure" in self.modules:
                return await self.modules["infrastructure"].get_system_status()
            else:
                return {'error': 'Módulo de infraestructura no disponible'}
        except Exception as e:
            logger.error(f"Error obteniendo estado de infraestructura: {e}")
            return {'error': str(e)}
    
    async def optimize_infrastructure(self) -> Dict[str, Any]:
        """Optimiza todos los sistemas de infraestructura"""
        try:
            if "infrastructure" in self.modules:
                return await self.modules["infrastructure"].optimize_systems()
            else:
                return {'error': 'Módulo de infraestructura no disponible'}
        except Exception as e:
            logger.error(f"Error optimizando infraestructura: {e}")
            return {'error': str(e)}
    
    async def cache_data(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Almacena datos en el caché inteligente"""
        try:
            if "infrastructure" in self.modules:
                return await self.modules["infrastructure"].cache_set(key, value, ttl)
            else:
                logger.warning("Sistema de caché no disponible")
                return False
        except Exception as e:
            logger.error(f"Error almacenando en caché: {e}")
            return False
    
    async def get_cached_data(self, key: str) -> Optional[Any]:
        """Obtiene datos del caché inteligente"""
        try:
            if "infrastructure" in self.modules:
                return await self.modules["infrastructure"].cache_get(key)
            else:
                logger.warning("Sistema de caché no disponible")
                return None
        except Exception as e:
            logger.error(f"Error obteniendo del caché: {e}")
            return None
    
    # ===== MÉTODOS DE ENTRENAMIENTO DE SEGURIDAD =====
    
    async def start_security_training(self, topics: List[str] = None) -> Dict[str, Any]:
        """Inicia entrenamiento profundo de seguridad con Gemini + TensorFlow"""
        try:
            logger.info("Iniciando entrenamiento profundo de seguridad...")
            
            if topics:
                result = await self.security_trainer.comprehensive_security_training(topics)
            else:
                # Usar temas por defecto
                default_topics = [
                    "authentication", "encryption", "malware", "phishing", 
                    "web_security", "vulnerability_assessment", "secure_coding"
                ]
                result = await self.security_trainer.comprehensive_security_training(default_topics)
            
            logger.info("Entrenamiento de seguridad completado")
            return result
            
        except Exception as e:
            logger.error(f"Error en entrenamiento de seguridad: {e}")
            return {'error': str(e)}
    
    async def train_on_security_topic(self, topic: str, complexity_levels: List[int] = [1, 2, 3, 4, 5]) -> Dict[str, Any]:
        """Entrena en un tema específico de seguridad"""
        try:
            logger.info(f"Entrenando en tema de seguridad: {topic}")
            
            result = await self.security_trainer.train_on_security_topic(
                topic=topic,
                complexity_levels=complexity_levels,
                questions_per_level=3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error entrenando en tema {topic}: {e}")
            return {'error': str(e)}
    
    async def get_security_training_status(self) -> Dict[str, Any]:
        """Obtiene el estado del entrenamiento de seguridad"""
        try:
            return await self.security_trainer.get_training_status()
        except Exception as e:
            logger.error(f"Error obteniendo estado de entrenamiento: {e}")
            return {'error': str(e)}
    
    async def analyze_security_question(self, question: str) -> Dict[str, Any]:
        """Analiza una pregunta de seguridad usando Gemini + TensorFlow"""
        try:
            # Obtener respuesta de Gemini
            response = await self.security_trainer.get_gemini_security_response(question)
            
            # Si hay modelos entrenados, usar TensorFlow para análisis adicional
            if self.security_trainer.tensorflow.models:
                # Aquí se podría agregar análisis con modelos TensorFlow
                response['tensorflow_analysis'] = 'Modelos disponibles para análisis'
            
            return response
            
        except Exception as e:
            logger.error(f"Error analizando pregunta de seguridad: {e}")
            return {'error': str(e)}
    
    async def generate_security_questions(self, topic: str, complexity: int = 3, num_questions: int = 5) -> List[str]:
        """Genera preguntas de seguridad usando Gemini"""
        try:
            return await self.security_trainer.generate_security_questions(topic, complexity, num_questions)
        except Exception as e:
            logger.error(f"Error generando preguntas de seguridad: {e}")
            return []
    
    async def shutdown(self):
        """Apaga el motor de IA de forma segura"""
        logger.info("Iniciando apagado del motor de IA...")
        self.is_running = False
        
        # Guardar estado de todos los módulos
        for module_name, module in self.modules.items():
            try:
                if hasattr(module, 'save_state'):
                    await module.save_state()
                    logger.info(f"Estado del módulo {module_name} guardado")
            except Exception as e:
                logger.error(f"Error guardando estado del módulo {module_name}: {e}")
        
        # Apagar sistema de resolución de errores
        if self.error_resolution_integration:
            try:
                await self.error_resolution_integration.shutdown()
                logger.info("Sistema de resolución de errores apagado")
            except Exception as e:
                logger.error(f"Error apagando sistema de resolución de errores: {e}")
        
        # Guardar configuración actualizada
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuración guardada")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
        
        logger.info("Motor de IA apagado correctamente")

async def main():
    """Función principal de entrada"""
    print("=" * 80)
    print("🤖 WoldVirtual3DlucIA - Motor de IA Modular v0.6.0")
    print("=" * 80)
    print("🔧 Instalando dependencias y ejecutando tests...")
    print()
    
    # Crear instancia del motor principal con auto-instalación y tests
    lucia = LucIACore(auto_install=True, auto_test=True)
    
    print("\n🚀 Iniciando LucIA con chat interactivo...")
    print("=" * 80)
    
    # Iniciar el motor (esto ahora incluye el chat interactivo)
    await lucia.start_ai_engine()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Sistema de IA detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)