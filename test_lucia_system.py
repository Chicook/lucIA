#!/usr/bin/env python3
"""
Test Suite para WoldVirtual3DlucIA
Versión: 0.6.0
Sistema de testing y validación para la IA modular
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import unittest
from unittest.mock import Mock, patch

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos principales
from main import LucIACore
from src.modulo1.main_modulo1 import MemorySystem
from src.modulo2.main_modulo2 import LearningEngine
from src.modulo3.main_modulo3 import AICommunicationHub
from src.modulo4.main_modulo4 import AITrainingInterface
from src.modulo5.main_modulo5 import ReasoningEngine
from src.modulo6.main_modulo6 import PerceptionSystem
from src.modulo7.main_modulo7 import ActionSystem
from src.modulo8.main_modulo8 import EvaluationSystem
from src.modulo9.main_modulo9 import OptimizationSystem
from src.modulo10.main_modulo10 import SecuritySystem
from src.modulo11.main_modulo11 import MonitoringSystem
from src.modulo12.main_modulo12 import IntegrationSystem

# Configurar logging para tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LucIA_Test')

class TestLucIASystem(unittest.TestCase):
    """Suite de tests para el sistema LucIA"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.core_engine = None
        self.test_data = {
            "text": "Hola, soy LucIA, una IA modular",
            "structured": {"key": "value", "number": 42},
            "features": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "targets": [1, 0, 1]
        }
    
    async def test_core_engine_initialization(self):
        """Test: Inicialización del motor principal"""
        try:
            core = LucIACore()
            self.assertIsNotNone(core)
            self.assertIsNotNone(core.config)
            self.assertIsInstance(core.modules, dict)
            logger.info("✅ Core Engine inicializado correctamente")
        except Exception as e:
            self.fail(f"Error inicializando Core Engine: {e}")
    
    async def test_memory_system(self):
        """Test: Sistema de memoria"""
        try:
            memory = MemorySystem()
            
            # Test almacenar memoria
            memory_id = await memory.store_memory(
                content="Test memory content",
                memory_type="test",
                importance=0.8
            )
            self.assertIsNotNone(memory_id)
            
            # Test recuperar memoria
            memories = await memory.retrieve_memory("test")
            self.assertIsInstance(memories, list)
            
            logger.info("✅ Sistema de memoria funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de memoria: {e}")
    
    async def test_learning_system(self):
        """Test: Sistema de aprendizaje"""
        try:
            learning = LearningEngine()
            
            # Test crear modelo
            success = await learning.create_model("test_model", "decision_tree")
            self.assertTrue(success)
            
            # Test entrenar modelo
            import numpy as np
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([1, 0, 1])
            
            metrics = await learning.train_model("test_model", X, y)
            self.assertIsInstance(metrics, dict)
            
            logger.info("✅ Sistema de aprendizaje funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de aprendizaje: {e}")
    
    async def test_communication_system(self):
        """Test: Sistema de comunicación"""
        try:
            comm = AICommunicationHub()
            
            # Test estadísticas
            stats = await comm.get_communication_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn("ai_id", stats)
            
            logger.info("✅ Sistema de comunicación funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de comunicación: {e}")
    
    async def test_training_interface(self):
        """Test: Interfaz de entrenamiento"""
        try:
            training = AITrainingInterface()
            
            # Test crear sesión de entrenamiento
            session_id = await training.create_training_session(
                trainer_ai_id="test_ai",
                protocol="rest_api",
                training_data={"features": [[1, 2]], "targets": [1]},
                model_config={"type": "neural_network"}
            )
            self.assertIsNotNone(session_id)
            
            # Test obtener progreso
            progress = await training.get_training_progress(session_id)
            self.assertIsInstance(progress, dict)
            
            logger.info("✅ Interfaz de entrenamiento funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en interfaz de entrenamiento: {e}")
    
    async def test_reasoning_system(self):
        """Test: Sistema de razonamiento"""
        try:
            reasoning = ReasoningEngine()
            
            # Test razonamiento
            result = await reasoning.reason_about("¿Qué es la inteligencia artificial?")
            self.assertIsNotNone(result)
            self.assertIn("final_conclusion", result.__dict__)
            
            logger.info("✅ Sistema de razonamiento funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de razonamiento: {e}")
    
    async def test_perception_system(self):
        """Test: Sistema de percepción"""
        try:
            perception = PerceptionSystem()
            
            # Test procesar texto
            result = await perception.process_input(
                "Hola mundo", "text"
            )
            self.assertIsNotNone(result)
            self.assertIn("features", result.__dict__)
            
            logger.info("✅ Sistema de percepción funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de percepción: {e}")
    
    async def test_action_system(self):
        """Test: Sistema de acciones"""
        try:
            action = ActionSystem()
            
            # Test crear acción
            action_id = await action.create_action(
                action_type="computation",
                name="Test Action",
                description="Test computation action",
                parameters={"operation": "add", "values": [1, 2, 3]}
            )
            self.assertIsNotNone(action_id)
            
            # Test obtener estado
            status = await action.get_action_status(action_id)
            self.assertIsInstance(status, dict)
            
            logger.info("✅ Sistema de acciones funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de acciones: {e}")
    
    async def test_evaluation_system(self):
        """Test: Sistema de evaluación"""
        try:
            evaluation = EvaluationSystem()
            
            # Test evaluar rendimiento
            predictions = [1, 0, 1, 0, 1]
            actuals = [1, 1, 1, 0, 0]
            
            results = await evaluation.evaluate_performance(
                predictions, actuals, ["accuracy"]
            )
            self.assertIsInstance(results, dict)
            
            logger.info("✅ Sistema de evaluación funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de evaluación: {e}")
    
    async def test_optimization_system(self):
        """Test: Sistema de optimización"""
        try:
            optimization = OptimizationSystem()
            
            # Test función objetivo simple
            def objective_function(params):
                return sum(v**2 for v in params.values())
            
            parameter_space = {"param1": (0, 10), "param2": (0, 10)}
            
            result = await optimization.optimize_parameters(
                objective_function, parameter_space, "genetic", 10, 20
            )
            self.assertIsNotNone(result)
            self.assertIn("best_parameters", result.__dict__)
            
            logger.info("✅ Sistema de optimización funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de optimización: {e}")
    
    async def test_security_system(self):
        """Test: Sistema de seguridad"""
        try:
            security = SecuritySystem()
            
            # Test crear clave API
            api_key = await security.create_api_key(
                name="Test Key",
                permissions=["read", "write"]
            )
            self.assertIsNotNone(api_key)
            
            # Test autenticar clave API
            auth_result = await security.authenticate_api_key(api_key)
            self.assertTrue(auth_result)
            
            logger.info("✅ Sistema de seguridad funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de seguridad: {e}")
    
    async def test_monitoring_system(self):
        """Test: Sistema de monitoreo"""
        try:
            monitoring = MonitoringSystem()
            
            # Test obtener estado de salud
            health = await monitoring.get_system_health()
            self.assertIsInstance(health, dict)
            self.assertIn("status", health)
            
            # Test obtener métricas
            metrics = await monitoring.get_metrics(hours=1)
            self.assertIsInstance(metrics, list)
            
            logger.info("✅ Sistema de monitoreo funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de monitoreo: {e}")
    
    async def test_integration_system(self):
        """Test: Sistema de integración"""
        try:
            integration = IntegrationSystem()
            
            # Test registrar servicio
            success = await integration.register_service(
                service_id="test_service",
                name="Test Service",
                integration_type="rest_api",
                endpoint="https://httpbin.org/get",
                config={}
            )
            self.assertTrue(success)
            
            # Test obtener estado del servicio
            status = await integration.get_service_status("test_service")
            self.assertIsInstance(status, dict)
            
            logger.info("✅ Sistema de integración funcionando correctamente")
        except Exception as e:
            self.fail(f"Error en sistema de integración: {e}")
    
    async def test_end_to_end_workflow(self):
        """Test: Flujo completo end-to-end"""
        try:
            # Inicializar motor principal
            core = LucIACore()
            
            # Procesar entrada de texto
            result = await core.process_input(
                "Hola, soy LucIA. ¿Puedes ayudarme a aprender?",
                {"context": "test"}
            )
            
            self.assertIsNotNone(result)
            logger.info("✅ Flujo end-to-end funcionando correctamente")
            
        except Exception as e:
            self.fail(f"Error en flujo end-to-end: {e}")

class TestRunner:
    """Ejecutor de tests para LucIA"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    async def run_all_tests(self):
        """Ejecuta todos los tests"""
        print("=" * 60)
        print("🧪 WoldVirtual3DlucIA - Test Suite v0.6.0")
        print("=" * 60)
        print("Ejecutando tests del sistema de IA modular...")
        print()
        
        test_suite = TestLucIASystem()
        
        # Lista de tests a ejecutar
        tests = [
            ("Core Engine", test_suite.test_core_engine_initialization),
            ("Memory System", test_suite.test_memory_system),
            ("Learning System", test_suite.test_learning_system),
            ("Communication System", test_suite.test_communication_system),
            ("Training Interface", test_suite.test_training_interface),
            ("Reasoning System", test_suite.test_reasoning_system),
            ("Perception System", test_suite.test_perception_system),
            ("Action System", test_suite.test_action_system),
            ("Evaluation System", test_suite.test_evaluation_system),
            ("Optimization System", test_suite.test_optimization_system),
            ("Security System", test_suite.test_security_system),
            ("Monitoring System", test_suite.test_monitoring_system),
            ("Integration System", test_suite.test_integration_system),
            ("End-to-End Workflow", test_suite.test_end_to_end_workflow)
        ]
        
        for test_name, test_func in tests:
            try:
                print(f"🔄 Ejecutando: {test_name}...")
                await test_func()
                print(f"✅ {test_name}: PASSED")
                self.passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
                self.failed_tests += 1
            
            self.total_tests += 1
        
        # Mostrar resumen
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de tests"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE TESTS")
        print("=" * 60)
        print(f"Total de tests: {self.total_tests}")
        print(f"Tests exitosos: {self.passed_tests}")
        print(f"Tests fallidos: {self.failed_tests}")
        print(f"Tasa de éxito: {(self.passed_tests / max(self.total_tests, 1)) * 100:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ¡Todos los tests pasaron exitosamente!")
            print("✅ El sistema LucIA está listo para ser entrenado por otras IAs")
        else:
            print(f"\n⚠️  {self.failed_tests} tests fallaron. Revisar logs para más detalles.")
        
        print("=" * 60)

async def main():
    """Función principal de testing"""
    try:
        runner = TestRunner()
        await runner.run_all_tests()
        
        # Guardar resultados
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": runner.total_tests,
            "passed_tests": runner.passed_tests,
            "failed_tests": runner.failed_tests,
            "success_rate": (runner.passed_tests / max(runner.total_tests, 1)) * 100
        }
        
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Resultados guardados en: test_results.json")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
