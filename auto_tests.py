#!/usr/bin/env python3
"""
Sistema de Tests Automáticos - LucIA v0.6.0
Ejecuta todos los tests automáticamente al cargar el sistema
"""

import asyncio
import sys
import os
import traceback
from typing import Dict, Any, List
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AutoTestRunner:
    """Ejecutor automático de tests para LucIA"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta todos los tests automáticamente"""
        print("=" * 80)
        print("🧪 SISTEMA DE TESTS AUTOMÁTICOS - LucIA v0.6.0")
        print("=" * 80)
        print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Lista de tests a ejecutar
        tests = [
            ("Dependencias", self.test_dependencies),
            ("Configuración", self.test_configuration),
            ("Módulos Básicos", self.test_basic_modules),
            ("Sistema @celebro", self.test_celebro),
            ("Sistema @red_neuronal", self.test_red_neuronal),
            ("Sistema @conocimientos", self.test_conocimientos),
            ("Integración Completa", self.test_integration),
            ("Motor Principal", self.test_main_engine)
        ]
        
        # Ejecutar todos los tests
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
        
        # Generar reporte final
        return self.generate_report()
    
    async def run_test(self, test_name: str, test_func) -> bool:
        """Ejecuta un test individual"""
        print(f"\n🔍 Ejecutando: {test_name}")
        self.total_tests += 1
        
        try:
            result = await test_func()
            if result:
                print(f"✅ {test_name} - EXITOSO")
                self.passed_tests += 1
                self.test_results[test_name] = {"status": "PASS", "error": None}
            else:
                print(f"❌ {test_name} - FALLÓ")
                self.failed_tests += 1
                self.test_results[test_name] = {"status": "FAIL", "error": "Test retornó False"}
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            self.failed_tests += 1
            self.test_results[test_name] = {"status": "ERROR", "error": str(e)}
            traceback.print_exc()
        
        return True
    
    async def test_dependencies(self) -> bool:
        """Test de dependencias críticas"""
        try:
            # Verificar dependencias básicas
            import numpy
            import pandas
            import sklearn
            import matplotlib
            import requests
            import sqlite3
            import json
            import asyncio
            import logging
            
            print("   ✅ Dependencias básicas - OK")
            
            # Verificar dependencias de IA
            try:
                import tensorflow
                print("   ✅ TensorFlow - OK")
            except ImportError:
                print("   ⚠️ TensorFlow - No disponible")
            
            try:
                import torch
                print("   ✅ PyTorch - OK")
            except ImportError:
                print("   ⚠️ PyTorch - No disponible")
            
            try:
                import google.generativeai
                print("   ✅ Google Generative AI - OK")
            except ImportError:
                print("   ⚠️ Google Generative AI - No disponible")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Dependencia faltante: {e}")
            return False
    
    async def test_configuration(self) -> bool:
        """Test de configuración del sistema"""
        try:
            # Verificar archivo de configuración
            config_path = "config/ai_config.json"
            if not os.path.exists(config_path):
                print("   ❌ Archivo de configuración no encontrado")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                import json
                config = json.load(f)
            
            # Verificar elementos críticos
            required_keys = ["ai_name", "version", "modules", "training"]
            for key in required_keys:
                if key not in config:
                    print(f"   ❌ Clave de configuración faltante: {key}")
                    return False
            
            print("   ✅ Configuración - OK")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en configuración: {e}")
            return False
    
    async def test_basic_modules(self) -> bool:
        """Test de módulos básicos"""
        try:
            # Verificar que los módulos básicos existan
            basic_modules = [
                "src.modulo1.main_modulo1",
                "src.modulo2.main_modulo2",
                "src.modulo3.main_modulo3",
                "src.modulo4.main_modulo4",
                "src.modulo5.main_modulo5"
            ]
            
            for module_path in basic_modules:
                try:
                    __import__(module_path, fromlist=[''])
                    print(f"   ✅ {module_path} - OK")
                except ImportError as e:
                    print(f"   ❌ {module_path} - ERROR: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error en módulos básicos: {e}")
            return False
    
    async def test_celebro(self) -> bool:
        """Test del sistema @celebro"""
        try:
            from celebro.celebro_core import CelebroCore
            
            # Crear instancia y probar inicialización
            celebro = CelebroCore()
            await celebro.initialize()
            
            # Probar funcionalidad básica
            test_response = "Esta es una respuesta de prueba"
            result = await celebro.process_external_response(test_response)
            
            print("   ✅ @celebro - OK")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en @celebro: {e}")
            return False
    
    async def test_red_neuronal(self) -> bool:
        """Test del sistema @red_neuronal"""
        try:
            from celebro.red_neuronal.neural_core import NeuralCore
            from celebro.red_neuronal.gemini_integration import GeminiIntegration
            
            # Crear instancias
            neural_core = NeuralCore()
            gemini_integration = GeminiIntegration()
            
            print("   ✅ @red_neuronal - OK")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en @red_neuronal: {e}")
            return False
    
    async def test_conocimientos(self) -> bool:
        """Test del sistema @conocimientos"""
        try:
            from celebro.red_neuronal.conocimientos import (
                SecurityTopics, PromptGenerator, KnowledgeBase
            )
            
            # Crear instancias
            security_topics = SecurityTopics()
            prompt_generator = PromptGenerator()
            knowledge_base = KnowledgeBase()
            
            # Probar funcionalidad básica
            topics = security_topics.get_all_topics()
            if len(topics) == 0:
                print("   ❌ No hay temas de seguridad disponibles")
                return False
            
            print(f"   ✅ @conocimientos - {len(topics)} temas disponibles")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en @conocimientos: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test de integración completa"""
        try:
            from src.modulo13.main_modulo13 import SistemaIntegracion
            
            # Crear instancia de integración
            integration = SistemaIntegracion(None)
            await integration.initialize_module(None)
            
            print("   ✅ Integración - OK")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en integración: {e}")
            return False
    
    async def test_main_engine(self) -> bool:
        """Test del motor principal"""
        try:
            from main import LucIACore
            
            # Crear instancia del motor
            lucia = LucIACore()
            
            # Verificar configuración
            if not lucia.config:
                print("   ❌ Configuración no cargada")
                return False
            
            # Verificar módulos
            if len(lucia.modules) == 0:
                print("   ⚠️ No hay módulos cargados (normal en inicialización)")
            
            print("   ✅ Motor principal - OK")
            return True
            
        except Exception as e:
            print(f"   ❌ Error en motor principal: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte final de tests"""
        print("\n" + "=" * 80)
        print("📊 REPORTE FINAL DE TESTS")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"📈 Total de tests: {self.total_tests}")
        print(f"✅ Tests exitosos: {self.passed_tests}")
        print(f"❌ Tests fallidos: {self.failed_tests}")
        print(f"📊 Tasa de éxito: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 ¡SISTEMA LISTO PARA USAR!")
            status = "READY"
        elif success_rate >= 60:
            print("\n⚠️ Sistema funcional con advertencias")
            status = "WARNING"
        else:
            print("\n❌ Sistema con errores críticos")
            status = "ERROR"
        
        print("=" * 80)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": success_rate,
            "status": status,
            "test_results": self.test_results
        }

async def run_auto_tests():
    """Función principal para ejecutar tests automáticos"""
    try:
        runner = AutoTestRunner()
        report = await runner.run_all_tests()
        
        # Guardar reporte
        with open("test_report.json", "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
        
    except Exception as e:
        print(f"\n❌ Error crítico en tests automáticos: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        asyncio.run(run_auto_tests())
    except KeyboardInterrupt:
        print("\n👋 Tests interrumpidos por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
