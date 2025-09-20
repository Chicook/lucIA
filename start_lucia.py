#!/usr/bin/env python3
"""
Script de Inicio Rápido para WoldVirtual3DlucIA
Versión: 0.6.0
Inicia el sistema de IA modular de forma rápida y sencilla
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('LucIA_Starter')

def print_banner():
    """Imprime el banner de inicio"""
    print("=" * 80)
    print("🤖 WoldVirtual3DlucIA - Inteligencia Artificial Modular Estándar")
    print("   Versión: 0.6.0 | Lista para Entrenamiento por otras IAs")
    print("=" * 80)
    print()

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import numpy
        import asyncio
        import sqlite3
        import json
        import logging
        print("✅ Dependencias básicas verificadas")
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False

def create_directories():
    """Crea los directorios necesarios"""
    directories = [
        "data", "data/memory", "data/learning", "data/communication",
        "data/training", "data/perception", "data/action", "data/evaluation",
        "data/optimization", "data/security", "data/monitoring", "data/integration",
        "models", "models/neural", "models/decision", "models/optimization",
        "logs", "cache", "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directorios del sistema creados")

def run_quick_test():
    """Ejecuta un test rápido del sistema"""
    try:
        print("🧪 Ejecutando test rápido...")
        
        # Importar y probar módulos básicos
        from main import LucIACore
        from src.modulo1.main_modulo1 import MemorySystem
        from src.modulo2.main_modulo2 import LearningEngine
        
        # Test básico de memoria
        memory = MemorySystem()
        print("✅ Sistema de memoria: OK")
        
        # Test básico de aprendizaje
        learning = LearningEngine()
        print("✅ Sistema de aprendizaje: OK")
        
        print("✅ Test rápido completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en test rápido: {e}")
        return False

async def start_lucia_interactive():
    """Inicia LucIA en modo interactivo"""
    try:
        from main import LucIACore
        
        print("🚀 Iniciando LucIA en modo interactivo...")
        print("💡 Escribe 'help' para ver comandos disponibles")
        print("💡 Escribe 'quit' para salir")
        print()
        
        # Inicializar LucIA
        lucia = LucIACore()
        
        # Mostrar estado inicial
        print("📊 Estado del sistema:")
        print(f"   - Módulos activos: {len(lucia.modules)}")
        print(f"   - Configuración: {lucia.config['ai_name']} v{lucia.config['version']}")
        print()
        
        # Loop interactivo
        while True:
            try:
                user_input = input("LucIA> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'salir']:
                    print("👋 ¡Hasta luego!")
                    break
                elif user_input.lower() == 'help':
                    print_help()
                elif user_input.lower() == 'status':
                    await show_status(lucia)
                elif user_input.lower() == 'test':
                    await run_interactive_test(lucia)
                elif user_input:
                    # Procesar entrada con LucIA
                    result = await lucia.process_input(user_input, {"mode": "interactive"})
                    print(f"LucIA: {result}")
                else:
                    continue
                    
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Apagar sistema
        await lucia.shutdown()
        
    except Exception as e:
        print(f"❌ Error iniciando LucIA: {e}")

def print_help():
    """Muestra la ayuda de comandos"""
    print("\n📚 Comandos disponibles:")
    print("   help     - Muestra esta ayuda")
    print("   status   - Muestra el estado del sistema")
    print("   test     - Ejecuta un test interactivo")
    print("   quit     - Sale del sistema")
    print("   <texto>  - Procesa el texto con LucIA")
    print()

async def show_status(lucia):
    """Muestra el estado del sistema"""
    try:
        print("\n📊 Estado del Sistema LucIA:")
        print(f"   - Nombre: {lucia.config['ai_name']}")
        print(f"   - Versión: {lucia.config['version']}")
        print(f"   - Módulos activos: {len(lucia.modules)}")
        print(f"   - Estado: {'🟢 Activo' if lucia.is_running else '🔴 Inactivo'}")
        
        # Mostrar estadísticas de módulos
        if hasattr(lucia, 'memory_system'):
            memory_stats = await lucia.memory_system.get_memory_stats()
            print(f"   - Memorias almacenadas: {memory_stats.get('total_memories', 0)}")
        
        if hasattr(lucia, 'learning_engine'):
            learning_stats = await lucia.learning_engine.get_learning_stats()
            print(f"   - Ciclos de aprendizaje: {learning_stats.get('learning_cycles', 0)}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error obteniendo estado: {e}")

async def run_interactive_test(lucia):
    """Ejecuta un test interactivo"""
    try:
        print("\n🧪 Ejecutando test interactivo...")
        
        # Test de procesamiento de texto
        test_input = "Hola, soy una prueba del sistema LucIA"
        result = await lucia.process_input(test_input, {"test": True})
        print(f"✅ Test de procesamiento: {type(result).__name__}")
        
        # Test de memoria
        if hasattr(lucia, 'memory_system'):
            memory_id = await lucia.memory_system.store_memory(
                content="Test de memoria interactiva",
                memory_type="test",
                importance=0.5
            )
            print(f"✅ Test de memoria: {memory_id[:8]}...")
        
        print("✅ Test interactivo completado")
        print()
        
    except Exception as e:
        print(f"❌ Error en test interactivo: {e}")

def main():
    """Función principal"""
    print_banner()
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Ejecutar test rápido
    if not run_quick_test():
        print("⚠️  Advertencia: Algunos tests fallaron, pero continuando...")
        print()
    
    # Preguntar modo de inicio
    print("🚀 ¿Cómo quieres iniciar LucIA?")
    print("1. Modo interactivo (recomendado)")
    print("2. Modo automático")
    print("3. Solo verificar sistema")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(start_lucia_interactive())
        elif choice == "2":
            print("🔄 Iniciando en modo automático...")
            from main import main as lucia_main
            asyncio.run(lucia_main())
        elif choice == "3":
            print("✅ Sistema verificado correctamente")
            print("💡 Para iniciar LucIA, ejecuta: python start_lucia.py")
        else:
            print("❌ Opción inválida")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
