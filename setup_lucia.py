#!/usr/bin/env python3
"""
Script de Configuración para WoldVirtual3DlucIA
Versión: 0.6.0
Configura automáticamente el sistema de IA modular
"""

import os
import json
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Imprime el header del script"""
    print("=" * 70)
    print("🔧 WoldVirtual3DlucIA - Configuración Automática")
    print("   Versión: 0.6.0 | Setup Inteligente")
    print("=" * 70)
    print()

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("💡 Se requiere Python 3.8 o superior")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True

def check_system_requirements():
    """Verifica los requisitos del sistema"""
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar memoria disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb < 2:
            print(f"⚠️  Memoria RAM: {memory_gb:.1f}GB (recomendado: 4GB+)")
        else:
            print(f"✅ Memoria RAM: {memory_gb:.1f}GB")
    except ImportError:
        print("⚠️  No se puede verificar memoria (psutil no instalado)")
    
    # Verificar espacio en disco
    disk_usage = os.statvfs('.')
    free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
    
    if free_gb < 1:
        print(f"⚠️  Espacio libre: {free_gb:.1f}GB (recomendado: 2GB+)")
    else:
        print(f"✅ Espacio libre: {free_gb:.1f}GB")
    
    print()

def install_dependencies():
    """Instala las dependencias necesarias"""
    print("📦 Instalando dependencias...")
    
    try:
        # Verificar si pip está disponible
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        
        # Instalar dependencias básicas
        basic_deps = [
            "numpy", "asyncio", "aiohttp", "scikit-learn", 
            "Pillow", "psutil", "requests"
        ]
        
        for dep in basic_deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                              check=True, capture_output=True)
                print(f"✅ {dep} instalado")
            except subprocess.CalledProcessError:
                print(f"⚠️  {dep} no se pudo instalar automáticamente")
        
        # Intentar instalar desde requirements.txt
        if os.path.exists("requirements.txt"):
            print("📋 Instalando desde requirements.txt...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              check=True)
                print("✅ Dependencias instaladas desde requirements.txt")
            except subprocess.CalledProcessError:
                print("⚠️  Algunas dependencias no se pudieron instalar")
        
        return True
        
    except Exception as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def create_directories():
    """Crea la estructura de directorios"""
    print("📁 Creando estructura de directorios...")
    
    directories = [
        "data", "data/memory", "data/learning", "data/communication",
        "data/training", "data/perception", "data/action", "data/evaluation",
        "data/optimization", "data/security", "data/monitoring", "data/integration",
        "models", "models/neural", "models/decision", "models/optimization",
        "logs", "cache", "temp", "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")
    
    print()

def create_config_files():
    """Crea archivos de configuración"""
    print("⚙️  Creando archivos de configuración...")
    
    # Configuración principal
    if not os.path.exists("config/ai_config.json"):
        config = {
            "ai_name": "LucIA",
            "version": "0.6.0",
            "description": "Inteligencia Artificial Modular Estándar",
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
                "integration": {"enabled": True, "priority": 12}
            }
        }
        
        with open("config/ai_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ config/ai_config.json")
    
    # Archivo de entorno
    if not os.path.exists(".env"):
        env_content = """# WoldVirtual3DlucIA - Variables de Entorno
# Configuración de APIs externas (opcional)

# OpenAI
# OPENAI_API_KEY=tu_clave_openai_aqui

# Claude
# CLAUDE_API_KEY=tu_clave_claude_aqui

# Gemini
# GEMINI_API_KEY=tu_clave_gemini_aqui

# Configuración del sistema
LUCIA_LOG_LEVEL=INFO
LUCIA_MAX_MEMORY=1000000
LUCIA_LEARNING_RATE=0.01
"""
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ .env")
    
    print()

def run_initial_test():
    """Ejecuta un test inicial"""
    print("🧪 Ejecutando test inicial...")
    
    try:
        # Test de importación
        import main
        print("✅ main.py importado correctamente")
        
        # Test de módulos básicos
        from src.modulo1.main_modulo1 import MemorySystem
        from src.modulo2.main_modulo2 import LearningEngine
        print("✅ Módulos básicos importados correctamente")
        
        # Test de configuración
        with open("config/ai_config.json", "r") as f:
            config = json.load(f)
        
        if config.get("ai_name") == "LucIA":
            print("✅ Configuración cargada correctamente")
        else:
            print("⚠️  Configuración no válida")
        
        print("✅ Test inicial completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en test inicial: {e}")
        return False

def create_startup_scripts():
    """Crea scripts de inicio"""
    print("📜 Creando scripts de inicio...")
    
    # Script de inicio para Windows
    if platform.system() == "Windows":
        bat_content = """@echo off
echo Iniciando WoldVirtual3DlucIA...
python start_lucia.py
pause
"""
        with open("start_lucia.bat", "w") as f:
            f.write(bat_content)
        print("✅ start_lucia.bat")
    
    # Script de inicio para Unix/Linux/Mac
    sh_content = """#!/bin/bash
echo "Iniciando WoldVirtual3DlucIA..."
python3 start_lucia.py
"""
    with open("start_lucia.sh", "w") as f:
        f.write(sh_content)
    
    # Hacer ejecutable en sistemas Unix
    if platform.system() != "Windows":
        os.chmod("start_lucia.sh", 0o755)
    
    print("✅ start_lucia.sh")
    print()

def show_next_steps():
    """Muestra los próximos pasos"""
    print("🎉 ¡Configuración completada exitosamente!")
    print()
    print("📋 Próximos pasos:")
    print("1. Ejecutar LucIA:")
    print("   python start_lucia.py")
    print()
    print("2. O ejecutar directamente:")
    print("   python main.py")
    print()
    print("3. Para testing completo:")
    print("   python test_lucia_system.py")
    print()
    print("4. Configurar APIs externas (opcional):")
    print("   Editar archivo .env con tus claves API")
    print()
    print("📚 Documentación completa en README.md")
    print("=" * 70)

def main():
    """Función principal de configuración"""
    print_header()
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Verificar requisitos del sistema
    check_system_requirements()
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias
    if not install_dependencies():
        print("⚠️  Algunas dependencias no se pudieron instalar")
        print("💡 Instala manualmente: pip install -r requirements.txt")
    
    # Crear archivos de configuración
    create_config_files()
    
    # Crear scripts de inicio
    create_startup_scripts()
    
    # Ejecutar test inicial
    if not run_initial_test():
        print("⚠️  Algunos tests fallaron, pero la configuración básica está lista")
    
    # Mostrar próximos pasos
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Configuración cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la configuración: {e}")
        sys.exit(1)
