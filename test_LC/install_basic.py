#!/usr/bin/env python3
"""
Instalación Básica para WoldVirtual3DlucIA
Versión: 0.6.0
Instala solo las dependencias esenciales para funcionamiento básico
"""

import subprocess
import sys
import os
from pathlib import Path

def print_header():
    """Imprime el header"""
    print("=" * 60)
    print("📦 WoldVirtual3DlucIA - Instalación Básica v0.6.0")
    print("=" * 60)
    print()

def create_directories():
    """Crea directorios necesarios"""
    print("📁 Creando directorios...")
    
    directories = [
        "logs", "data", "data/memory", "data/learning", "data/communication",
        "data/training", "data/perception", "data/action", "data/evaluation",
        "data/optimization", "data/security", "data/monitoring", "data/integration",
        "models", "models/neural", "models/decision", "models/optimization",
        "cache", "temp", "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")
    
    print("✅ Directorios creados\n")

def install_basic_dependencies():
    """Instala dependencias básicas"""
    print("📦 Instalando dependencias básicas...")
    
    # Dependencias esenciales para funcionamiento básico
    basic_deps = [
        "numpy",
        "aiohttp", 
        "requests",
        "Pillow"
    ]
    
    for dep in basic_deps:
        try:
            print(f"🔄 Instalando {dep}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", dep, "--quiet"
            ], check=True, capture_output=True)
            print(f"✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {dep} no se pudo instalar: {e}")
    
    print("✅ Dependencias básicas instaladas\n")

def create_basic_config():
    """Crea configuración básica"""
    print("⚙️  Creando configuración básica...")
    
    import json
    
    # Configuración mínima
    config = {
        "ai_name": "LucIA",
        "version": "0.6.0",
        "max_memory_size": 100000,
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
    
    print("✅ config/ai_config.json creado\n")

def test_basic_functionality():
    """Prueba funcionalidad básica"""
    print("🧪 Probando funcionalidad básica...")
    
    try:
        # Test de importación básica
        import main
        print("✅ main.py importado correctamente")
        
        # Test de configuración
        with open("config/ai_config.json", "r") as f:
            config = json.load(f)
        
        if config.get("ai_name") == "LucIA":
            print("✅ Configuración cargada correctamente")
        else:
            print("⚠️  Configuración no válida")
        
        print("✅ Funcionalidad básica verificada\n")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba básica: {e}")
        return False

def main():
    """Función principal"""
    print_header()
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias básicas
    install_basic_dependencies()
    
    # Crear configuración básica
    create_basic_config()
    
    # Probar funcionalidad básica
    if test_basic_functionality():
        print("🎉 ¡Instalación básica completada exitosamente!")
        print()
        print("📋 Para ejecutar LucIA:")
        print("   python run_lucia.py")
        print()
        print("📋 Para instalación completa:")
        print("   python setup_lucia.py")
        print()
        print("📋 Para testing completo:")
        print("   python test_lucia_system.py")
        print("=" * 60)
    else:
        print("❌ Instalación básica falló")
        print("💡 Revisa los errores anteriores")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Instalación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la instalación: {e}")
        sys.exit(1)
