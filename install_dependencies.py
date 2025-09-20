#!/usr/bin/env python3
"""
Instalador Automático de Dependencias - LucIA v0.6.0
Instala todas las dependencias necesarias para el funcionamiento completo
"""

import subprocess
import sys
import os
import importlib
from typing import List, Dict, Any

def install_package(package: str) -> bool:
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def check_package(package: str) -> bool:
    """Verifica si un paquete está instalado"""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Instala todas las dependencias necesarias"""
    print("=" * 60)
    print("🔧 INSTALADOR AUTOMÁTICO DE DEPENDENCIAS - LucIA v0.6.0")
    print("=" * 60)
    
    # Lista de dependencias necesarias
    dependencies = [
        # Dependencias básicas
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        
        # Dependencias de IA y ML
        "tensorflow",
        "torch",
        "transformers",
        "openai",
        "anthropic",
        "google-generativeai",
        
        # Dependencias de base de datos
        "sqlite3",  # Viene con Python
        "sqlalchemy",
        "psycopg2-binary",
        
        # Dependencias de web y APIs
        "requests",
        "aiohttp",
        "fastapi",
        "uvicorn",
        "websockets",
        
        # Dependencias de seguridad
        "cryptography",
        "PyJWT",
        "bcrypt",
        "passlib",
        
        # Dependencias de procesamiento
        "Pillow",
        "opencv-python",
        "librosa",
        "soundfile",
        
        # Dependencias de utilidades
        "python-dotenv",
        "pydantic",
        "click",
        "rich",
        "tqdm",
        "psutil",
        "schedule",
        
        # Dependencias de testing
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        
        # Dependencias de desarrollo
        "black",
        "flake8",
        "mypy",
        "pre-commit"
    ]
    
    print(f"\n📦 Verificando e instalando {len(dependencies)} dependencias...")
    
    installed_count = 0
    failed_count = 0
    
    for package in dependencies:
        # Verificar si ya está instalado
        if check_package(package):
            print(f"✅ {package} ya está instalado")
            installed_count += 1
        else:
            print(f"📥 Instalando {package}...")
            if install_package(package):
                installed_count += 1
            else:
                failed_count += 1
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE INSTALACIÓN")
    print("=" * 60)
    print(f"✅ Paquetes instalados: {installed_count}")
    print(f"❌ Paquetes fallidos: {failed_count}")
    print(f"📦 Total procesados: {len(dependencies)}")
    
    if failed_count == 0:
        print("\n🎉 ¡Todas las dependencias instaladas correctamente!")
        return True
    else:
        print(f"\n⚠️ {failed_count} paquetes fallaron. Revisar errores arriba.")
        return False

def verify_installation():
    """Verifica que todas las dependencias críticas estén funcionando"""
    print("\n🔍 VERIFICANDO INSTALACIÓN...")
    
    critical_packages = [
        "numpy",
        "pandas", 
        "sklearn",
        "matplotlib",
        "requests",
        "sqlite3",
        "json",
        "asyncio",
        "logging"
    ]
    
    all_good = True
    
    for package in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - ERROR: {e}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    try:
        # Instalar dependencias
        success = install_dependencies()
        
        if success:
            # Verificar instalación
            if verify_installation():
                print("\n🎉 ¡Instalación completada exitosamente!")
                print("🚀 LucIA está listo para funcionar")
            else:
                print("\n⚠️ Algunas dependencias críticas fallaron")
        else:
            print("\n❌ La instalación falló")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Instalación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
