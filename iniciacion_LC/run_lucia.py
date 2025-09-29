#!/usr/bin/env python3
"""
Script de Ejecución Rápida para WoldVirtual3DlucIA
Versión: 0.6.0
Asegura que todos los directorios existan antes de ejecutar
"""

import os
import sys
import asyncio
from pathlib import Path

def create_required_directories():
    """Crea todos los directorios necesarios"""
    print("📁 Creando directorios necesarios...")
    
    directories = [
        "logs",
        "data", "data/memory", "data/learning", "data/communication",
        "data/training", "data/perception", "data/action", "data/evaluation",
        "data/optimization", "data/security", "data/monitoring", "data/integration",
        "models", "models/neural", "models/decision", "models/optimization",
        "cache", "temp", "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")
    
    print("✅ Todos los directorios creados correctamente\n")

def check_dependencies():
    """Verifica dependencias básicas"""
    print("🔍 Verificando dependencias básicas...")
    
    try:
        import asyncio
        import json
        import logging
        import sqlite3
        print("✅ Dependencias básicas de Python: OK")
        
        # Verificar si numpy está disponible (opcional)
        try:
            import numpy
            print("✅ NumPy: OK")
        except ImportError:
            print("⚠️  NumPy no encontrado (opcional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("🤖 WoldVirtual3DlucIA - Ejecución Rápida v0.6.0")
    print("=" * 60)
    print()
    
    # Crear directorios necesarios
    create_required_directories()
    
    # Verificar dependencias básicas
    if not check_dependencies():
        print("❌ Faltan dependencias básicas")
        print("💡 Ejecuta: python setup_lucia.py")
        sys.exit(1)
    
    # Importar y ejecutar LucIA
    try:
        print("🚀 Iniciando LucIA...")
        from main import main as lucia_main
        asyncio.run(lucia_main())
        
    except KeyboardInterrupt:
        print("\n👋 LucIA detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando LucIA: {e}")
        print("💡 Para más información, ejecuta: python setup_lucia.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
