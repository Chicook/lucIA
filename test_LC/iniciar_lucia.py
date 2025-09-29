#!/usr/bin/env python3
"""
Iniciador Simple de LucIA v0.6.0
Ejecuta LucIA con chat interactivo automático
"""

import asyncio
import sys
import os

def main():
    """Función principal que ejecuta LucIA"""
    try:
        print("🚀 Iniciando LucIA v0.6.0...")
        print("=" * 50)
        
        # Importar y ejecutar el main
        from main import main as lucia_main
        asyncio.run(lucia_main())
        
    except KeyboardInterrupt:
        print("\n👋 LucIA detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error iniciando LucIA: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
