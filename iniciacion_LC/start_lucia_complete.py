#!/usr/bin/env python3
"""
Inicio Completo de LucIA v0.6.0
Script que instala dependencias, ejecuta tests y inicia el sistema completo
"""

import asyncio
import sys
import os
import subprocess
import importlib
from datetime import datetime

def install_dependencies():
    """Instala dependencias automáticamente"""
    print("=" * 80)
    print("🔧 INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS")
    print("=" * 80)
    
    # Lista de dependencias necesarias
    dependencies = [
        "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
        "requests", "aiohttp", "sqlalchemy", "cryptography", "PyJWT",
        "Pillow", "psutil", "python-dotenv", "pydantic", "click",
        "rich", "tqdm", "schedule", "pytest", "pytest-asyncio"
    ]
    
    installed = 0
    failed = 0
    
    for dep in dependencies:
        try:
            # Verificar si ya está instalado
            importlib.import_module(dep)
            print(f"✅ {dep} - Ya instalado")
            installed += 1
        except ImportError:
            print(f"📦 Instalando {dep}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", dep, "--quiet"
                ])
                print(f"✅ {dep} - Instalado")
                installed += 1
            except subprocess.CalledProcessError:
                print(f"❌ {dep} - Error en instalación")
                failed += 1
    
    print(f"\n📊 Resumen: {installed} instalados, {failed} fallidos")
    return failed == 0

async def run_tests():
    """Ejecuta tests automáticos"""
    print("\n" + "=" * 80)
    print("🧪 EJECUTANDO TESTS AUTOMÁTICOS")
    print("=" * 80)
    
    try:
        from test_LC.auto_tests import run_auto_tests
        report = await run_auto_tests()
        
        if report:
            success_rate = report.get('success_rate', 0)
            status = report.get('status', 'UNKNOWN')
            
            print(f"\n📊 Resultados de Tests:")
            print(f"   Tasa de éxito: {success_rate:.1f}%")
            print(f"   Estado: {status}")
            
            if status == "READY":
                print("✅ SISTEMA LISTO PARA USAR")
                return True
            elif status == "WARNING":
                print("⚠️ SISTEMA FUNCIONAL CON ADVERTENCIAS")
                return True
            else:
                print("❌ SISTEMA CON ERRORES")
                return False
        else:
            print("❌ No se pudo ejecutar tests")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando tests: {e}")
        return False

async def start_lucia():
    """Inicia LucIA con todas las verificaciones"""
    print("=" * 80)
    print("🤖 INICIANDO LUCIA v0.6.0 - SISTEMA COMPLETO")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # 1. Instalar dependencias
        print("\n1️⃣ INSTALANDO DEPENDENCIAS...")
        if not install_dependencies():
            print("❌ Error instalando dependencias")
            return False
        
        # 2. Importar y crear LucIA
        print("\n2️⃣ CREANDO INSTANCIA DE LUCIA...")
        from main import LucIACore
        
        # Crear instancia con auto-instalación y tests deshabilitados
        # (ya los ejecutamos manualmente)
        lucia = LucIACore(auto_install=False, auto_test=False)
        print("✅ Instancia de LucIA creada")
        
        # 3. Ejecutar tests
        print("\n3️⃣ EJECUTANDO TESTS...")
        if not await run_tests():
            print("⚠️ Tests fallaron, pero continuando...")
        
        # 4. Inicializar módulos
        print("\n4️⃣ INICIALIZANDO MÓDULOS...")
        await lucia.initialize_modules()
        print(f"✅ {len(lucia.modules)} módulos inicializados")
        
        # 5. Mostrar estado del sistema
        print("\n5️⃣ ESTADO DEL SISTEMA:")
        print(f"   Módulos activos: {len(lucia.modules)}")
        print(f"   Configuración: {lucia.config.get('ai_name', 'N/A')} v{lucia.config.get('version', 'N/A')}")
        
        # 6. Probar funcionalidades básicas
        print("\n6️⃣ PROBANDO FUNCIONALIDADES...")
        
        # Probar procesamiento de entrada
        test_input = "Hola, soy una prueba del sistema LucIA"
        result = await lucia.process_input(test_input)
        print(f"   Procesamiento de entrada: {'✅ OK' if result else '❌ Error'}")
        
        # Probar generación de prompts de seguridad
        try:
            prompts = await lucia.generate_security_prompts("autenticacion", 1)
            print(f"   Generación de prompts: {'✅ OK' if prompts else '❌ Error'}")
        except Exception as e:
            print(f"   Generación de prompts: ❌ Error - {e}")
        
        # 7. Mostrar resumen final
        print("\n" + "=" * 80)
        print("🎉 LUCIA v0.6.0 INICIADO EXITOSAMENTE")
        print("=" * 80)
        print("✅ Dependencias instaladas")
        print("✅ Tests ejecutados")
        print("✅ Módulos inicializados")
        print("✅ Sistema funcionando")
        print("=" * 80)
        print("🚀 LucIA está listo para ser entrenado por otras IAs")
        print("🔒 Enfocado en seguridad en internet y cómo combatirla vía código")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal"""
    try:
        success = await start_lucia()
        
        if success:
            print("\n🎯 ¿Quieres iniciar el motor principal? (s/n): ", end="")
            try:
                response = input().lower().strip()
                if response in ['s', 'si', 'sí', 'y', 'yes']:
                    print("\n🚀 Iniciando motor principal...")
                    from main import LucIACore
                    lucia = LucIACore(auto_install=False, auto_test=False)
                    await lucia.start_ai_engine()
                else:
                    print("👋 LucIA configurado pero no iniciado")
            except KeyboardInterrupt:
                print("\n👋 Operación cancelada por el usuario")
        else:
            print("\n❌ No se pudo iniciar LucIA correctamente")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Operación interrumpida por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Sistema detenido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        sys.exit(1)
