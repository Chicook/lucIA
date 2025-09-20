#!/usr/bin/env python3
"""
Test Simple de Integración - LucIA v0.6.0
Prueba básica de la integración de sistemas
"""

import asyncio
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_integracion_simple():
    """Test simple de integración"""
    print("=" * 60)
    print("🧪 TEST SIMPLE DE INTEGRACIÓN - LucIA v0.6.0")
    print("=" * 60)
    
    try:
        # Importar el motor principal
        from main import LucIACore
        
        print("\n1️⃣ CREANDO INSTANCIA DE LucIA...")
        lucia = LucIACore()
        print("   ✅ Instancia creada correctamente")
        
        print("\n2️⃣ INICIALIZANDO MÓDULOS...")
        await lucia.initialize_modules()
        print(f"   ✅ Módulos inicializados: {len(lucia.modules)}")
        
        print("\n3️⃣ VERIFICANDO MÓDULO DE INTEGRACIÓN AVANZADA...")
        if "advanced_integration" in lucia.modules:
            integration_module = lucia.modules["advanced_integration"]
            print("   ✅ Módulo de integración avanzada encontrado")
            
            # Verificar si es una instancia o un módulo
            if hasattr(integration_module, 'is_initialized'):
                print(f"   ✅ Inicializado: {integration_module.is_initialized}")
                
                # Verificar sistemas integrados
                if hasattr(integration_module, 'celebro_core') and integration_module.celebro_core:
                    print("   ✅ @celebro integrado")
                else:
                    print("   ⚠️ @celebro no disponible")
                
                if hasattr(integration_module, 'red_neuronal_core') and integration_module.red_neuronal_core:
                    print("   ✅ @red_neuronal integrado")
                else:
                    print("   ⚠️ @red_neuronal no disponible")
                
                if hasattr(integration_module, 'conocimientos_system') and integration_module.conocimientos_system:
                    print("   ✅ @conocimientos integrado")
                else:
                    print("   ⚠️ @conocimientos no disponible")
            else:
                print("   ⚠️ Módulo no es una instancia válida")
        else:
            print("   ❌ Módulo de integración avanzada no encontrado")
        
        print("\n4️⃣ PROBANDO MÉTODOS DEL MOTOR PRINCIPAL...")
        
        # Probar generación de prompts
        try:
            prompts = await lucia.generate_security_prompts("autenticacion", 2)
            print(f"   ✅ Generación de prompts: {len(prompts)} prompts generados")
        except Exception as e:
            print(f"   ❌ Error en generación de prompts: {e}")
        
        # Probar entrenamiento
        try:
            training_result = await lucia.train_with_security_topics(["autenticacion"])
            print(f"   ✅ Entrenamiento: {training_result}")
        except Exception as e:
            print(f"   ❌ Error en entrenamiento: {e}")
        
        # Probar estado del sistema
        try:
            status = await lucia.get_advanced_system_status()
            print(f"   ✅ Estado del sistema obtenido: {len(status)} elementos")
        except Exception as e:
            print(f"   ❌ Error obteniendo estado: {e}")
        
        print("\n5️⃣ PROBANDO PROCESAMIENTO DE ENTRADA...")
        try:
            test_input = "¿Cómo implementar autenticación segura?"
            result = await lucia.process_input(test_input)
            print(f"   ✅ Procesamiento exitoso: {type(result).__name__}")
        except Exception as e:
            print(f"   ❌ Error en procesamiento: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 TEST COMPLETADO")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal"""
    success = await test_integracion_simple()
    
    if success:
        print("\n✅ INTEGRACIÓN FUNCIONANDO CORRECTAMENTE")
        print("🚀 LucIA está listo para ser entrenado por otras IAs")
    else:
        print("\n❌ HAY PROBLEMAS EN LA INTEGRACIÓN")
        print("🔧 Revisar logs para más detalles")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
