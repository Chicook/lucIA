#!/usr/bin/env python3
"""
Test de Conexión con Gemini API - LucIA v0.6.0
Verifica que la conexión con Gemini funcione correctamente
"""

import asyncio
import sys
import os

async def test_gemini_connection():
    """Prueba la conexión con Gemini API"""
    print("=" * 60)
    print("🧪 TEST DE CONEXIÓN CON GEMINI API")
    print("=" * 60)
    
    try:
        # Importar la integración con Gemini
        from celebro.red_neuronal.gemini_integration import GeminiIntegration
        
        print("📦 Creando instancia de GeminiIntegration...")
        gemini = GeminiIntegration()
        
        print("🔗 Probando conexión con Gemini...")
        test_prompt = "Hola, soy LucIA. ¿Puedes confirmar que la conexión funciona correctamente?"
        
        result = gemini.generate_text(test_prompt)
        response = result.get('text', '') if isinstance(result, dict) else str(result)
        
        if response and len(response.strip()) > 0:
            print("✅ Conexión con Gemini EXITOSA")
            print(f"📝 Respuesta de Gemini: {response}")
            return True
        else:
            print("❌ Conexión con Gemini FALLÓ - Sin respuesta")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando GeminiIntegration: {e}")
        return False
    except Exception as e:
        print(f"❌ Error conectando con Gemini: {e}")
        return False

async def test_gemini_with_security_prompt():
    """Prueba Gemini con un prompt de ciberseguridad"""
    print("\n" + "=" * 60)
    print("🔒 TEST CON PROMPT DE CIBERSEGURIDAD")
    print("=" * 60)
    
    try:
        from celebro.red_neuronal.gemini_integration import GeminiIntegration
        
        gemini = GeminiIntegration()
        
        security_prompt = """Eres LucIA, un asistente especializado en ciberseguridad. 
        Explica brevemente qué es la autenticación de dos factores (2FA) y por qué es importante."""
        
        print("🔒 Enviando prompt de ciberseguridad...")
        result = gemini.generate_text(security_prompt)
        response = result.get('text', '') if isinstance(result, dict) else str(result)
        
        if response and len(response.strip()) > 0:
            print("✅ Respuesta de ciberseguridad EXITOSA")
            print(f"📝 Respuesta: {response}")
            return True
        else:
            print("❌ Sin respuesta de ciberseguridad")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de ciberseguridad: {e}")
        return False

async def main():
    """Función principal de prueba"""
    print("🚀 Iniciando tests de conexión con Gemini...")
    
    # Test 1: Conexión básica
    test1_success = await test_gemini_connection()
    
    # Test 2: Prompt de ciberseguridad
    test2_success = await test_gemini_with_security_prompt()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS")
    print("=" * 60)
    print(f"✅ Conexión básica: {'EXITOSA' if test1_success else 'FALLÓ'}")
    print(f"✅ Prompt ciberseguridad: {'EXITOSA' if test2_success else 'FALLÓ'}")
    
    if test1_success and test2_success:
        print("\n🎉 ¡TODOS LOS TESTS EXITOSOS! Gemini está funcionando correctamente.")
        return True
    else:
        print("\n❌ Algunos tests fallaron. Revisar configuración de Gemini.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests interrumpidos por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
