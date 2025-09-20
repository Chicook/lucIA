#!/usr/bin/env python3
"""
Test Simple de Gemini API
Versión: 0.6.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """Prueba la importación de configuración"""
    print("🔧 Probando importación de configuración...")
    
    try:
        from celebro.red_neuronal.config_simple import get_gemini_api_key, get_gemini_config
        print("✅ Importación de configuración exitosa")
        
        # Probar funciones
        api_key = get_gemini_api_key()
        config = get_gemini_config()
        
        print(f"   API Key: {api_key[:10]}...")
        print(f"   Modelo: {config['model']}")
        print(f"   Max Tokens: {config['max_tokens']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en importación: {e}")
        return False

def test_gemini_connection():
    """Prueba la conexión con Gemini"""
    print("\n🔗 Probando conexión con Gemini...")
    
    try:
        from celebro.red_neuronal.gemini_integration import GeminiIntegration
        
        # Crear instancia
        gemini = GeminiIntegration()
        
        # Probar conexión
        result = gemini.test_connection()
        
        if result['success']:
            print("✅ Conexión exitosa con Gemini API")
            print(f"   Respuesta: {result['response']}")
        else:
            print("❌ Error de conexión")
            print(f"   Error: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")
        return False

def test_network_analysis():
    """Prueba el análisis de red neuronal"""
    print("\n🧠 Probando análisis de red neuronal...")
    
    try:
        from celebro.red_neuronal.gemini_integration import analyze_network
        
        # Configuración de ejemplo
        network_config = {
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'dropout_rate': 0.2
        }
        
        print(f"📊 Analizando configuración: {network_config}")
        
        # Analizar configuración
        result = analyze_network(network_config)
        
        if result['success']:
            print("✅ Análisis completado")
            print(f"\n📋 Análisis de Gemini:")
            print("-" * 40)
            print(result['analysis'])
            print("-" * 40)
        else:
            print("❌ Error en análisis")
            print(f"   Error: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return False

def main():
    """Función principal"""
    print("🧠 TEST SIMPLE DE GEMINI API")
    print("=" * 50)
    
    # Ejecutar pruebas
    tests = [
        ("Importación de Configuración", test_config_import),
        ("Conexión con Gemini", test_gemini_connection),
        ("Análisis de Red Neuronal", test_network_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🚀 Ejecutando: {test_name}")
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name} - EXITOSO")
            else:
                print(f"❌ {test_name} - FALLÓ")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results[test_name] = False
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    successful_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    for test_name, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Resultado general: {successful_tests}/{total_tests} pruebas exitosas")
    print(f"📈 Tasa de éxito: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ¡Todas las pruebas pasaron! La integración con Gemini está funcionando.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa la configuración.")

if __name__ == "__main__":
    main()
