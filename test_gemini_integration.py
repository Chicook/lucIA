#!/usr/bin/env python3
"""
Test de Integración con Gemini API
Versión: 0.6.0
Prueba la integración con Google Gemini API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celebro.red_neuronal import GeminiIntegration, test_gemini_connection, analyze_network, suggest_architecture

def test_gemini_connection():
    """Prueba la conexión con Gemini API"""
    print("=" * 60)
    print("🔗 PROBANDO CONEXIÓN CON GEMINI API")
    print("=" * 60)
    
    try:
        # Probar conexión
        result = test_gemini_connection()
        
        if result['success']:
            print("✅ Conexión exitosa con Gemini API")
            print(f"   Respuesta: {result['response']}")
            print(f"   Estado: {result['status']}")
        else:
            print("❌ Error de conexión")
            print(f"   Error: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")
        return False

def test_network_analysis():
    """Prueba el análisis de red neuronal"""
    print("\n" + "=" * 60)
    print("🧠 PROBANDO ANÁLISIS DE RED NEURONAL")
    print("=" * 60)
    
    try:
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

def test_architecture_suggestion():
    """Prueba la sugerencia de arquitectura"""
    print("\n" + "=" * 60)
    print("🏗️ PROBANDO SUGERENCIA DE ARQUITECTURA")
    print("=" * 60)
    
    try:
        # Parámetros de ejemplo
        problem_type = "classification"
        input_size = 784
        output_size = 10
        data_size = 10000
        
        print(f"📊 Sugiriendo arquitectura para:")
        print(f"   Tipo de problema: {problem_type}")
        print(f"   Tamaño de entrada: {input_size}")
        print(f"   Tamaño de salida: {output_size}")
        print(f"   Tamaño del dataset: {data_size}")
        
        # Sugerir arquitectura
        result = suggest_architecture(problem_type, input_size, output_size, data_size)
        
        if result['success']:
            print("✅ Sugerencia completada")
            print(f"\n🏗️ Sugerencia de Gemini:")
            print("-" * 40)
            print(result['suggestion'])
            print("-" * 40)
        else:
            print("❌ Error en sugerencia")
            print(f"   Error: {result['error']}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error en sugerencia: {e}")
        return False

def test_gemini_integration():
    """Prueba completa de la integración con Gemini"""
    print("🧠 TEST COMPLETO DE INTEGRACIÓN CON GEMINI API")
    print("=" * 80)
    
    # Ejecutar todas las pruebas
    tests = [
        ("Conexión con Gemini", test_gemini_connection),
        ("Análisis de Red Neuronal", test_network_analysis),
        ("Sugerencia de Arquitectura", test_architecture_suggestion)
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
    
    # Resumen de resultados
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    successful_tests = sum(1 for success in results.values() if success)
    total_tests = len(results)
    
    for test_name, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Resultado general: {successful_tests}/{total_tests} pruebas exitosas")
    print(f"📈 Tasa de éxito: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ¡Todas las pruebas pasaron! La integración con Gemini está funcionando correctamente.")
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa la configuración y la conexión.")
    
    return results

def main():
    """Función principal"""
    try:
        print("🚀 Iniciando test de integración con Gemini API...")
        results = test_gemini_integration()
        
        # Preguntar si quiere ver detalles específicos
        print("\n" + "=" * 60)
        print("¿Quieres ver detalles específicos de alguna prueba?")
        print("1. Conexión con Gemini")
        print("2. Análisis de Red Neuronal")
        print("3. Sugerencia de Arquitectura")
        print("0. Salir")
        
        try:
            choice = input("\nSelecciona una opción (0-3): ").strip()
            
            if choice == "1":
                test_gemini_connection()
            elif choice == "2":
                test_network_analysis()
            elif choice == "3":
                test_architecture_suggestion()
            elif choice == "0":
                print("👋 ¡Hasta luego!")
            else:
                print("❌ Opción no válida")
        
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
    
    except Exception as e:
        print(f"❌ Error en test principal: {e}")

if __name__ == "__main__":
    main()
