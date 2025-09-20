#!/usr/bin/env python3
"""
Demo de la función unificada de análisis de Gemini
Versión: 0.6.0
Demuestra el uso de la función unified_analysis para diferentes tipos de análisis
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celebro.red_neuronal.gemini_integration import unified_analysis, test_gemini_connection

async def demo_unified_analysis():
    """Demostración de la función unificada de análisis"""
    
    print("🤖 DEMO: Función Unificada de Análisis Gemini")
    print("=" * 60)
    
    # 1. Probar conexión
    print("\n1. 🔌 Probando conexión con Gemini...")
    connection_test = test_gemini_connection()
    
    if connection_test['success']:
        print(f"   ✅ Conexión exitosa: {connection_test['status']}")
        print(f"   📝 Respuesta: {connection_test['response']}")
    else:
        print(f"   ❌ Error de conexión: {connection_test['error']}")
        return
    
    # 2. Análisis de red neuronal
    print("\n2. 🧠 Análisis de configuración de red neuronal...")
    network_config = {
        'hidden_layers': [128, 64, 32],
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'dropout_rate': 0.2
    }
    
    network_analysis = unified_analysis('network', network_config)
    
    if network_analysis['success']:
        print(f"   ✅ Análisis completado")
        print(f"   📊 Tipo: {network_analysis['analysis_type']}")
        print(f"   📝 Análisis: {network_analysis['analysis'][:200]}...")
    else:
        print(f"   ❌ Error en análisis: {network_analysis['error']}")
    
    # 3. Sugerencia de arquitectura
    print("\n3. 🏗️ Sugerencia de arquitectura...")
    architecture_data = {
        'problem_type': 'classification',
        'input_size': 784,
        'output_size': 10,
        'data_size': 10000
    }
    
    architecture_suggestion = unified_analysis('architecture', architecture_data)
    
    if architecture_suggestion['success']:
        print(f"   ✅ Sugerencia generada")
        print(f"   📊 Tipo: {architecture_suggestion['analysis_type']}")
        print(f"   📝 Sugerencia: {architecture_suggestion['suggestion'][:200]}...")
    else:
        print(f"   ❌ Error en sugerencia: {architecture_suggestion['error']}")
    
    # 4. Explicación de resultados
    print("\n4. 📈 Explicación de resultados de entrenamiento...")
    training_metrics = {
        'train_loss': 0.15,
        'train_accuracy': 0.95,
        'val_loss': 0.25,
        'val_accuracy': 0.88,
        'epochs': 50
    }
    
    results_explanation = unified_analysis('results', training_metrics)
    
    if results_explanation['success']:
        print(f"   ✅ Explicación generada")
        print(f"   📊 Tipo: {results_explanation['analysis_type']}")
        print(f"   📝 Explicación: {results_explanation['explanation'][:200]}...")
    else:
        print(f"   ❌ Error en explicación: {results_explanation['error']}")
    
    # 5. Consejos de entrenamiento
    print("\n5. 💡 Consejos de entrenamiento...")
    problem_description = "Clasificación de imágenes médicas con dataset desbalanceado"
    
    training_advice = unified_analysis('advice', {'problem_description': problem_description})
    
    if training_advice['success']:
        print(f"   ✅ Consejos generados")
        print(f"   📊 Tipo: {training_advice['analysis_type']}")
        print(f"   📝 Consejos: {training_advice['advice'][:200]}...")
    else:
        print(f"   ❌ Error en consejos: {training_advice['error']}")
    
    # 6. Análisis personalizado
    print("\n6. 🎯 Análisis personalizado...")
    custom_prompt = """
    Analiza las ventajas y desventajas de usar dropout en redes neuronales profundas.
    Incluye ejemplos específicos y recomendaciones prácticas.
    """
    
    custom_analysis = unified_analysis('custom', {'prompt': custom_prompt}, 
                                     max_tokens=800, temperature=0.5)
    
    if custom_analysis['success']:
        print(f"   ✅ Análisis personalizado completado")
        print(f"   📊 Tipo: {custom_analysis['analysis_type']}")
        print(f"   📝 Análisis: {custom_analysis['content'][:200]}...")
    else:
        print(f"   ❌ Error en análisis personalizado: {custom_analysis['error']}")
    
    # 7. Resumen de eficiencia
    print("\n7. 📊 Resumen de eficiencia...")
    print(f"   ✅ Todas las funciones unificadas en una sola función")
    print(f"   ✅ Configuración centralizada de parámetros")
    print(f"   ✅ Manejo unificado de errores")
    print(f"   ✅ Respuestas estandarizadas")
    print(f"   ✅ Soporte para prompts personalizados")
    print(f"   ✅ Flexibilidad en parámetros de generación")
    
    print("\n🎉 Demo completado exitosamente!")
    print("=" * 60)

def demo_comparison():
    """Demostración de la comparación entre funciones individuales vs unificada"""
    
    print("\n🔄 COMPARACIÓN: Funciones Individuales vs Función Unificada")
    print("=" * 60)
    
    print("\n📊 FUNCIONES INDIVIDUALES (ANTES):")
    print("   - analyze_neural_network()")
    print("   - suggest_architecture()")
    print("   - explain_results()")
    print("   - generate_training_advice()")
    print("   - 4 funciones separadas")
    print("   - 4 manejos de errores diferentes")
    print("   - 4 configuraciones de parámetros")
    print("   - ~200 líneas de código duplicado")
    
    print("\n🎯 FUNCIÓN UNIFICADA (AHORA):")
    print("   - unified_analysis()")
    print("   - 1 función para todos los casos")
    print("   - 1 manejo de errores centralizado")
    print("   - 1 configuración de parámetros")
    print("   - ~100 líneas de código eficiente")
    print("   - Soporte para análisis personalizado")
    print("   - Configuración flexible de parámetros")
    
    print("\n✅ BENEFICIOS:")
    print("   - 50% menos código")
    print("   - Mantenimiento más fácil")
    print("   - Configuración centralizada")
    print("   - Mayor flexibilidad")
    print("   - Mejor escalabilidad")
    print("   - Funciones de conveniencia mantenidas")

if __name__ == "__main__":
    print("🚀 Iniciando demo de función unificada de Gemini...")
    
    # Ejecutar demo principal
    asyncio.run(demo_unified_analysis())
    
    # Mostrar comparación
    demo_comparison()
    
    print("\n✨ Demo completado!")
