#!/usr/bin/env python3
"""
Demostración de Integración TensorFlow + @celebro - LucIA
Versión: 0.6.0
Demuestra las capacidades de aprendizaje profundo integradas con @celebro
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_tensorflow_celebro():
    """Demuestra la integración de TensorFlow con @celebro"""
    print("=" * 80)
    print("🧠 DEMO: INTEGRACIÓN TENSORFLOW + @CELEBRO")
    print("=" * 80)
    
    try:
        from celebro.celebro_core import CelebroCore
        from celebro.tensorflow_integration import ModelType, TrainingStatus
        
        # Inicializar @celebro con TensorFlow
        print("🚀 Inicializando @celebro con TensorFlow...")
        celebro = CelebroCore()
        
        # Verificar estado del sistema
        status = celebro.get_system_status()
        print(f"✅ @celebro inicializado: {status['components_status']['tensorflow_integration']}")
        print(f"📊 Modelos TensorFlow disponibles: {status['tensorflow_models']}")
        
        # Crear modelos de diferentes tipos
        print("\n🔧 Creando modelos de TensorFlow...")
        
        # 1. Modelo de análisis de sentimientos
        sentiment_model_id = celebro.create_tensorflow_model(
            "Analizador_Sentimientos", 
            "sentiment_analysis"
        )
        print(f"   ✅ Modelo de sentimientos creado: {sentiment_model_id}")
        
        # 2. Modelo de análisis de seguridad
        security_model_id = celebro.create_tensorflow_model(
            "Analizador_Seguridad",
            "security_analysis",
            security_categories=[
                'authentication', 'encryption', 'malware', 'phishing',
                'firewall', 'vulnerability', 'compliance', 'incident_response'
            ]
        )
        print(f"   ✅ Modelo de seguridad creado: {security_model_id}")
        
        # 3. Modelo de clasificación de texto
        classification_model_id = celebro.create_tensorflow_model(
            "Clasificador_Texto",
            "text_classification",
            num_classes=5
        )
        print(f"   ✅ Modelo de clasificación creado: {classification_model_id}")
        
        # Preparar datos de entrenamiento de ejemplo
        print("\n📚 Preparando datos de entrenamiento...")
        
        # Datos de sentimientos
        sentiment_texts = [
            "Me encanta este sistema de seguridad",
            "Este código tiene vulnerabilidades graves",
            "La implementación es correcta pero básica",
            "Excelente trabajo en la encriptación",
            "Necesitamos mejorar la autenticación",
            "El firewall está bien configurado",
            "Hay un problema de seguridad crítico",
            "La documentación es muy clara",
            "Este ataque es muy sofisticado",
            "La respuesta fue rápida y efectiva"
        ]
        
        sentiment_labels = [
            "Positivo", "Negativo", "Neutral", "Positivo", "Neutral",
            "Positivo", "Negativo", "Positivo", "Negativo", "Positivo"
        ]
        
        # Datos de seguridad
        security_texts = [
            "Implementar autenticación de dos factores",
            "Usar encriptación AES-256 para datos sensibles",
            "Detectar malware con análisis de comportamiento",
            "Prevenir ataques de phishing con filtros",
            "Configurar firewall con reglas estrictas",
            "Escanear vulnerabilidades regularmente",
            "Cumplir con normativas GDPR",
            "Responder a incidentes de seguridad"
        ]
        
        security_labels = [
            "authentication", "encryption", "malware", "phishing",
            "firewall", "vulnerability", "compliance", "incident_response"
        ]
        
        # Entrenar modelos
        print("\n🎓 Entrenando modelos...")
        
        # Entrenar modelo de sentimientos
        print("   📊 Entrenando modelo de sentimientos...")
        sentiment_metrics = celebro.train_tensorflow_model(
            sentiment_model_id,
            sentiment_texts,
            sentiment_labels
        )
        
        if 'error' not in sentiment_metrics:
            print(f"      ✅ Precisión: {sentiment_metrics['accuracy']:.3f}")
            print(f"      ✅ F1-Score: {sentiment_metrics['f1_score']:.3f}")
            print(f"      ⏱️ Tiempo: {sentiment_metrics['training_time']:.2f}s")
        else:
            print(f"      ❌ Error: {sentiment_metrics['error']}")
        
        # Entrenar modelo de seguridad
        print("   🔒 Entrenando modelo de seguridad...")
        security_metrics = celebro.train_tensorflow_model(
            security_model_id,
            security_texts,
            security_labels
        )
        
        if 'error' not in security_metrics:
            print(f"      ✅ Precisión: {security_metrics['accuracy']:.3f}")
            print(f"      ✅ F1-Score: {security_metrics['f1_score']:.3f}")
            print(f"      ⏱️ Tiempo: {security_metrics['training_time']:.2f}s")
        else:
            print(f"      ❌ Error: {security_metrics['error']}")
        
        # Probar predicciones
        print("\n🔮 Probando predicciones...")
        
        test_texts = [
            "Este sistema de autenticación es excelente",
            "Hay una vulnerabilidad crítica en el código",
            "Implementar encriptación end-to-end",
            "El malware se propagó rápidamente"
        ]
        
        for text in test_texts:
            print(f"\n   📝 Texto: '{text}'")
            
            # Análisis de sentimientos
            try:
                sentiment_result = celebro.predict_with_tensorflow(sentiment_model_id, text)
                if 'error' not in sentiment_result:
                    print(f"      😊 Sentimiento: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.3f})")
            except Exception as e:
                print(f"      ❌ Error sentimientos: {e}")
            
            # Análisis de seguridad
            try:
                security_result = celebro.predict_with_tensorflow(security_model_id, text)
                if 'error' not in security_result:
                    print(f"      🔒 Categoría: {security_result['security_category']} ({security_result['confidence']:.3f})")
            except Exception as e:
                print(f"      ❌ Error seguridad: {e}")
        
        # Análisis combinado con @celebro
        print("\n🤖 Análisis combinado con @celebro...")
        
        sample_response = "Implementar autenticación multifactor y encriptación AES-256 para proteger los datos sensibles del usuario."
        
        combined_analysis = celebro.analyze_response_with_ai(sample_response, {
            'context': 'security_recommendation',
            'user_level': 'intermediate'
        })
        
        if 'error' not in combined_analysis:
            print(f"   📊 Análisis tradicional: {len(combined_analysis['traditional_analysis'])} componentes")
            print(f"   🧠 Análisis IA: {combined_analysis['models_used']} modelos utilizados")
            print(f"   ⏰ Timestamp: {combined_analysis['analysis_timestamp']}")
        else:
            print(f"   ❌ Error en análisis combinado: {combined_analysis['error']}")
        
        # Mostrar información de modelos
        print("\n📋 Información de modelos creados...")
        models_info = celebro.get_tensorflow_models()
        
        for model_info in models_info:
            print(f"   🔧 {model_info['model_name']} ({model_info['model_id']})")
            print(f"      Tipo: {model_info['model_type']}")
            print(f"      Parámetros: {model_info['total_params']:,}")
            print(f"      Creado: {model_info['created_at']}")
        
        # Estado final del sistema
        print("\n📊 Estado final del sistema...")
        final_status = celebro.get_system_status()
        print(f"   🧠 Modelos TensorFlow: {final_status['tensorflow_models']}")
        print(f"   📊 Estado entrenamiento: {final_status['tensorflow_status']}")
        print(f"   🔧 Componentes activos: {sum(final_status['components_status'].values())}/{len(final_status['components_status'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de TensorFlow + @celebro: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_advanced_features():
    """Demuestra características avanzadas de la integración"""
    print("\n" + "=" * 80)
    print("🚀 DEMO: CARACTERÍSTICAS AVANZADAS")
    print("=" * 80)
    
    try:
        from celebro.celebro_core import CelebroCore
        
        celebro = CelebroCore()
        
        # Crear un modelo de generación de respuestas
        print("🤖 Creando modelo de generación de respuestas...")
        generation_model_id = celebro.create_tensorflow_model(
            "Generador_Respuestas",
            "response_generation"
        )
        print(f"   ✅ Modelo creado: {generation_model_id}")
        
        # Simular entrenamiento con datos de respuestas
        print("📚 Preparando datos de generación...")
        response_texts = [
            "Para mejorar la seguridad, implementa autenticación de dos factores",
            "Usa encriptación AES-256 para proteger datos sensibles",
            "Configura un firewall con reglas estrictas",
            "Realiza auditorías de seguridad regularmente",
            "Mantén actualizado el software y parches de seguridad"
        ]
        
        # Para generación, las etiquetas son las mismas respuestas
        generation_labels = response_texts.copy()
        
        print("🎓 Entrenando modelo de generación...")
        generation_metrics = celebro.train_tensorflow_model(
            generation_model_id,
            response_texts,
            generation_labels
        )
        
        if 'error' not in generation_metrics:
            print(f"   ✅ Entrenamiento completado")
            print(f"   📊 Precisión: {generation_metrics['accuracy']:.3f}")
        else:
            print(f"   ❌ Error: {generation_metrics['error']}")
        
        # Probar generación de respuestas
        print("\n💬 Probando generación de respuestas...")
        test_prompts = [
            "¿Cómo mejorar la seguridad?",
            "¿Qué hacer con datos sensibles?",
            "¿Cómo configurar protección?"
        ]
        
        for prompt in test_prompts:
            print(f"\n   📝 Prompt: '{prompt}'")
            try:
                result = celebro.predict_with_tensorflow(generation_model_id, prompt)
                if 'error' not in result:
                    print(f"      🤖 Respuesta generada: {result.get('predicted_class', 'N/A')}")
                else:
                    print(f"      ❌ Error: {result['error']}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo avanzado: {e}")
        return False

async def main():
    """Función principal de demostración"""
    print("🚀 DEMOSTRACIÓN TENSORFLOW + @CELEBRO - LucIA")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    demos = [
        ("Integración TensorFlow + @celebro", demo_tensorflow_celebro),
        ("Características Avanzadas", demo_advanced_features)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 Ejecutando: {demo_name}")
        try:
            success = await demo_func()
            results.append((demo_name, success))
            if success:
                print(f"✅ {demo_name} - EXITOSO")
            else:
                print(f"❌ {demo_name} - FALLÓ")
        except Exception as e:
            print(f"❌ {demo_name} - ERROR: {e}")
            results.append((demo_name, False))
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE DEMOSTRACIONES")
    print("=" * 80)
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    for demo_name, success in results:
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"   {demo_name}: {status}")
    
    print(f"\n🎯 Resultado: {successful}/{total} demostraciones exitosas")
    
    if successful == total:
        print("🎉 ¡TODAS LAS DEMOSTRACIONES EXITOSAS!")
        print("🧠 TensorFlow está completamente integrado con @celebro")
        print("🚀 LucIA ahora tiene capacidades de aprendizaje profundo avanzadas")
    else:
        print("⚠️ Algunas demostraciones fallaron. Revisar errores arriba.")
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demostración interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
