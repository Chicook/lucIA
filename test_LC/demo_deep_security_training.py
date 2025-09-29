#!/usr/bin/env python3
"""
Demostración de Entrenamiento Profundo de Seguridad - LucIA
Versión: 0.6.0
Demuestra el entrenamiento automático con Gemini + TensorFlow sobre temas de seguridad
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_deep_security_training():
    """Demuestra el entrenamiento profundo de seguridad"""
    print("=" * 80)
    print("🔒 DEMO: ENTRENAMIENTO PROFUNDO DE SEGURIDAD")
    print("=" * 80)
    
    try:
        from celebro.deep_security_training import DeepSecurityTrainer
        
        # Inicializar entrenador
        print("🚀 Inicializando entrenador de seguridad profunda...")
        trainer = DeepSecurityTrainer()
        
        # Verificar conexión con Gemini
        print("🔗 Verificando conexión con Gemini...")
        gemini_status = trainer.gemini.test_connection()
        if gemini_status:
            print("✅ Gemini conectado correctamente")
        else:
            print("❌ Error conectando con Gemini")
            return False
        
        # Mostrar estado inicial
        print("\n📊 Estado inicial del entrenamiento...")
        initial_status = await trainer.get_training_status()
        print(f"   📚 Datos de entrenamiento: {initial_status['total_training_data']}")
        print(f"   🧠 Modelos TensorFlow: {initial_status['tensorflow_models']}")
        print(f"   🔗 Gemini conectado: {initial_status['gemini_connected']}")
        print(f"   📋 Temas disponibles: {len(initial_status['available_topics'])}")
        
        # Iniciar sesión de entrenamiento
        print("\n🎯 Iniciando sesión de entrenamiento...")
        session_id = await trainer.start_training_session("Demo Security Training")
        print(f"   ✅ Sesión iniciada: {session_id}")
        
        # Entrenar en temas específicos de seguridad
        security_topics = [
            "authentication",
            "encryption", 
            "malware",
            "web_security"
        ]
        
        print(f"\n🔒 Entrenando en {len(security_topics)} temas de seguridad...")
        
        for i, topic in enumerate(security_topics, 1):
            print(f"\n   📚 Tema {i}/{len(security_topics)}: {topic}")
            
            try:
                # Entrenar en el tema con diferentes niveles de complejidad
                result = await trainer.train_on_security_topic(
                    topic=topic,
                    complexity_levels=[1, 2, 3],  # Niveles básico, intermedio, avanzado
                    questions_per_level=2  # 2 preguntas por nivel
                )
                
                if 'error' not in result:
                    print(f"      ✅ Preguntas generadas: {result['questions_generated']}")
                    print(f"      ✅ Respuestas obtenidas: {result['responses_obtained']}")
                    print(f"      ✅ Modelos creados: {result['models_created']}")
                    print(f"      📊 Precisión categorías: {result['category_accuracy']:.3f}")
                    print(f"      📊 Precisión sentimientos: {result['sentiment_accuracy']:.3f}")
                    print(f"      📊 Precisión complejidad: {result['complexity_accuracy']:.3f}")
                    print(f"      🎯 Precisión promedio: {result['average_accuracy']:.3f}")
                    print(f"      ⭐ Calidad: {result['training_quality']}")
                else:
                    print(f"      ❌ Error: {result['error']}")
                    
            except Exception as e:
                print(f"      ❌ Error entrenando {topic}: {e}")
        
        # Mostrar estado final
        print("\n📊 Estado final del entrenamiento...")
        final_status = await trainer.get_training_status()
        
        print(f"   📚 Datos de entrenamiento: {final_status['total_training_data']}")
        print(f"   🧠 Modelos TensorFlow: {final_status['tensorflow_models']}")
        print(f"   📋 Temas cubiertos: {len(final_status['current_session']['topics_covered'])}")
        print(f"   ❓ Preguntas realizadas: {final_status['current_session']['questions_asked']}")
        print(f"   🤖 Respuestas generadas: {final_status['current_session']['responses_generated']}")
        print(f"   🔧 Modelos entrenados: {len(final_status['current_session']['models_trained'])}")
        print(f"   📈 Mejora de precisión: {final_status['current_session']['accuracy_improvement']:.3f}")
        
        # Probar predicciones con los modelos entrenados
        print("\n🔮 Probando predicciones con modelos entrenados...")
        
        test_questions = [
            "¿Cómo implementar autenticación de dos factores?",
            "¿Qué es el cifrado AES-256?",
            "¿Cómo detectar malware en un sistema?",
            "¿Cuáles son las vulnerabilidades web más comunes?"
        ]
        
        for question in test_questions:
            print(f"\n   📝 Pregunta: '{question}'")
            
            # Obtener respuesta de Gemini
            try:
                gemini_response = await trainer.get_gemini_security_response(question)
                print(f"      🤖 Respuesta Gemini: {gemini_response['text'][:100]}...")
                print(f"      🔒 Categoría: {gemini_response['security_category']}")
                print(f"      📊 Complejidad: {gemini_response['complexity_level']}/5")
                print(f"      🎯 Confianza: {gemini_response['confidence_score']:.3f}")
                print(f"      ⭐ Calidad: {gemini_response['quality']}")
            except Exception as e:
                print(f"      ❌ Error con Gemini: {e}")
        
        # Guardar datos de entrenamiento
        print("\n💾 Guardando datos de entrenamiento...")
        try:
            saved_file = trainer.save_training_data()
            print(f"   ✅ Datos guardados en: {saved_file}")
        except Exception as e:
            print(f"   ❌ Error guardando datos: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de entrenamiento profundo: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_comprehensive_training():
    """Demuestra el entrenamiento comprensivo en todos los temas"""
    print("\n" + "=" * 80)
    print("🌐 DEMO: ENTRENAMIENTO COMPRENSIVO")
    print("=" * 80)
    
    try:
        from celebro.deep_security_training import DeepSecurityTrainer
        
        trainer = DeepSecurityTrainer()
        
        # Seleccionar temas para entrenamiento comprensivo
        selected_topics = [
            "authentication",
            "encryption",
            "malware",
            "phishing",
            "web_security"
        ]
        
        print(f"🚀 Iniciando entrenamiento comprensivo en {len(selected_topics)} temas...")
        print(f"   Temas: {', '.join(selected_topics)}")
        
        # Ejecutar entrenamiento comprensivo
        result = await trainer.comprehensive_security_training(selected_topics)
        
        if 'error' not in result:
            print(f"\n✅ Entrenamiento comprensivo completado!")
            print(f"   📚 Temas procesados: {result['topics_processed']}")
            print(f"   ✅ Temas exitosos: {result['successful_topics']}")
            print(f"   ❌ Temas fallidos: {result['failed_topics']}")
            print(f"   ❓ Total preguntas: {result['total_questions']}")
            print(f"   🤖 Total respuestas: {result['total_responses']}")
            print(f"   🔧 Total modelos: {result['total_models']}")
            print(f"   📊 Precisión promedio: {result['average_accuracy']:.3f}")
            print(f"   ⭐ Calidad general: {result['training_quality']}")
            
            # Mostrar resultados por tema
            print(f"\n📋 Resultados por tema:")
            for topic_result in result['results']:
                print(f"   🔒 {topic_result['topic']}: {topic_result['average_accuracy']:.3f} ({topic_result['training_quality']})")
            
        else:
            print(f"❌ Error en entrenamiento comprensivo: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo comprensivo: {e}")
        return False

async def demo_advanced_security_analysis():
    """Demuestra análisis avanzado de seguridad"""
    print("\n" + "=" * 80)
    print("🔍 DEMO: ANÁLISIS AVANZADO DE SEGURIDAD")
    print("=" * 80)
    
    try:
        from celebro.deep_security_training import DeepSecurityTrainer
        
        trainer = DeepSecurityTrainer()
        
        # Preguntas complejas de seguridad
        complex_questions = [
            "¿Cómo implementar un sistema de autenticación multifactor robusto para una empresa con 10,000 empleados?",
            "¿Cuáles son las mejores prácticas para cifrar datos en tránsito y en reposo en un entorno cloud híbrido?",
            "¿Cómo detectar y responder a un ataque de ransomware avanzado en tiempo real?",
            "¿Qué estrategias de seguridad implementar para proteger una aplicación web contra OWASP Top 10?",
            "¿Cómo diseñar un programa de concienciación en ciberseguridad efectivo para empleados?"
        ]
        
        print(f"🔍 Analizando {len(complex_questions)} preguntas complejas de seguridad...")
        
        for i, question in enumerate(complex_questions, 1):
            print(f"\n   📝 Pregunta {i}: {question[:60]}...")
            
            try:
                # Obtener respuesta detallada de Gemini
                response = await trainer.get_gemini_security_response(question)
                
                print(f"      🔒 Categoría: {response['security_category']}")
                print(f"      📊 Complejidad: {response['complexity_level']}/5")
                print(f"      🎯 Confianza: {response['confidence_score']:.3f}")
                print(f"      ⭐ Calidad: {response['quality']}")
                print(f"      📝 Respuesta: {response['text'][:150]}...")
                
            except Exception as e:
                print(f"      ❌ Error analizando pregunta: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de análisis avanzado: {e}")
        return False

async def main():
    """Función principal de demostración"""
    print("🚀 DEMOSTRACIÓN ENTRENAMIENTO PROFUNDO DE SEGURIDAD - LucIA")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    demos = [
        ("Entrenamiento Profundo de Seguridad", demo_deep_security_training),
        ("Entrenamiento Comprensivo", demo_comprehensive_training),
        ("Análisis Avanzado de Seguridad", demo_advanced_security_analysis)
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
        print("🔒 LucIA ahora está entrenada en profundidad sobre seguridad")
        print("🧠 TensorFlow + Gemini + @celebro = Sistema de IA de seguridad avanzado")
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
