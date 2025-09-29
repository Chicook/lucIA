#!/usr/bin/env python3
"""
Demostración del Sistema de Aprendizaje Profundo - @red_neuronal
Versión: 0.6.0
Demuestra las capacidades avanzadas de análisis y aprendizaje profundo
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_query_analyzer():
    """Demuestra el analizador profundo de consultas"""
    print("=" * 80)
    print("🔍 DEMO: ANALIZADOR PROFUNDO DE CONSULTAS")
    print("=" * 80)
    
    try:
        from celebro.red_neuronal.query_analyzer import DeepQueryAnalyzer
        
        analyzer = DeepQueryAnalyzer()
        
        # Consultas de ejemplo
        test_queries = [
            "¿Qué es la autenticación de dos factores?",
            "¿Cómo implementar encriptación AES-256 en Python?",
            "¿Cuáles son las mejores prácticas para prevenir ataques de phishing?",
            "Explícame la arquitectura de un firewall de próxima generación",
            "¿Cómo realizar análisis forense de malware en un sistema comprometido?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Consulta {i}: {query}")
            print("-" * 60)
            
            analysis = analyzer.analyze_query(query)
            
            print(f"🔍 Análisis:")
            print(f"   ID: {analysis.query_id}")
            print(f"   Complejidad: {analysis.complexity.value}")
            print(f"   Categoría: {analysis.category.value}")
            print(f"   Palabras clave: {', '.join(analysis.keywords[:5])}")
            print(f"   Entidades: {', '.join(analysis.entities[:3])}")
            print(f"   Intención: {analysis.intent}")
            print(f"   Potencial de aprendizaje: {analysis.learning_potential:.2f}")
            print(f"   Prompts sugeridos: {len(analysis.suggested_prompts)}")
            
            if analysis.suggested_prompts:
                print(f"   Primer prompt: {analysis.suggested_prompts[0][:100]}...")
        
        # Obtener insights
        print(f"\n📊 INSIGHTS DEL SISTEMA:")
        insights = analyzer.get_learning_insights()
        print(f"   Total de consultas: {insights.get('total_queries', 0)}")
        print(f"   Distribución de complejidad: {insights.get('complexity_distribution', {})}")
        print(f"   Distribución de categorías: {insights.get('category_distribution', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo del analizador: {e}")
        return False

async def demo_deep_learning_engine():
    """Demuestra el motor de aprendizaje profundo"""
    print("\n" + "=" * 80)
    print("🧠 DEMO: MOTOR DE APRENDIZAJE PROFUNDO")
    print("=" * 80)
    
    try:
        from celebro.red_neuronal.deep_learning_engine import DeepLearningEngine
        
        engine = DeepLearningEngine()
        
        # Iniciar sesión de aprendizaje
        print("🚀 Iniciando sesión de aprendizaje profundo...")
        session_id = await engine.start_learning_session(
            "¿Cómo implementar autenticación segura en una aplicación web?",
            ["Dominar autenticación web", "Implementar OAuth 2.0", "Entender vulnerabilidades comunes"]
        )
        
        print(f"✅ Sesión iniciada: {session_id}")
        
        # Procesar consultas adicionales
        follow_up_queries = [
            "¿Cuál es la diferencia entre JWT y sesiones tradicionales?",
            "¿Cómo prevenir ataques de session hijacking?",
            "¿Qué es OAuth 2.0 y cómo implementarlo?",
            "¿Cuáles son las vulnerabilidades más comunes en autenticación?"
        ]
        
        for i, query in enumerate(follow_up_queries, 1):
            print(f"\n📝 Procesando consulta {i}: {query}")
            
            result = await engine.process_query(session_id, query)
            
            if 'query_analysis' in result:
                analysis = result['query_analysis']
                if hasattr(analysis, 'complexity'):
                    print(f"   Complejidad: {analysis.complexity.value}")
                    print(f"   Categoría: {analysis.category.value}")
                    print(f"   Potencial: {analysis.learning_potential:.2f}")
                else:
                    print(f"   Análisis: {type(analysis)}")
            
            if 'insights' in result and result['insights']:
                print(f"   Insights generados: {len(result['insights'])}")
                for insight in result['insights'][:2]:  # Mostrar primeros 2
                    print(f"     - {insight['type']}: {insight['content'][:80]}...")
        
        # Obtener analytics finales
        print(f"\n📊 ANALYTICS DE LA SESIÓN:")
        analytics = engine.get_learning_analytics()
        print(f"   Sesiones activas: {analytics.get('active_sessions', 0)}")
        print(f"   Sesiones completadas: {analytics.get('completed_sessions', 0)}")
        print(f"   Total de insights: {analytics.get('total_insights', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo del motor de aprendizaje: {e}")
        return False

async def demo_neural_core_integration():
    """Demuestra la integración completa con NeuralCore"""
    print("\n" + "=" * 80)
    print("🔗 DEMO: INTEGRACIÓN COMPLETA CON NEURAL CORE")
    print("=" * 80)
    
    try:
        from celebro.red_neuronal.neural_core import NeuralCore
        
        neural_core = NeuralCore()
        
        # Test de análisis profundo
        print("🔍 Probando análisis profundo de consultas...")
        query = "¿Cómo implementar un sistema de detección de intrusos usando machine learning?"
        
        analysis = await neural_core.analyze_query_deep(query)
        print(f"   Análisis completado: {analysis['complexity']} - {analysis['category']}")
        
        # Test de prompt adaptativo
        print("\n🎯 Generando prompt adaptativo...")
        adaptive_prompt = await neural_core.generate_adaptive_prompt(query)
        print(f"   Prompt generado: {len(adaptive_prompt)} caracteres")
        print(f"   Preview: {adaptive_prompt[:200]}...")
        
        # Test de sesión de aprendizaje profundo
        print("\n🧠 Iniciando sesión de aprendizaje profundo...")
        session_id = await neural_core.start_deep_learning_session(
            "¿Cuáles son las mejores prácticas de seguridad en desarrollo de software?",
            ["Desarrollo seguro", "OWASP Top 10", "DevSecOps"]
        )
        print(f"   Sesión iniciada: {session_id}")
        
        # Procesar consulta en la sesión
        print("\n📝 Procesando consulta en sesión...")
        result = await neural_core.process_deep_learning_query(
            session_id, 
            "¿Cómo implementar validación de entrada segura?"
        )
        print(f"   Consulta procesada exitosamente")
        
        # Obtener insights
        print("\n📊 Obteniendo insights de aprendizaje...")
        insights = await neural_core.get_learning_insights()
        print(f"   Insights obtenidos: {len(insights.get('combined_insights', {}))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de integración: {e}")
        return False

async def demo_gemini_integration():
    """Demuestra la integración con Gemini"""
    print("\n" + "=" * 80)
    print("🤖 DEMO: INTEGRACIÓN CON GEMINI")
    print("=" * 80)
    
    try:
        from celebro.red_neuronal.gemini_integration import GeminiIntegration
        
        gemini = GeminiIntegration()
        
        # Test básico
        print("🔗 Probando conexión con Gemini...")
        test_prompt = "Explica brevemente qué es la ciberseguridad y por qué es importante."
        
        result = gemini.generate_text(test_prompt)
        response = result.get('text', '') if isinstance(result, dict) else str(result)
        
        if response:
            print("✅ Conexión exitosa con Gemini")
            print(f"📝 Respuesta: {response[:200]}...")
        else:
            print("❌ No se recibió respuesta de Gemini")
            return False
        
        # Test con prompt especializado
        print("\n🎯 Probando prompt especializado...")
        specialized_prompt = """Eres LucIA, un asistente especializado en ciberseguridad.
        
        Analiza esta consulta: ¿Cómo implementar autenticación de dos factores en una aplicación web?
        
        Proporciona:
        1. Explicación técnica clara
        2. Ejemplos de implementación
        3. Mejores prácticas
        4. Consideraciones de seguridad
        
        Responde en español de manera técnica pero accesible."""
        
        result2 = gemini.generate_text(specialized_prompt)
        response2 = result2.get('text', '') if isinstance(result2, dict) else str(result2)
        
        if response2:
            print("✅ Prompt especializado procesado")
            print(f"📝 Respuesta especializada: {response2[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de Gemini: {e}")
        return False

async def main():
    """Función principal de demostración"""
    print("🚀 DEMOSTRACIÓN DEL SISTEMA DE APRENDIZAJE PROFUNDO - @red_neuronal")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    demos = [
        ("Analizador Profundo de Consultas", demo_query_analyzer),
        ("Motor de Aprendizaje Profundo", demo_deep_learning_engine),
        ("Integración con Neural Core", demo_neural_core_integration),
        ("Integración con Gemini", demo_gemini_integration)
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
        print("🧠 El sistema de aprendizaje profundo está funcionando correctamente")
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
