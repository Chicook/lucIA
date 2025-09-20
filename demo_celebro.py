#!/usr/bin/env python3
"""
Demostración de @celebro - Sistema de Interpretación de Respuestas de IAs
Versión: 0.6.0
"""

import asyncio
import json
import logging
from datetime import datetime
from celebro import CelebroCore

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Demo_Celebro')

class CelebroDemo:
    """Demostración de @celebro"""
    
    def __init__(self):
        self.celebro = CelebroCore()
        self.demo_responses = self._load_demo_responses()
    
    def _load_demo_responses(self):
        """Carga respuestas de demostración de diferentes IAs"""
        return [
            {
                "response": "La inteligencia artificial es una tecnología que permite a las máquinas realizar tareas que normalmente requieren inteligencia humana, como el reconocimiento de patrones, el aprendizaje y la toma de decisiones.",
                "source_ai": "GPT-4",
                "context": {"topic": "definicion_ia", "user_level": "beginner"}
            },
            {
                "response": "La IA funciona mediante algoritmos de machine learning que procesan grandes cantidades de datos para identificar patrones y hacer predicciones. Los modelos de deep learning, especialmente las redes neuronales, son fundamentales en este proceso.",
                "source_ai": "Claude-3",
                "context": {"topic": "funcionamiento_ia", "user_level": "intermediate"}
            },
            {
                "response": "¡Es increíble cómo la IA está transformando el mundo! Desde asistentes virtuales hasta vehículos autónomos, estamos viviendo una revolución tecnológica que cambiará para siempre la forma en que trabajamos y vivimos.",
                "source_ai": "Gemini-Pro",
                "context": {"topic": "impacto_ia", "user_level": "beginner"}
            },
            {
                "response": "Los algoritmos de optimización en IA incluyen descenso de gradiente, algoritmos genéticos, y métodos de optimización bayesiana. Cada uno tiene sus ventajas computacionales y casos de uso específicos.",
                "source_ai": "GPT-4",
                "context": {"topic": "algoritmos_ia", "user_level": "expert"}
            },
            {
                "response": "La ética en IA es crucial. Debemos considerar el sesgo algorítmico, la privacidad de datos, y el impacto social de estas tecnologías para asegurar un desarrollo responsable.",
                "source_ai": "Claude-3",
                "context": {"topic": "etica_ia", "user_level": "intermediate"}
            }
        ]
    
    async def run_demo(self):
        """Ejecuta la demostración completa"""
        try:
            print("=" * 80)
            print("🧠 @celebro - Sistema de Interpretación de Respuestas de IAs")
            print("=" * 80)
            print()
            
            # Inicializar @celebro
            print("🔄 Inicializando @celebro...")
            if not await self.celebro.initialize():
                print("❌ Error inicializando @celebro")
                return
            
            print("✅ @celebro inicializado correctamente")
            print()
            
            # Procesar respuestas de demostración
            print("📥 Procesando respuestas de IAs externas...")
            print("-" * 60)
            
            session_id = f"demo_session_{int(datetime.now().timestamp())}"
            
            for i, demo_data in enumerate(self.demo_responses, 1):
                print(f"\n🔍 Procesando respuesta {i}/{len(self.demo_responses)}")
                print(f"   Fuente: {demo_data['source_ai']}")
                print(f"   Respuesta: {demo_data['response'][:100]}...")
                
                # Procesar respuesta
                result = await self.celebro.process_ai_response(
                    response=demo_data['response'],
                    source_ai=demo_data['source_ai'],
                    user_context=demo_data['context'],
                    session_id=session_id
                )
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Procesada exitosamente")
                    print(f"   📊 Tipo: {result['analysis']['response_type']}")
                    print(f"   🎯 Confianza: {result['analysis']['confidence']:.2f}")
                    print(f"   🔧 Nivel técnico: {result['analysis']['technical_level']}")
                    print(f"   💭 Sentimiento: {result['analysis']['sentiment']:.2f}")
                    print(f"   🎨 Alternativas generadas: {len(result['alternatives'])}")
                    
                    if result['knowledge_synthesis']:
                        print(f"   🧠 Conocimiento sintetizado: {result['knowledge_synthesis']['topic']}")
            
            print("\n" + "=" * 60)
            print("🔍 DEMOSTRACIÓN DE CONSULTAS A @celebro")
            print("=" * 60)
            
            # Demostrar consultas a @celebro
            demo_queries = [
                "¿Qué es la inteligencia artificial?",
                "¿Cómo funciona el machine learning?",
                "¿Cuáles son los algoritmos de optimización en IA?",
                "¿Qué aspectos éticos debo considerar en IA?",
                "¿Cuál es el impacto de la IA en la sociedad?"
            ]
            
            for i, query in enumerate(demo_queries, 1):
                print(f"\n❓ Consulta {i}: {query}")
                
                # Consultar @celebro
                result = await self.celebro.query_celebro(
                    query=query,
                    context={"user_level": "intermediate", "time_of_day": "afternoon"},
                    session_id=session_id
                )
                
                if 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   🧠 Respuesta de @celebro:")
                    print(f"   {result['response']}")
                    print(f"   📚 Fuentes de conocimiento: {result['knowledge_sources']}")
            
            print("\n" + "=" * 60)
            print("📊 ESTADÍSTICAS DE @celebro")
            print("=" * 60)
            
            # Mostrar estadísticas
            stats = await self.celebro.get_celebro_stats()
            print(f"📈 Sesiones activas: {stats['sessions']['active']}")
            print(f"📥 Total respuestas procesadas: {stats['sessions']['total_responses_processed']}")
            print(f"🧠 Conocimiento sintetizado: {stats['knowledge_base']['synthesized_knowledge']}")
            print(f"📚 Entradas de contexto: {stats['knowledge_base']['context_entries']}")
            
            print("\n🔧 Estadísticas de módulos:")
            print(f"   📊 Analizador: {stats['modules']['analyzer']['total_analyses']} análisis")
            print(f"   🎯 Generador: {stats['modules']['generator']['total_generated']} alternativas")
            print(f"   🧠 Sintetizador: {stats['modules']['synthesizer']['total_syntheses']} síntesis")
            
            print("\n" + "=" * 60)
            print("🎉 DEMOSTRACIÓN COMPLETADA")
            print("=" * 60)
            print("💡 @celebro ha procesado y sintetizado conocimiento de múltiples IAs")
            print("💡 Ahora puede responder consultas de manera contextualizada")
            print("💡 El sistema aprende y mejora con cada interacción")
            
        except Exception as e:
            logger.error(f"Error en demostración: {e}")
            print(f"❌ Error en demostración: {e}")
    
    async def interactive_demo(self):
        """Demostración interactiva"""
        try:
            print("\n" + "=" * 60)
            print("🎮 MODO INTERACTIVO DE @celebro")
            print("=" * 60)
            print("💡 Escribe 'quit' para salir")
            print("💡 Escribe 'stats' para ver estadísticas")
            print("💡 Escribe 'export' para exportar conocimiento")
            print()
            
            session_id = f"interactive_{int(datetime.now().timestamp())}"
            
            while True:
                try:
                    query = input("\n❓ Tu consulta: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'salir']:
                        print("👋 ¡Hasta luego!")
                        break
                    
                    elif query.lower() == 'stats':
                        stats = await self.celebro.get_celebro_stats()
                        print(f"\n📊 Estadísticas actuales:")
                        print(f"   Sesiones: {stats['sessions']['total']}")
                        print(f"   Respuestas procesadas: {stats['sessions']['total_responses_processed']}")
                        print(f"   Conocimiento sintetizado: {stats['knowledge_base']['synthesized_knowledge']}")
                        continue
                    
                    elif query.lower() == 'export':
                        export_data = await self.celebro.export_knowledge()
                        print(f"\n📤 Conocimiento exportado:")
                        print(f"   Sesiones: {len(export_data['sessions'])}")
                        print(f"   Conocimiento: {len(export_data['knowledge_database'])}")
                        continue
                    
                    elif not query:
                        continue
                    
                    # Procesar consulta
                    result = await self.celebro.query_celebro(
                        query=query,
                        context={"user_level": "intermediate"},
                        session_id=session_id
                    )
                    
                    if 'error' in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"\n🧠 @celebro responde:")
                        print(f"{result['response']}")
                        print(f"\n📚 Basado en {result['knowledge_sources']} fuentes de conocimiento")
                
                except KeyboardInterrupt:
                    print("\n👋 ¡Hasta luego!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        except Exception as e:
            logger.error(f"Error en demo interactiva: {e}")
            print(f"❌ Error en demo interactiva: {e}")

async def main():
    """Función principal"""
    demo = CelebroDemo()
    
    print("🚀 Iniciando demostración de @celebro...")
    
    # Ejecutar demostración completa
    await demo.run_demo()
    
    # Preguntar si quiere modo interactivo
    try:
        interactive = input("\n¿Quieres probar el modo interactivo? (y/n): ").strip().lower()
        if interactive in ['y', 'yes', 'sí', 'si']:
            await demo.interactive_demo()
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error: {e}")
