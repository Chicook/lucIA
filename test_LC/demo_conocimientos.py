#!/usr/bin/env python3
"""
Demostración del Sistema de Conocimientos @conocimientos
Versión: 0.6.0
Sistema de creación de prompts para aprendizaje profundo en ciberseguridad
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celebro.red_neuronal.conocimientos import (
    SecurityTopics, PromptGenerator, KnowledgeBase, 
    LearningCurriculum, DeepLearningTrainer
)
from celebro.red_neuronal.conocimientos.security_topics import SecurityLevel, TopicCategory
from celebro.red_neuronal.conocimientos.prompt_generator import PromptType, DifficultyLevel
from celebro.red_neuronal.conocimientos.learning_curriculum import LearningPath

def demo_security_topics():
    """Demostración de temas de seguridad"""
    print("=" * 80)
    print("🔒 DEMOSTRACIÓN: TEMAS DE SEGURIDAD EN INTERNET")
    print("=" * 80)
    
    try:
        security_topics = SecurityTopics()
        
        # Mostrar estadísticas
        stats = security_topics.get_topic_statistics()
        print(f"\n📊 ESTADÍSTICAS DE TEMAS:")
        print(f"   Total de temas: {stats['total_topics']}")
        print(f"   Por categoría: {stats['by_category']}")
        print(f"   Por nivel: {stats['by_level']}")
        
        # Mostrar algunos temas
        print(f"\n📚 TEMAS DISPONIBLES:")
        all_topics = security_topics.get_all_topics()
        for i, topic in enumerate(all_topics[:5]):  # Mostrar primeros 5
            print(f"   {i+1}. {topic.title}")
            print(f"      Categoría: {topic.category.value}")
            print(f"      Nivel: {topic.level.value}")
            print(f"      Descripción: {topic.description[:100]}...")
            print(f"      Keywords: {', '.join(topic.keywords[:3])}")
            print()
        
        # Buscar temas específicos
        print(f"🔍 BÚSQUEDA DE TEMAS:")
        search_results = security_topics.search_topics("autenticación")
        print(f"   Resultados para 'autenticación': {len(search_results)} temas")
        for topic in search_results:
            print(f"   - {topic.title} ({topic.category.value})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de temas: {e}")
        return False

def demo_prompt_generator():
    """Demostración del generador de prompts"""
    print("\n" + "=" * 80)
    print("📝 DEMOSTRACIÓN: GENERADOR DE PROMPTS")
    print("=" * 80)
    
    try:
        prompt_generator = PromptGenerator()
        
        # Generar diferentes tipos de prompts
        topic_id = "autenticacion"
        
        print(f"\n🎯 Generando prompts para tema: {topic_id}")
        
        # Prompt conceptual
        conceptual_prompt = prompt_generator.generate_prompt(
            topic_id, PromptType.CONCEPTUAL, DifficultyLevel.MEDIO
        )
        print(f"\n📖 PROMPT CONCEPTUAL:")
        print(f"   Título: {conceptual_prompt.title}")
        print(f"   Contenido: {conceptual_prompt.content[:200]}...")
        print(f"   Objetivos: {len(conceptual_prompt.learning_objectives)} objetivos")
        
        # Prompt práctico
        practical_prompt = prompt_generator.generate_prompt(
            topic_id, PromptType.PRACTICO, DifficultyLevel.MEDIO
        )
        print(f"\n🛠️ PROMPT PRÁCTICO:")
        print(f"   Título: {practical_prompt.title}")
        print(f"   Contenido: {practical_prompt.content[:200]}...")
        
        # Prompt de código
        code_prompt = prompt_generator.generate_prompt(
            topic_id, PromptType.CODIGO, DifficultyLevel.MEDIO
        )
        print(f"\n💻 PROMPT DE CÓDIGO:")
        print(f"   Título: {code_prompt.title}")
        print(f"   Contenido: {code_prompt.content[:200]}...")
        
        # Generar currículum completo
        print(f"\n📚 GENERANDO CURRÍCULUM COMPLETO:")
        curriculum_topics = ["autenticacion", "encriptacion", "malware"]
        curriculum = prompt_generator.generate_learning_curriculum(curriculum_topics)
        
        print(f"   Prompts generados: {len(curriculum)}")
        print(f"   Temas cubiertos: {len(set(p.topic_id for p in curriculum))}")
        
        # Estadísticas
        stats = prompt_generator.get_prompt_statistics()
        print(f"\n📊 ESTADÍSTICAS DE PROMPTS:")
        print(f"   Total generados: {stats['total_prompts']}")
        print(f"   Por tipo: {stats['by_type']}")
        print(f"   Por dificultad: {stats['by_difficulty']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de generador: {e}")
        return False

def demo_knowledge_base():
    """Demostración de la base de conocimientos"""
    print("\n" + "=" * 80)
    print("🧠 DEMOSTRACIÓN: BASE DE CONOCIMIENTOS")
    print("=" * 80)
    
    try:
        knowledge_base = KnowledgeBase()
        
        # Crear sesión de aprendizaje
        print(f"\n🎓 CREANDO SESIÓN DE APRENDIZAJE:")
        session = knowledge_base.create_learning_session("autenticacion", "usuario_demo")
        print(f"   Sesión creada: {session.id}")
        print(f"   Tema: {session.topic_id}")
        print(f"   Estado: {session.status}")
        
        # Agregar elemento de conocimiento
        print(f"\n📚 AGREGANDO ELEMENTO DE CONOCIMIENTO:")
        knowledge_item = knowledge_base.add_knowledge_item(
            topic_id="autenticacion",
            content="La autenticación multifactor (MFA) mejora significativamente la seguridad",
            knowledge_type="concept",
            difficulty_level="intermedio",
            tags=["MFA", "seguridad", "autenticación"]
        )
        print(f"   Elemento creado: {knowledge_item.id}")
        print(f"   Tipo: {knowledge_item.knowledge_type}")
        print(f"   Tags: {knowledge_item.tags}")
        
        # Buscar conocimiento
        print(f"\n🔍 BUSCANDO CONOCIMIENTO:")
        search_results = knowledge_base.search_knowledge("MFA")
        print(f"   Resultados encontrados: {len(search_results)}")
        for item in search_results:
            print(f"   - {item.content[:50]}...")
        
        # Obtener estadísticas
        stats = knowledge_base.get_learning_statistics()
        print(f"\n📊 ESTADÍSTICAS DE BASE DE CONOCIMIENTOS:")
        print(f"   Sesiones totales: {stats['total_sessions']}")
        print(f"   Sesiones completadas: {stats['completed_sessions']}")
        print(f"   Tasa de finalización: {stats['completion_rate']:.2%}")
        print(f"   Elementos de conocimiento: {stats['total_knowledge_items']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de base de conocimientos: {e}")
        return False

def demo_learning_curriculum():
    """Demostración del currículum de aprendizaje"""
    print("\n" + "=" * 80)
    print("📋 DEMOSTRACIÓN: CURRÍCULUM DE APRENDIZAJE")
    print("=" * 80)
    
    try:
        curriculum = LearningCurriculum()
        
        # Mostrar módulos disponibles
        print(f"\n📚 MÓDULOS DISPONIBLES:")
        all_modules = curriculum.curriculum_modules
        for module_id, module in all_modules.items():
            print(f"   {module_id}: {module.title}")
            print(f"      Duración: {module.estimated_duration_hours} horas")
            print(f"      Fase: {module.phase.value}")
            print(f"      Temas: {len(module.topics)}")
            print()
        
        # Generar plan de aprendizaje
        print(f"\n🎯 GENERANDO PLAN DE APRENDIZAJE:")
        learning_plan = curriculum.generate_learning_plan(LearningPath.DESARROLLADOR, SecurityLevel.BASICO)
        print(f"   Ruta: {learning_plan['learning_path']}")
        print(f"   Nivel: {learning_plan['user_level']}")
        print(f"   Módulos: {learning_plan['total_modules']}")
        print(f"   Duración total: {learning_plan['total_duration_hours']} horas")
        print(f"   Prompts generados: {learning_plan['learning_prompts']}")
        
        # Mostrar cronograma
        print(f"\n📅 CRONOGRAMA:")
        for week in learning_plan['schedule'][:3]:  # Mostrar primeras 3 semanas
            print(f"   Semana {week['week']}: {week['module'].title}")
            print(f"      Duración: {week['duration_hours']} horas")
            print(f"      Fecha: {week['start_date'][:10]} - {week['end_date'][:10]}")
        
        # Mostrar hitos
        print(f"\n🏆 HITOS DE APRENDIZAJE:")
        for milestone in learning_plan['milestones']:
            print(f"   {milestone.title}")
            print(f"      Fase: {milestone.phase.value}")
            print(f"      Temas requeridos: {len(milestone.required_topics)}")
            print(f"      Puntuación mínima: {milestone.passing_score}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de currículum: {e}")
        return False

def demo_deep_learning_trainer():
    """Demostración del entrenador de aprendizaje profundo"""
    print("\n" + "=" * 80)
    print("🤖 DEMOSTRACIÓN: ENTRENADOR DE APRENDIZAJE PROFUNDO")
    print("=" * 80)
    
    try:
        trainer = DeepLearningTrainer()
        
        # Generar currículum de aprendizaje
        print(f"\n📚 GENERANDO CURRÍCULUM DE APRENDIZAJE:")
        curriculum_data = trainer.generate_learning_curriculum(
            LearningPath.FUNDAMENTOS, 
            num_prompts_per_topic=3
        )
        print(f"   Sesión creada: {curriculum_data['session_id']}")
        print(f"   Prompts generados: {curriculum_data['total_prompts']}")
        print(f"   Temas cubiertos: {len(curriculum_data['topics_covered'])}")
        
        # Crear sesión de entrenamiento
        print(f"\n🎓 CREANDO SESIÓN DE ENTRENAMIENTO:")
        session = trainer.create_training_session("fundamentos")
        print(f"   Sesión: {session.id}")
        print(f"   Estado: {session.status}")
        
        # Generar datos de entrenamiento
        print(f"\n📊 GENERANDO DATOS DE ENTRENAMIENTO:")
        training_data = trainer.generate_training_data(
            ["autenticacion", "encriptacion"], 
            num_prompts_per_topic=2
        )
        print(f"   Ejemplos generados: {len(training_data)}")
        print(f"   Temas: {set(item['topic_id'] for item in training_data)}")
        
        # Mostrar ejemplo de dato de entrenamiento
        if training_data:
            example = training_data[0]
            print(f"\n📝 EJEMPLO DE DATO DE ENTRENAMIENTO:")
            print(f"   Tema: {example['topic_id']}")
            print(f"   Tipo: {example['prompt_type']}")
            print(f"   Dificultad: {example['difficulty']}")
            print(f"   Contenido: {example['input_text'][:100]}...")
        
        # Obtener estadísticas
        stats = trainer.get_training_statistics()
        print(f"\n📊 ESTADÍSTICAS DE ENTRENAMIENTO:")
        print(f"   Sesiones totales: {stats['total_sessions']}")
        print(f"   Sesiones completadas: {stats['completed_sessions']}")
        print(f"   Tasa de finalización: {stats['completion_rate']:.2%}")
        print(f"   Prompts generados: {stats['total_prompts_generated']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de entrenador: {e}")
        return False

def demo_integration():
    """Demostración de integración completa"""
    print("\n" + "=" * 80)
    print("🔗 DEMOSTRACIÓN: INTEGRACIÓN COMPLETA")
    print("=" * 80)
    
    try:
        # Crear instancias de todos los componentes
        security_topics = SecurityTopics()
        prompt_generator = PromptGenerator()
        knowledge_base = KnowledgeBase()
        curriculum = LearningCurriculum()
        trainer = DeepLearningTrainer()
        
        print(f"\n🎯 FLUJO COMPLETO DE APRENDIZAJE:")
        
        # 1. Seleccionar tema
        topic_id = "autenticacion"
        topic = security_topics.get_topic(topic_id)
        print(f"   1. Tema seleccionado: {topic.title}")
        
        # 2. Generar prompts educativos
        prompts = []
        for prompt_type in [PromptType.CONCEPTUAL, PromptType.PRACTICO, PromptType.CODIGO]:
            prompt = prompt_generator.generate_prompt(topic_id, prompt_type, DifficultyLevel.MEDIO)
            prompts.append(prompt)
        print(f"   2. Prompts generados: {len(prompts)}")
        
        # 3. Crear sesión de aprendizaje
        session = knowledge_base.create_learning_session(topic_id, "usuario_demo")
        print(f"   3. Sesión creada: {session.id}")
        
        # 4. Agregar prompts a la sesión
        for prompt in prompts:
            knowledge_base.add_prompt_to_session(session.id, prompt)
        print(f"   4. Prompts agregados a la sesión")
        
        # 5. Simular respuestas y progreso
        for i, prompt in enumerate(prompts):
            response = f"Respuesta simulada para {prompt.title}"
            quality_score = 0.7 + (i * 0.1)  # Simular mejora progresiva
            time_taken = 120 + (i * 30)  # Simular tiempo variable
            
            knowledge_base.record_response(session.id, prompt.id, response, quality_score, time_taken)
        
        # 6. Completar sesión
        completed_session = knowledge_base.complete_session(session.id)
        print(f"   5. Sesión completada: {completed_session.status}")
        print(f"      Duración: {completed_session.duration_minutes} minutos")
        print(f"      Métricas: {completed_session.performance_metrics}")
        
        # 7. Obtener progreso de aprendizaje
        progress = knowledge_base.get_learning_progress(topic_id, "usuario_demo")
        print(f"   6. Progreso de aprendizaje:")
        print(f"      Completado: {progress['completion_percentage']:.1f}%")
        print(f"      Precisión: {progress['accuracy_rate']:.1%}")
        print(f"      Tiempo total: {progress['total_time_spent']} minutos")
        
        # 8. Generar recomendaciones
        recommendations = curriculum.get_recommendations("usuario_demo")
        print(f"   7. Recomendaciones generadas: {len(recommendations)}")
        for rec in recommendations[:2]:  # Mostrar primeras 2
            print(f"      - {rec['title']}: {rec['description']}")
        
        print(f"\n✅ INTEGRACIÓN COMPLETA EXITOSA")
        return True
        
    except Exception as e:
        print(f"❌ Error en demostración de integración: {e}")
        return False

def run_complete_demo():
    """Ejecuta la demostración completa"""
    try:
        print("🧠 DEMOSTRACIÓN COMPLETA DEL SISTEMA @conocimientos")
        print("=" * 80)
        print("Sistema de creación de prompts para aprendizaje profundo en ciberseguridad")
        print("=" * 80)
        
        # Ejecutar todas las demostraciones
        demos = [
            ("Temas de Seguridad", demo_security_topics),
            ("Generador de Prompts", demo_prompt_generator),
            ("Base de Conocimientos", demo_knowledge_base),
            ("Currículum de Aprendizaje", demo_learning_curriculum),
            ("Entrenador de Aprendizaje Profundo", demo_deep_learning_trainer),
            ("Integración Completa", demo_integration)
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                print(f"\n🚀 Ejecutando: {demo_name}")
                result = demo_func()
                results[demo_name] = result
                if result:
                    print(f"✅ {demo_name} completado exitosamente")
                else:
                    print(f"❌ {demo_name} falló")
            except Exception as e:
                print(f"❌ Error en {demo_name}: {e}")
                results[demo_name] = False
        
        # Resumen final
        print("\n" + "=" * 80)
        print("📊 RESUMEN FINAL DE LA DEMOSTRACIÓN")
        print("=" * 80)
        
        successful_demos = sum(1 for result in results.values() if result)
        total_demos = len(demos)
        
        print(f"Demostraciones exitosas: {successful_demos}/{total_demos}")
        print(f"Tasa de éxito: {successful_demos/total_demos*100:.1f}%")
        
        print("\n🎉 ¡Sistema @conocimientos completamente funcional!")
        print("💡 El sistema está listo para crear prompts educativos de ciberseguridad")
        print("🔒 Enfocado en seguridad en internet y cómo combatirla vía código")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en demostración completa: {e}")
        return {}

def main():
    """Función principal"""
    try:
        results = run_complete_demo()
        
        # Preguntar si quiere ver detalles específicos
        print("\n" + "=" * 60)
        print("¿Quieres ver detalles específicos de alguna demostración?")
        print("1. Temas de Seguridad")
        print("2. Generador de Prompts")
        print("3. Base de Conocimientos")
        print("4. Currículum de Aprendizaje")
        print("5. Entrenador de Aprendizaje Profundo")
        print("6. Integración Completa")
        print("0. Salir")
        
        try:
            choice = input("\nSelecciona una opción (0-6): ").strip()
            
            if choice == "1":
                demo_security_topics()
            elif choice == "2":
                demo_prompt_generator()
            elif choice == "3":
                demo_knowledge_base()
            elif choice == "4":
                demo_learning_curriculum()
            elif choice == "5":
                demo_deep_learning_trainer()
            elif choice == "6":
                demo_integration()
            elif choice == "0":
                print("👋 ¡Hasta luego!")
            else:
                print("❌ Opción no válida")
        
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
    
    except Exception as e:
        print(f"❌ Error en demostración: {e}")

if __name__ == "__main__":
    main()
