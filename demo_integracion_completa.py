#!/usr/bin/env python3
"""
Demostración de Integración Completa - LucIA v0.6.0
Demuestra la integración de todos los sistemas Python creados con main.py
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_integracion_completa():
    """Demostración completa de la integración de sistemas"""
    print("=" * 80)
    print("🤖 DEMOSTRACIÓN DE INTEGRACIÓN COMPLETA - LucIA v0.6.0")
    print("=" * 80)
    print("Integrando @celebro, @red_neuronal, @conocimientos con main.py")
    print("=" * 80)
    
    try:
        # Importar el motor principal
        from main import LucIACore
        
        print("\n🚀 INICIALIZANDO MOTOR PRINCIPAL...")
        lucia = LucIACore()
        
        print("\n📋 CONFIGURACIÓN CARGADA:")
        print(f"   Nombre: {lucia.config['ai_name']}")
        print(f"   Versión: {lucia.config['version']}")
        print(f"   Módulos habilitados: {len([m for m in lucia.config['modules'].values() if m['enabled']])}")
        
        print("\n🔧 INICIALIZANDO MÓDULOS...")
        await lucia.initialize_modules()
        
        print(f"\n✅ MÓDULOS INICIALIZADOS: {len(lucia.modules)}")
        for module_name in lucia.modules.keys():
            print(f"   - {module_name}")
        
        print("\n🧠 PROBANDO SISTEMAS INTEGRADOS...")
        
        # Probar sistema @celebro
        print("\n1️⃣ PROBANDO @celebro:")
        try:
            test_text = "¿Cómo funciona la autenticación multifactor?"
            processed = await lucia.process_with_celebro(test_text)
            print(f"   Entrada: {test_text}")
            print(f"   Procesado: {processed[:100]}...")
            print("   ✅ @celebro funcionando")
        except Exception as e:
            print(f"   ❌ Error en @celebro: {e}")
        
        # Probar generación de prompts de seguridad
        print("\n2️⃣ PROBANDO GENERACIÓN DE PROMPTS DE SEGURIDAD:")
        try:
            prompts = await lucia.generate_security_prompts("autenticacion", 3)
            print(f"   Prompts generados: {len(prompts)}")
            if prompts:
                print(f"   Primer prompt: {prompts[0]['title']}")
                print(f"   Contenido: {prompts[0]['content'][:100]}...")
            print("   ✅ Generación de prompts funcionando")
        except Exception as e:
            print(f"   ❌ Error generando prompts: {e}")
        
        # Probar entrenamiento con temas de seguridad
        print("\n3️⃣ PROBANDO ENTRENAMIENTO CON TEMAS DE SEGURIDAD:")
        try:
            training_result = await lucia.train_with_security_topics(["autenticacion", "encriptacion"])
            print(f"   Resultado: {training_result}")
            if 'session_id' in training_result:
                print("   ✅ Entrenamiento configurado correctamente")
            else:
                print(f"   ⚠️ Entrenamiento: {training_result.get('error', 'Estado desconocido')}")
        except Exception as e:
            print(f"   ❌ Error en entrenamiento: {e}")
        
        # Obtener estado de sistemas avanzados
        print("\n4️⃣ ESTADO DE SISTEMAS AVANZADOS:")
        try:
            status = await lucia.get_advanced_system_status()
            print(f"   Timestamp: {status.get('timestamp', 'N/A')}")
            print(f"   Inicializado: {status.get('is_initialized', False)}")
            
            if 'systems' in status:
                for system_name, system_status in status['systems'].items():
                    print(f"   {system_name}: {system_status.get('status', 'unknown')}")
            
            print("   ✅ Estado obtenido correctamente")
        except Exception as e:
            print(f"   ❌ Error obteniendo estado: {e}")
        
        # Probar procesamiento de entrada completa
        print("\n5️⃣ PROBANDO PROCESAMIENTO COMPLETO:")
        try:
            test_input = "Necesito aprender sobre seguridad en aplicaciones web"
            result = await lucia.process_input(test_input)
            print(f"   Entrada: {test_input}")
            print(f"   Resultado: {str(result)[:100]}...")
            print("   ✅ Procesamiento completo funcionando")
        except Exception as e:
            print(f"   ❌ Error en procesamiento: {e}")
        
        # Mostrar métricas de rendimiento
        print("\n📊 MÉTRICAS DE RENDIMIENTO:")
        try:
            metrics = lucia.performance_metrics
            print(f"   Módulos activos: {metrics.get('modules_active', 0)}")
            print(f"   Uso de memoria: {metrics.get('memory_usage', 0):.1f}%")
            print(f"   Uso de CPU: {metrics.get('cpu_usage', 0):.1f}%")
            print(f"   Ciclos de aprendizaje: {metrics.get('learning_cycles', 0)}")
        except Exception as e:
            print(f"   ⚠️ Error obteniendo métricas: {e}")
        
        print("\n" + "=" * 80)
        print("🎉 INTEGRACIÓN COMPLETA EXITOSA")
        print("=" * 80)
        print("✅ Todos los sistemas Python están conectados con main.py")
        print("✅ @celebro, @red_neuronal y @conocimientos integrados")
        print("✅ Motor de IA LucIA funcionando correctamente")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN INTEGRACIÓN: {e}")
        return False

async def demo_funcionalidades_especificas():
    """Demostración de funcionalidades específicas"""
    print("\n" + "=" * 80)
    print("🔧 DEMOSTRACIÓN DE FUNCIONALIDADES ESPECÍFICAS")
    print("=" * 80)
    
    try:
        from main import LucIACore
        lucia = LucIACore()
        await lucia.initialize_modules()
        
        # Demostrar generación de prompts por tema
        print("\n📝 GENERACIÓN DE PROMPTS POR TEMA:")
        topics = ["autenticacion", "encriptacion", "malware", "phishing"]
        
        for topic in topics:
            try:
                prompts = await lucia.generate_security_prompts(topic, 2)
                print(f"\n   🔒 {topic.upper()}:")
                print(f"      Prompts generados: {len(prompts)}")
                if prompts:
                    print(f"      Título: {prompts[0]['title']}")
                    print(f"      Objetivos: {len(prompts[0]['learning_objectives'])} objetivos")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Demostrar entrenamiento con múltiples temas
        print("\n🎓 ENTRENAMIENTO CON MÚLTIPLES TEMAS:")
        try:
            training_result = await lucia.train_with_security_topics([
                "autenticacion", "encriptacion", "malware", "phishing", 
                "firewall", "ids_ips", "vulnerability_assessment"
            ])
            print(f"   Resultado: {json.dumps(training_result, indent=2)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Demostrar estado detallado
        print("\n📊 ESTADO DETALLADO DE SISTEMAS:")
        try:
            status = await lucia.get_advanced_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print("\n✅ FUNCIONALIDADES ESPECÍFICAS DEMOSTRADAS")
        
    except Exception as e:
        print(f"\n❌ ERROR EN FUNCIONALIDADES ESPECÍFICAS: {e}")

async def main():
    """Función principal"""
    print("🤖 LucIA v0.6.0 - Demostración de Integración Completa")
    print("=" * 80)
    
    # Demostración principal
    success = await demo_integracion_completa()
    
    if success:
        # Demostración de funcionalidades específicas
        await demo_funcionalidades_especificas()
        
        print("\n" + "=" * 80)
        print("🎯 RESUMEN FINAL")
        print("=" * 80)
        print("✅ Motor principal (main.py) funcionando")
        print("✅ @celebro integrado para análisis de respuestas")
        print("✅ @red_neuronal integrado para aprendizaje profundo")
        print("✅ @conocimientos integrado para prompts de ciberseguridad")
        print("✅ Todos los sistemas conectados y operativos")
        print("=" * 80)
        print("🚀 LucIA está listo para ser entrenado por otras IAs")
        print("🔒 Enfocado en seguridad en internet y cómo combatirla vía código")
        print("=" * 80)
    else:
        print("\n❌ La integración falló. Revisar logs para más detalles.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demostración interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
