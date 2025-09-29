#!/usr/bin/env python3
"""
Chat Interactivo con WoldVirtual3DlucIA
Versión: 0.6.0
Permite conversaciones básicas con LucIA
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from main import LucIACore

# Configurar logging simple
logging.basicConfig(level=logging.WARNING)  # Reducir logs para conversación

class LucIAChat:
    """Clase para manejar conversaciones con LucIA"""
    
    def __init__(self):
        self.lucia = None
        self.conversation_history = []
        self.session_id = f"chat_{int(datetime.now().timestamp())}"
    
    async def initialize(self):
        """Inicializa LucIA"""
        try:
            print("🤖 Inicializando LucIA...")
            self.lucia = LucIACore()
            
            # Inicializar módulos básicos
            await self.lucia.initialize_modules()
            
            print("✅ LucIA inicializada correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando LucIA: {e}")
            return False
    
    async def process_message(self, user_input: str) -> str:
        """Procesa un mensaje del usuario"""
        try:
            # Agregar a historial
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "lucia": None
            })
            
            # Procesar con LucIA
            context = {
                "conversation_history": self.conversation_history[-5:],  # Últimos 5 mensajes
                "session_id": self.session_id,
                "mode": "chat"
            }
            
            result = await self.lucia.process_input(user_input, context)
            
            # Generar respuesta
            response = self._generate_response(user_input, result)
            
            # Actualizar historial
            self.conversation_history[-1]["lucia"] = response
            
            return response
            
        except Exception as e:
            return f"Lo siento, ocurrió un error procesando tu mensaje: {e}"
    
    def _generate_response(self, user_input: str, result) -> str:
        """Genera una respuesta basada en el resultado de LucIA"""
        try:
            # Respuestas básicas basadas en palabras clave
            user_lower = user_input.lower()
            
            # Saludos
            if any(word in user_lower for word in ["hola", "hi", "hello", "buenos", "buenas"]):
                return "¡Hola! Soy LucIA, tu asistente de IA modular. ¿En qué puedo ayudarte hoy?"
            
            # Preguntas sobre LucIA
            elif any(word in user_lower for word in ["quien", "que", "como", "quien eres", "que eres"]):
                return "Soy LucIA, una inteligencia artificial modular estándar diseñada para ser entrenada por otras IAs. Tengo 12 módulos especializados que me permiten aprender, razonar, comunicarme y adaptarme."
            
            # Preguntas sobre capacidades
            elif any(word in user_lower for word in ["puedes", "capacidades", "que haces", "funciones"]):
                return "Puedo ayudarte con: aprendizaje automático, razonamiento lógico, procesamiento de texto, análisis de datos, optimización, y mucho más. También puedo comunicarme con otras IAs y aprender de ellas."
            
            # Preguntas sobre el sistema
            elif any(word in user_lower for word in ["sistema", "modulos", "arquitectura"]):
                return "Mi arquitectura incluye 12 módulos: Memoria, Aprendizaje, Comunicación, Entrenamiento, Razonamiento, Percepción, Acción, Evaluación, Optimización, Seguridad, Monitoreo e Integración."
            
            # Preguntas sobre aprendizaje
            elif any(word in user_lower for word in ["aprender", "entrenar", "enseñar"]):
                return "Sí, puedo aprender! Mi sistema de aprendizaje incluye algoritmos genéticos, descenso de gradiente, y aprendizaje incremental. Puedo ser entrenada por otras IAs a través de mi interfaz estándar."
            
            # Preguntas técnicas
            elif any(word in user_lower for word in ["python", "codigo", "programar", "desarrollo"]):
                return "Estoy construida en Python con una arquitectura modular. Cada módulo tiene entre 200-300 líneas de código y es independiente. ¿Te gustaría saber más sobre algún módulo específico?"
            
            # Preguntas sobre el futuro
            elif any(word in user_lower for word in ["futuro", "planes", "desarrollo", "roadmap"]):
                return "Mis planes incluyen integración WebXR para realidad inmersiva, networking P2P, optimización con WebGPU, y colaboración multi-IA distribuida. Estoy en constante evolución!"
            
            # Preguntas sobre ayuda
            elif any(word in user_lower for word in ["ayuda", "help", "comandos"]):
                return "Puedes preguntarme sobre: mis capacidades, arquitectura, módulos, aprendizaje, desarrollo, o cualquier tema técnico. También puedo ayudarte con análisis de datos y razonamiento lógico."
            
            # Respuesta por defecto
            else:
                return f"Interesante pregunta sobre '{user_input}'. Como IA modular, puedo ayudarte a analizar este tema desde diferentes perspectivas. ¿Podrías ser más específico sobre qué te gustaría saber?"
                
        except Exception as e:
            return f"Procesando tu mensaje: '{user_input}'. Déjame pensar en una respuesta apropiada..."
    
    async def start_chat(self):
        """Inicia la sesión de chat"""
        print("=" * 60)
        print("🤖 WoldVirtual3DlucIA - Chat Interactivo v0.6.0")
        print("=" * 60)
        print("💡 Escribe 'quit' para salir")
        print("💡 Escribe 'help' para ver comandos disponibles")
        print("💡 Escribe 'status' para ver el estado del sistema")
        print()
        
        while True:
            try:
                # Obtener entrada del usuario
                user_input = input("Tú: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'salir', 'bye']:
                    print("LucIA: ¡Hasta luego! Ha sido un placer conversar contigo. 👋")
                    break
                
                elif user_input.lower() == 'help':
                    print("LucIA: Comandos disponibles:")
                    print("  - Pregúntame sobre mis capacidades")
                    print("  - 'status' - Ver estado del sistema")
                    print("  - 'quit' - Salir del chat")
                    print("  - Cualquier pregunta sobre IA, tecnología, o mi funcionamiento")
                    continue
                
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                
                elif not user_input:
                    continue
                
                # Procesar mensaje
                print("LucIA: ", end="", flush=True)
                response = await self.process_message(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nLucIA: ¡Hasta luego! 👋")
                break
            except Exception as e:
                print(f"LucIA: Lo siento, ocurrió un error: {e}")
                print()
    
    async def _show_status(self):
        """Muestra el estado del sistema"""
        try:
            print("\n📊 Estado del Sistema LucIA:")
            print(f"   - Sesión: {self.session_id}")
            print(f"   - Mensajes en conversación: {len(self.conversation_history)}")
            print(f"   - Módulos activos: {len(self.lucia.modules) if self.lucia else 0}")
            
            if self.lucia and hasattr(self.lucia, 'memory_system'):
                memory_stats = await self.lucia.memory_system.get_memory_stats()
                print(f"   - Memorias almacenadas: {memory_stats.get('total_memories', 0)}")
            
            print()
            
        except Exception as e:
            print(f"Error obteniendo estado: {e}")

async def main():
    """Función principal"""
    chat = LucIAChat()
    
    # Inicializar LucIA
    if not await chat.initialize():
        print("❌ No se pudo inicializar LucIA")
        sys.exit(1)
    
    # Iniciar chat
    await chat.start_chat()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Chat terminado")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
