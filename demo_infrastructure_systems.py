#!/usr/bin/env python3
"""
Demostración de Sistemas de Infraestructura - LucIA
Versión: 0.6.0
Demuestra todos los sistemas de infraestructura creados
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_cache_system():
    """Demuestra el sistema de caché inteligente"""
    print("=" * 80)
    print("🗄️ DEMO: SISTEMA DE CACHÉ INTELIGENTE")
    print("=" * 80)
    
    try:
        from cache.intelligent_cache import IntelligentCache, CacheStrategy
        
        cache = IntelligentCache()
        
        # Test básico
        print("📝 Probando almacenamiento y recuperación...")
        cache.set("test_key", "test_value", ttl=60)
        value = cache.get("test_key")
        print(f"   Valor almacenado: {value}")
        
        # Test con diferentes estrategias
        print("\n🔄 Probando diferentes estrategias de caché...")
        cache.set("lru_key", "lru_value", strategy=CacheStrategy.LRU)
        cache.set("lfu_key", "lfu_value", strategy=CacheStrategy.LFU)
        cache.set("ttl_key", "ttl_value", ttl=5, strategy=CacheStrategy.TTL)
        
        # Simular accesos para LFU
        for _ in range(5):
            cache.get("lfu_key")
        
        # Obtener estadísticas
        stats = cache.get_stats()
        print(f"   Total de elementos: {stats.total_items}")
        print(f"   Tasa de aciertos: {stats.hit_rate:.2%}")
        print(f"   Uso de memoria: {stats.memory_usage:.1f}%")
        
        # Test de optimización
        print("\n⚡ Probando optimización...")
        optimization = cache.optimize()
        print(f"   Puntuación de optimización: {optimization.get('optimization_score', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo del caché: {e}")
        return False

async def demo_temp_system():
    """Demuestra el sistema de archivos temporales"""
    print("\n" + "=" * 80)
    print("📁 DEMO: SISTEMA DE ARCHIVOS TEMPORALES")
    print("=" * 80)
    
    try:
        from temp.temp_manager import TempFileManager, TempFileType
        
        temp_manager = TempFileManager()
        
        # Test básico
        print("📝 Creando archivos temporales...")
        file_id1 = temp_manager.create_temp_file(
            TempFileType.CACHE, 
            "Contenido de prueba para caché",
            ".txt",
            ttl=60
        )
        file_id2 = temp_manager.create_temp_file(
            TempFileType.PROCESSING,
            "Datos de procesamiento",
            ".json",
            ttl=300
        )
        
        print(f"   Archivo 1 creado: {file_id1}")
        print(f"   Archivo 2 creado: {file_id2}")
        
        # Test de lectura
        print("\n📖 Leyendo archivos temporales...")
        content1 = temp_manager.read_temp_file(file_id1)
        content2 = temp_manager.read_temp_file(file_id2)
        print(f"   Contenido 1: {content1}")
        print(f"   Contenido 2: {content2}")
        
        # Test de escritura
        print("\n✏️ Escribiendo en archivo temporal...")
        success = temp_manager.write_temp_file(file_id1, "Contenido actualizado")
        print(f"   Escritura exitosa: {success}")
        
        # Obtener estadísticas
        stats = temp_manager.get_stats()
        print(f"\n📊 Estadísticas:")
        print(f"   Total de archivos: {stats['total_files']}")
        print(f"   Tamaño total: {stats['total_size']} bytes")
        print(f"   Uso de espacio: {stats['usage_percentage']:.1f}%")
        
        # Limpiar archivos
        temp_manager.delete_temp_file(file_id1)
        temp_manager.delete_temp_file(file_id2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de archivos temporales: {e}")
        return False

async def demo_models_systems():
    """Demuestra los sistemas de modelos"""
    print("\n" + "=" * 80)
    print("🧠 DEMO: SISTEMAS DE MODELOS")
    print("=" * 80)
    
    try:
        from models.neural.neural_models import NeuralModelManager, ModelType, ModelStatus
        from models.decision.decision_models import DecisionModelManager, DecisionType, DecisionStatus
        from models.optimization.optimization_models import OptimizationModelManager, OptimizationType, OptimizationStatus
        
        # Test de modelos neuronales
        print("🧠 Probando modelos neuronales...")
        neural_manager = NeuralModelManager()
        
        # Simular un modelo simple
        import numpy as np
        simple_model = np.random.random((10, 5))
        
        model_id = neural_manager.save_model(
            simple_model,
            "Modelo de Prueba",
            ModelType.CLASSIFICATION,
            "Modelo de prueba para demostración",
            input_shape=(10, 5),
            output_shape=(1,),
            parameters={"learning_rate": 0.01, "epochs": 100},
            performance_metrics={"accuracy": 0.95, "loss": 0.05}
        )
        
        print(f"   Modelo guardado: {model_id}")
        
        # Cargar modelo
        loaded_model = neural_manager.load_model(model_id)
        print(f"   Modelo cargado: {type(loaded_model)}")
        
        # Estadísticas
        neural_stats = neural_manager.get_model_stats()
        print(f"   Total de modelos: {neural_stats['total_models']}")
        
        # Test de modelos de decisión
        print("\n🎯 Probando modelos de decisión...")
        decision_manager = DecisionModelManager()
        
        # Simular un modelo de decisión
        decision_model = {"rules": ["if x > 0 then positive", "if x < 0 then negative"]}
        
        decision_id = decision_manager.save_decision_model(
            decision_model,
            "Clasificador Simple",
            DecisionType.RULE_BASED,
            "Clasificador basado en reglas",
            input_features=["x"],
            output_classes=["positive", "negative"]
        )
        
        print(f"   Modelo de decisión guardado: {decision_id}")
        
        # Añadir regla
        rule_id = decision_manager.add_decision_rule(
            decision_id,
            "x > 0",
            "positive",
            confidence=0.9,
            priority=1
        )
        print(f"   Regla añadida: {rule_id}")
        
        # Evaluar decisión
        result = decision_manager.evaluate_decision(decision_id, {"x": 5})
        print(f"   Evaluación: {result}")
        
        # Test de modelos de optimización
        print("\n⚡ Probando modelos de optimización...")
        optimization_manager = OptimizationModelManager()
        
        # Simular un modelo de optimización
        optimization_model = {"algorithm": "genetic", "population_size": 100}
        
        opt_id = optimization_manager.save_optimization_model(
            optimization_model,
            "Optimizador Genético",
            OptimizationType.GENETIC_ALGORITHM,
            "Algoritmo genético para optimización",
            objective_function="minimize f(x) = x^2",
            constraints=["x >= 0", "x <= 10"]
        )
        
        print(f"   Modelo de optimización guardado: {opt_id}")
        
        # Ejecutar optimización
        result = optimization_manager.run_optimization(opt_id, max_iterations=100)
        print(f"   Optimización completada: {result.best_fitness:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de modelos: {e}")
        return False

async def demo_learning_data_system():
    """Demuestra el sistema de datos de aprendizaje"""
    print("\n" + "=" * 80)
    print("📊 DEMO: SISTEMA DE DATOS DE APRENDIZAJE")
    print("=" * 80)
    
    try:
        from data.learning.learning_data_manager import LearningDataManager, DataType, DataFormat
        import pandas as pd
        import numpy as np
        
        data_manager = LearningDataManager()
        
        # Crear datos de prueba
        print("📝 Creando dataset de prueba...")
        data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Guardar dataset
        dataset_id = data_manager.save_dataset(
            data,
            "Dataset de Prueba",
            DataType.TRAINING,
            "Dataset de prueba para demostración",
            features=["feature1", "feature2", "feature3"],
            target_column="target"
        )
        
        print(f"   Dataset guardado: {dataset_id}")
        
        # Cargar dataset
        loaded_data = data_manager.load_dataset(dataset_id)
        print(f"   Dataset cargado: {type(loaded_data)} con {len(loaded_data)} filas")
        
        # Dividir dataset
        print("\n✂️ Dividiendo dataset...")
        split_result = data_manager.split_dataset(dataset_id, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        print(f"   Conjuntos creados: {split_result}")
        
        # Estadísticas
        stats = data_manager.get_dataset_stats()
        print(f"\n📊 Estadísticas:")
        print(f"   Total de datasets: {stats['total_datasets']}")
        print(f"   Tamaño total: {stats['total_size']} bytes")
        print(f"   Total de filas: {stats['total_rows']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de datos de aprendizaje: {e}")
        return False

async def demo_integration():
    """Demuestra la integración completa"""
    print("\n" + "=" * 80)
    print("🔗 DEMO: INTEGRACIÓN COMPLETA")
    print("=" * 80)
    
    try:
        from src.modulo14.main_modulo14 import InfrastructureIntegrator
        
        integrator = InfrastructureIntegrator()
        
        # Inicializar
        print("🚀 Inicializando integrador...")
        await integrator.initialize_module(None)
        
        # Test de caché
        print("\n🗄️ Probando caché integrado...")
        await integrator.cache_set("integration_test", "test_value", ttl=60)
        cached_value = await integrator.cache_get("integration_test")
        print(f"   Valor en caché: {cached_value}")
        
        # Test de archivos temporales
        print("\n📁 Probando archivos temporales integrados...")
        from temp.temp_manager import TempFileType
        file_id = await integrator.create_temp_file(
            TempFileType.CACHE,
            "Contenido de integración",
            ".txt"
        )
        print(f"   Archivo temporal creado: {file_id}")
        
        # Obtener estado del sistema
        print("\n📊 Obteniendo estado del sistema...")
        status = await integrator.get_system_status()
        print(f"   Estado general: {status['overall_health']}")
        print(f"   Sistemas inicializados: {status['is_initialized']}")
        
        # Optimizar sistemas
        print("\n⚡ Optimizando sistemas...")
        optimization = await integrator.optimize_systems()
        print(f"   Optimización completada")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en demo de integración: {e}")
        return False

async def main():
    """Función principal de demostración"""
    print("🚀 DEMOSTRACIÓN DE SISTEMAS DE INFRAESTRUCTURA - LucIA")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    demos = [
        ("Sistema de Caché Inteligente", demo_cache_system),
        ("Sistema de Archivos Temporales", demo_temp_system),
        ("Sistemas de Modelos", demo_models_systems),
        ("Sistema de Datos de Aprendizaje", demo_learning_data_system),
        ("Integración Completa", demo_integration)
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
        print("🏗️ Los sistemas de infraestructura están funcionando correctamente")
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
