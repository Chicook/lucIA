"""
Script de Prueba para Red Neuronal Profunda (DNN)
================================================

Este script verifica que todos los módulos funcionen correctamente
antes de ejecutar el entrenamiento completo. Incluye pruebas unitarias
básicas y validación de la integración entre módulos.

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import sys
import traceback
import numpy as np
from datetime import datetime

def test_imports():
    """Prueba que todos los imports funcionen correctamente."""
    try:
        print("🔍 Probando imports...")
        
        # Test modelo.py
        from modelo import crear_modelo, get_model_info, validate_model_architecture
        print("   ✅ modelo.py importado correctamente")
        
        # Test datos.py
        from datos import cargar_datos_simulados, get_data_statistics
        print("   ✅ datos.py importado correctamente")
        
        # Test entrenar.py
        from entrenar import DNNTrainer
        print("   ✅ entrenar.py importado correctamente")
        
        # Test TensorFlow
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__} disponible")
        
        # Test otras dependencias
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("   ✅ Dependencias principales disponibles")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en imports: {str(e)}")
        return False

def test_data_generation():
    """Prueba la generación de datos simulados."""
    try:
        print("\n📊 Probando generación de datos...")
        
        from datos import cargar_datos_simulados, get_data_statistics
        
        # Generar datos pequeños para prueba
        X_train, y_train = cargar_datos_simulados(n_samples=100, n_features=8, n_classes=4)
        
        # Verificar formas
        assert X_train.shape == (100, 8), f"Forma X_train incorrecta: {X_train.shape}"
        assert y_train.shape == (100, 4), f"Forma y_train incorrecta: {y_train.shape}"
        print(f"   ✅ Datos generados: X={X_train.shape}, y={y_train.shape}")
        
        # Verificar tipos
        assert X_train.dtype == np.float64, f"Tipo X_train incorrecto: {X_train.dtype}"
        assert y_train.dtype == np.float32, f"Tipo y_train incorrecto: {y_train.dtype}"
        print("   ✅ Tipos de datos correctos")
        
        # Verificar estadísticas
        stats = get_data_statistics(X_train, y_train)
        assert stats['n_samples'] == 100, "Número de muestras incorrecto"
        assert stats['n_features'] == 8, "Número de características incorrecto"
        assert stats['n_classes'] == 4, "Número de clases incorrecto"
        print("   ✅ Estadísticas calculadas correctamente")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en generación de datos: {str(e)}")
        return False

def test_model_creation():
    """Prueba la creación del modelo DNN."""
    try:
        print("\n🧠 Probando creación del modelo...")
        
        from modelo import crear_modelo, validate_model_architecture, get_model_info
        
        # Crear modelo
        model = crear_modelo(input_shape=8, num_classes=4)
        print("   ✅ Modelo creado exitosamente")
        
        # Verificar arquitectura
        assert len(model.layers) == 5, f"Número de capas incorrecto: {len(model.layers)}"
        print(f"   ✅ Arquitectura correcta: {len(model.layers)} capas")
        
        # Validar arquitectura
        is_valid = validate_model_architecture(model)
        assert is_valid, "Arquitectura del modelo no es válida"
        print("   ✅ Arquitectura validada")
        
        # Obtener información
        info = get_model_info(model)
        assert info['total_params'] > 0, "Modelo sin parámetros"
        print(f"   ✅ Modelo con {info['total_params']:,} parámetros")
        
        # Verificar compilación
        assert model.optimizer is not None, "Modelo no compilado"
        assert model.loss is not None, "Función de pérdida no configurada"
        print("   ✅ Modelo compilado correctamente")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en creación del modelo: {str(e)}")
        return False

def test_model_prediction():
    """Prueba que el modelo pueda hacer predicciones."""
    try:
        print("\n🔮 Probando predicciones del modelo...")
        
        from modelo import crear_modelo
        from datos import cargar_datos_simulados
        
        # Crear modelo
        model = crear_modelo()
        
        # Generar datos de prueba
        X_test, y_test = cargar_datos_simulados(n_samples=10)
        
        # Hacer predicciones
        predictions = model.predict(X_test, verbose=0)
        print(f"   ✅ Predicciones generadas: {predictions.shape}")
        
        # Verificar formas
        assert predictions.shape == (10, 4), f"Forma de predicciones incorrecta: {predictions.shape}"
        
        # Verificar que las predicciones sumen 1 (softmax)
        prediction_sums = np.sum(predictions, axis=1)
        assert np.allclose(prediction_sums, 1.0), "Predicciones no suman 1"
        print("   ✅ Predicciones válidas (suman 1)")
        
        # Verificar que no haya NaN
        assert not np.any(np.isnan(predictions)), "Predicciones contienen NaN"
        print("   ✅ Predicciones sin NaN")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en predicciones: {str(e)}")
        return False

def test_trainer_initialization():
    """Prueba la inicialización del entrenador."""
    try:
        print("\n🏋️ Probando inicialización del entrenador...")
        
        from entrenar import DNNTrainer
        
        # Crear entrenador
        trainer = DNNTrainer(epochs=2, validation_split=0.2, batch_size=16)
        print("   ✅ Entrenador inicializado")
        
        # Verificar configuración
        assert trainer.epochs == 2, "Épocas incorrectas"
        assert trainer.validation_split == 0.2, "Validation split incorrecto"
        assert trainer.batch_size == 16, "Batch size incorrecto"
        print("   ✅ Configuración correcta")
        
        # Verificar directorios
        import os
        assert os.path.exists('logs'), "Directorio logs no creado"
        assert os.path.exists('models'), "Directorio models no creado"
        assert os.path.exists('plots'), "Directorio plots no creado"
        assert os.path.exists('reports'), "Directorio reports no creado"
        print("   ✅ Directorios creados correctamente")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en inicialización del entrenador: {str(e)}")
        return False

def test_quick_training():
    """Prueba un entrenamiento rápido con pocas épocas."""
    try:
        print("\n🚀 Probando entrenamiento rápido...")
        
        from entrenar import DNNTrainer
        from datos import cargar_datos_simulados
        
        # Crear entrenador con configuración mínima
        trainer = DNNTrainer(epochs=1, validation_split=0.2, batch_size=32)
        
        # Generar datos pequeños
        X_train, y_train = cargar_datos_simulados(n_samples=50)
        print(f"   ✅ Datos de prueba: {X_train.shape}")
        
        # Entrenar modelo
        results = trainer.train_model(X_train, y_train)
        print("   ✅ Entrenamiento completado")
        
        # Verificar resultados
        assert 'training_duration' in results, "Duración no registrada"
        assert 'epochs_completed' in results, "Épocas no registradas"
        assert results['epochs_completed'] >= 1, "Ninguna época completada"
        print(f"   ✅ Resultados: {results['epochs_completed']} épocas en {results['training_duration']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en entrenamiento rápido: {str(e)}")
        return False

def main():
    """Función principal de pruebas."""
    print("=" * 80)
    print("🧪 PRUEBAS DE RED NEURONAL PROFUNDA (DNN)")
    print("=" * 80)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Generación de Datos", test_data_generation),
        ("Creación del Modelo", test_model_creation),
        ("Predicciones del Modelo", test_model_prediction),
        ("Inicialización del Entrenador", test_trainer_initialization),
        ("Entrenamiento Rápido", test_quick_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"📊 RESULTADOS: {passed}/{total} pruebas pasaron")
    print(f"⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El módulo DNN está listo para usar.")
        print("\n📋 Para ejecutar el entrenamiento completo:")
        print("   python entrenar.py")
        return True
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores antes de continuar.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
