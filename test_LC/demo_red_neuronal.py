#!/usr/bin/env python3
"""
Demostración del Sistema de Red Neuronal @red_neuronal
Versión: 0.6.0
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from celebro.red_neuronal import NeuralCore, NetworkConfig, TrainingConfig

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Demo_Red_Neuronal')

class RedNeuronalDemo:
    """Demostración del sistema de red neuronal"""
    
    def __init__(self):
        self.neural_core = NeuralCore()
        self.scaler = StandardScaler()
        
    def demo_clasificacion_binaria(self):
        """Demostración de clasificación binaria"""
        print("=" * 80)
        print("🔍 DEMOSTRACIÓN: CLASIFICACIÓN BINARIA")
        print("=" * 80)
        
        # Generar datos de ejemplo
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar datos
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convertir a one-hot encoding
        y_train_onehot = np.eye(2)[y_train]
        y_test_onehot = np.eye(2)[y_test]
        
        # Configurar red neuronal
        config = NetworkConfig(
            input_size=20,
            hidden_layers=[64, 32, 16],
            output_size=2,
            activation='relu',
            output_activation='softmax',
            learning_rate=0.001,
            dropout_rate=0.3,
            batch_normalization=True
        )
        
        # Crear red
        network = self.neural_core.create_network(config)
        print(f"\n📊 Arquitectura de la red:")
        print(self.neural_core.get_network_summary())
        
        # Configurar entrenamiento
        training_config = TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping=True,
            patience=10,
            verbose=1
        )
        
        # Entrenar red
        print(f"\n🚀 Iniciando entrenamiento...")
        history = self.neural_core.train(X_train, y_train_onehot, config=training_config)
        
        # Evaluar red
        print(f"\n📈 Evaluando red...")
        test_metrics = self.neural_core.evaluate(X_test, y_test_onehot)
        
        # Predicciones
        predictions = self.neural_core.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   Precisión en prueba: {test_metrics['accuracy']:.4f}")
        print(f"   Pérdida en prueba: {test_metrics['loss']:.4f}")
        
        print(f"\n📋 Reporte de clasificación:")
        print(classification_report(y_test, predicted_classes))
        
        # Mostrar resumen de entrenamiento
        print(f"\n{self.neural_core.get_training_summary()}")
        
        return history, test_metrics
    
    def demo_clasificacion_multiclase(self):
        """Demostración de clasificación multiclase"""
        print("\n" + "=" * 80)
        print("🔍 DEMOSTRACIÓN: CLASIFICACIÓN MULTICLASE (IRIS)")
        print("=" * 80)
        
        # Cargar dataset Iris
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar datos
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convertir a one-hot encoding
        y_train_onehot = np.eye(3)[y_train]
        y_test_onehot = np.eye(3)[y_test]
        
        # Configurar red neuronal
        config = NetworkConfig(
            input_size=4,
            hidden_layers=[32, 16],
            output_size=3,
            activation='relu',
            output_activation='softmax',
            learning_rate=0.01,
            dropout_rate=0.2
        )
        
        # Crear red
        network = self.neural_core.create_network(config)
        
        # Configurar entrenamiento
        training_config = TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=0.01,
            validation_split=0.2,
            early_stopping=True,
            patience=15,
            verbose=1
        )
        
        # Entrenar red
        print(f"\n🚀 Iniciando entrenamiento...")
        history = self.neural_core.train(X_train, y_train_onehot, config=training_config)
        
        # Evaluar red
        print(f"\n📈 Evaluando red...")
        test_metrics = self.neural_core.evaluate(X_test, y_test_onehot)
        
        # Predicciones
        predictions = self.neural_core.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   Precisión en prueba: {test_metrics['accuracy']:.4f}")
        print(f"   Pérdida en prueba: {test_metrics['loss']:.4f}")
        
        print(f"\n📋 Reporte de clasificación:")
        print(classification_report(y_test, predicted_classes, target_names=iris.target_names))
        
        return history, test_metrics
    
    def demo_regresion(self):
        """Demostración de regresión"""
        print("\n" + "=" * 80)
        print("🔍 DEMOSTRACIÓN: REGRESIÓN")
        print("=" * 80)
        
        # Generar datos de regresión
        X, y = make_regression(
            n_samples=1000,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar datos
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Normalizar target
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Configurar red neuronal
        config = NetworkConfig(
            input_size=10,
            hidden_layers=[64, 32, 16],
            output_size=1,
            activation='relu',
            output_activation='linear',
            learning_rate=0.001,
            dropout_rate=0.2
        )
        
        # Crear red
        network = self.neural_core.create_network(config)
        
        # Configurar entrenamiento
        training_config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping=True,
            patience=15,
            verbose=1
        )
        
        # Entrenar red
        print(f"\n🚀 Iniciando entrenamiento...")
        history = self.neural_core.train(X_train, y_train, config=training_config)
        
        # Evaluar red
        print(f"\n📈 Evaluando red...")
        test_metrics = self.neural_core.evaluate(X_test, y_test)
        
        # Predicciones
        predictions = self.neural_core.predict(X_test)
        
        # Calcular métricas adicionales
        mse = np.mean((y_test - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        return history, test_metrics
    
    def demo_red_convolucional(self):
        """Demostración de red convolucional"""
        print("\n" + "=" * 80)
        print("🔍 DEMOSTRACIÓN: RED CONVOLUCIONAL")
        print("=" * 80)
        
        # Generar datos de imagen sintéticos
        n_samples = 1000
        img_size = 28
        n_channels = 1
        n_classes = 10
        
        # Generar imágenes aleatorias con patrones
        X = np.random.randn(n_samples, img_size, img_size, n_channels)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Agregar patrones simples a las imágenes
        for i in range(n_samples):
            class_id = y[i]
            # Crear patrón simple basado en la clase
            X[i, :, :, 0] += np.sin(np.linspace(0, 4*np.pi, img_size)) * (class_id + 1)
            X[i, :, :, 0] += np.cos(np.linspace(0, 4*np.pi, img_size)) * (class_id + 1)
        
        # Normalizar datos
        X = (X - X.mean()) / X.std()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convertir a one-hot encoding
        y_train_onehot = np.eye(n_classes)[y_train]
        y_test_onehot = np.eye(n_classes)[y_test]
        
        # Crear red convolucional
        config = {
            'filters': [32, 64, 128],
            'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
            'pool_sizes': [(2, 2), (2, 2), (2, 2)],
            'dense_units': [128, 64],
            'dropout_rate': 0.5,
            'activation': 'relu',
            'output_activation': 'softmax'
        }
        
        network = self.neural_core.create_convolutional_network(
            input_shape=(img_size, img_size, n_channels),
            num_classes=n_classes,
            config=config
        )
        
        print(f"\n📊 Arquitectura de la red convolucional:")
        print(self.neural_core.get_network_summary())
        
        # Configurar entrenamiento
        training_config = TrainingConfig(
            epochs=20,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping=True,
            patience=5,
            verbose=1
        )
        
        # Entrenar red
        print(f"\n🚀 Iniciando entrenamiento...")
        history = self.neural_core.train(X_train, y_train_onehot, config=training_config)
        
        # Evaluar red
        print(f"\n📈 Evaluando red...")
        test_metrics = self.neural_core.evaluate(X_test, y_test_onehot)
        
        # Predicciones
        predictions = self.neural_core.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Mostrar resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   Precisión en prueba: {test_metrics['accuracy']:.4f}")
        print(f"   Pérdida en prueba: {test_metrics['loss']:.4f}")
        
        return history, test_metrics
    
    def demo_guardar_cargar_modelo(self):
        """Demostración de guardar y cargar modelo"""
        print("\n" + "=" * 80)
        print("🔍 DEMOSTRACIÓN: GUARDAR Y CARGAR MODELO")
        print("=" * 80)
        
        # Generar datos simples
        X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        y_train_onehot = np.eye(2)[y_train]
        y_test_onehot = np.eye(2)[y_test]
        
        # Crear y entrenar modelo
        config = NetworkConfig(
            input_size=10,
            hidden_layers=[32, 16],
            output_size=2,
            activation='relu',
            output_activation='softmax'
        )
        
        network = self.neural_core.create_network(config)
        
        training_config = TrainingConfig(epochs=20, verbose=0)
        history = self.neural_core.train(X_train, y_train_onehot, config=training_config)
        
        # Evaluar modelo original
        original_metrics = self.neural_core.evaluate(X_test, y_test_onehot)
        print(f"📊 Métricas del modelo original:")
        print(f"   Precisión: {original_metrics['accuracy']:.4f}")
        print(f"   Pérdida: {original_metrics['loss']:.4f}")
        
        # Guardar modelo
        model_path = "celebro/red_neuronal/models/demo_model.pkl"
        self.neural_core.save_model(model_path)
        print(f"✅ Modelo guardado en: {model_path}")
        
        # Crear nueva instancia y cargar modelo
        new_neural_core = NeuralCore()
        new_neural_core.load_model(model_path)
        
        # Evaluar modelo cargado
        loaded_metrics = new_neural_core.evaluate(X_test, y_test_onehot)
        print(f"\n📊 Métricas del modelo cargado:")
        print(f"   Precisión: {loaded_metrics['accuracy']:.4f}")
        print(f"   Pérdida: {loaded_metrics['loss']:.4f}")
        
        # Verificar que son iguales
        if abs(original_metrics['accuracy'] - loaded_metrics['accuracy']) < 1e-6:
            print("✅ Modelo cargado correctamente - métricas idénticas")
        else:
            print("❌ Error al cargar modelo - métricas diferentes")
        
        return original_metrics, loaded_metrics
    
    def run_complete_demo(self):
        """Ejecuta la demostración completa"""
        try:
            print("🧠 DEMOSTRACIÓN COMPLETA DEL SISTEMA DE RED NEURONAL @red_neuronal")
            print("=" * 80)
            print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Ejecutar todas las demostraciones
            demos = [
                ("Clasificación Binaria", self.demo_clasificacion_binaria),
                ("Clasificación Multiclase", self.demo_clasificacion_multiclase),
                ("Regresión", self.demo_regresion),
                ("Red Convolucional", self.demo_red_convolucional),
                ("Guardar/Cargar Modelo", self.demo_guardar_cargar_modelo)
            ]
            
            results = {}
            
            for demo_name, demo_func in demos:
                try:
                    print(f"\n🚀 Ejecutando: {demo_name}")
                    result = demo_func()
                    results[demo_name] = result
                    print(f"✅ {demo_name} completado exitosamente")
                except Exception as e:
                    print(f"❌ Error en {demo_name}: {e}")
                    results[demo_name] = None
            
            # Resumen final
            print("\n" + "=" * 80)
            print("📊 RESUMEN FINAL DE LA DEMOSTRACIÓN")
            print("=" * 80)
            
            successful_demos = sum(1 for result in results.values() if result is not None)
            total_demos = len(demos)
            
            print(f"Demostraciones exitosas: {successful_demos}/{total_demos}")
            print(f"Tasa de éxito: {successful_demos/total_demos*100:.1f}%")
            
            print("\n🎉 ¡Demostración completada!")
            print("💡 El sistema @red_neuronal está listo para uso en producción")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en demostración completa: {e}")
            print(f"❌ Error en demostración: {e}")

def main():
    """Función principal"""
    try:
        demo = RedNeuronalDemo()
        results = demo.run_complete_demo()
        
        # Preguntar si quiere ver detalles específicos
        print("\n" + "=" * 60)
        print("¿Quieres ver detalles específicos de alguna demostración?")
        print("1. Clasificación Binaria")
        print("2. Clasificación Multiclase")
        print("3. Regresión")
        print("4. Red Convolucional")
        print("5. Guardar/Cargar Modelo")
        print("0. Salir")
        
        try:
            choice = input("\nSelecciona una opción (0-5): ").strip()
            
            if choice == "1" and results.get("Clasificación Binaria"):
                print("\n📊 Detalles de Clasificación Binaria:")
                # Mostrar más detalles si es necesario
            
            elif choice == "2" and results.get("Clasificación Multiclase"):
                print("\n📊 Detalles de Clasificación Multiclase:")
                # Mostrar más detalles si es necesario
            
            elif choice == "3" and results.get("Regresión"):
                print("\n📊 Detalles de Regresión:")
                # Mostrar más detalles si es necesario
            
            elif choice == "4" and results.get("Red Convolucional"):
                print("\n📊 Detalles de Red Convolucional:")
                # Mostrar más detalles si es necesario
            
            elif choice == "5" and results.get("Guardar/Cargar Modelo"):
                print("\n📊 Detalles de Guardar/Cargar Modelo:")
                # Mostrar más detalles si es necesario
            
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
