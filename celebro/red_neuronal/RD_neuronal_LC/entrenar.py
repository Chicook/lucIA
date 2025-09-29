"""
Módulo 3: Script Principal de Entrenamiento de Red Neuronal Profunda
===================================================================

Este módulo es el script principal que coordina el entrenamiento de la DNN.
Integra los módulos modelo.py y datos.py para realizar el proceso completo
de entrenamiento con las siguientes características:
- Carga de datos simulados
- Creación del modelo DNN
- Entrenamiento con 10 epochs y validation_split=0.2
- Visualización del resumen del modelo y métricas de entrenamiento

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import warnings

# Importar módulos locales
from modelo import crear_modelo, get_model_info, validate_model_architecture
from datos import cargar_datos_simulados, get_data_statistics, save_dataset_to_file

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configurar matplotlib para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DNNTrainer:
    """
    Clase para gestionar el entrenamiento de la Red Neuronal Profunda.
    
    Esta clase encapsula todo el proceso de entrenamiento incluyendo
    configuración de callbacks, monitoreo de métricas y generación
    de reportes detallados.
    """
    
    def __init__(self, epochs: int = 10, validation_split: float = 0.2, batch_size: int = 32):
        """
        Inicializa el entrenador de la DNN.
        
        Args:
            epochs (int): Número de épocas de entrenamiento (default: 10)
            validation_split (float): Proporción de datos para validación (default: 0.2)
            batch_size (int): Tamaño del lote (default: 32)
        """
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.training_start_time = None
        self.training_end_time = None
        
        # Crear directorio para logs y modelos
        self.create_directories()
        
        logger.info(f"Entrenador DNN inicializado: {epochs} epochs, validation_split={validation_split}")
    
    def create_directories(self):
        """Crea directorios necesarios para logs y modelos."""
        try:
            directories = ['logs', 'models', 'plots', 'reports']
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logger.info(f"Directorio creado: {directory}")
        except Exception as e:
            logger.error(f"Error al crear directorios: {str(e)}")
            raise
    
    def setup_callbacks(self) -> list:
        """
        Configura los callbacks para el entrenamiento.
        
        Returns:
            list: Lista de callbacks configurados
        """
        try:
            callbacks = []
            
            # Early Stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce Learning Rate on Plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Model Checkpoint
            model_checkpoint = ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
            
            logger.info(f"Callbacks configurados: {len(callbacks)} callbacks")
            return callbacks
            
        except Exception as e:
            logger.error(f"Error al configurar callbacks: {str(e)}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Entrena el modelo DNN con los datos proporcionados.
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento
            y_train (np.ndarray): Etiquetas de entrenamiento
            
        Returns:
            Dict[str, Any]: Diccionario con resultados del entrenamiento
        """
        try:
            logger.info("Iniciando entrenamiento del modelo...")
            self.training_start_time = datetime.now()
            
            # Crear modelo
            self.model = crear_modelo(
                input_shape=X_train.shape[1],
                num_classes=y_train.shape[1]
            )
            
            # Configurar callbacks
            callbacks = self.setup_callbacks()
            
            # Entrenar modelo
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            self.training_end_time = datetime.now()
            training_duration = self.training_end_time - self.training_start_time
            
            # Calcular métricas finales
            results = self.calculate_final_metrics()
            results['training_duration'] = str(training_duration)
            results['epochs_completed'] = len(self.history.history['loss'])
            
            logger.info(f"Entrenamiento completado en {training_duration}")
            return results
            
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    def calculate_final_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas finales del modelo entrenado.
        
        Returns:
            Dict[str, Any]: Diccionario con métricas finales
        """
        try:
            if self.history is None:
                return {}
            
            # Obtener métricas del último epoch
            final_metrics = {}
            for metric in self.history.history.keys():
                final_metrics[metric] = self.history.history[metric][-1]
            
            # Calcular métricas adicionales
            final_metrics['best_val_accuracy'] = max(self.history.history['val_accuracy'])
            final_metrics['best_val_loss'] = min(self.history.history['val_loss'])
            final_metrics['final_train_accuracy'] = self.history.history['accuracy'][-1]
            final_metrics['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
            
            logger.info(f"Métricas finales calculadas: val_accuracy={final_metrics.get('final_val_accuracy', 0):.4f}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error al calcular métricas finales: {str(e)}")
            return {}
    
    def plot_training_history(self, save_plots: bool = True) -> None:
        """
        Genera gráficos del historial de entrenamiento.
        
        Args:
            save_plots (bool): Si guardar los gráficos en archivos
        """
        try:
            if self.history is None:
                logger.warning("No hay historial de entrenamiento para graficar")
                return
            
            # Configurar subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Historial de Entrenamiento de la DNN', fontsize=16, fontweight='bold')
            
            # Gráfico 1: Loss
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss', color='blue', linewidth=2)
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
            axes[0, 0].set_title('Model Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Gráfico 2: Accuracy
            axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
            axes[0, 1].set_title('Model Accuracy', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Gráfico 3: Precision
            if 'precision' in self.history.history:
                axes[1, 0].plot(self.history.history['precision'], label='Training Precision', color='blue', linewidth=2)
                axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision', color='red', linewidth=2)
                axes[1, 0].set_title('Model Precision', fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Gráfico 4: Recall
            if 'recall' in self.history.history:
                axes[1, 1].plot(self.history.history['recall'], label='Training Recall', color='blue', linewidth=2)
                axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall', color='red', linewidth=2)
                axes[1, 1].set_title('Model Recall', fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
                logger.info("Gráficos de entrenamiento guardados en plots/training_history.png")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error al generar gráficos: {str(e)}")
    
    def generate_model_summary(self) -> str:
        """
        Genera un resumen detallado del modelo.
        
        Returns:
            str: Resumen del modelo en formato texto
        """
        try:
            if self.model is None:
                return "Modelo no disponible"
            
            summary_lines = []
            summary_lines.append("=" * 80)
            summary_lines.append("RESUMEN DEL MODELO DNN")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            
            # Información básica del modelo
            model_info = get_model_info(self.model)
            summary_lines.append(f"Total de parámetros: {model_info.get('total_params', 'N/A'):,}")
            summary_lines.append(f"Parámetros entrenables: {model_info.get('trainable_params', 'N/A'):,}")
            summary_lines.append(f"Parámetros no entrenables: {model_info.get('non_trainable_params', 'N/A'):,}")
            summary_lines.append(f"Número de capas: {model_info.get('layers_count', 'N/A')}")
            summary_lines.append("")
            
            # Arquitectura del modelo
            summary_lines.append("ARQUITECTURA:")
            summary_lines.append("-" * 40)
            self.model.summary(print_fn=lambda x: summary_lines.append(x))
            summary_lines.append("")
            
            # Validación de arquitectura
            is_valid = validate_model_architecture(self.model)
            summary_lines.append(f"Arquitectura válida: {'SÍ' if is_valid else 'NO'}")
            summary_lines.append("")
            
            summary = "\n".join(summary_lines)
            logger.info("Resumen del modelo generado")
            return summary
            
        except Exception as e:
            logger.error(f"Error al generar resumen del modelo: {str(e)}")
            return f"Error al generar resumen: {str(e)}"
    
    def save_training_report(self, results: Dict[str, Any], dataset_stats: Dict[str, Any]) -> None:
        """
        Guarda un reporte completo del entrenamiento.
        
        Args:
            results (Dict[str, Any]): Resultados del entrenamiento
            dataset_stats (Dict[str, Any]): Estadísticas del dataset
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"reports/training_report_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("REPORTE DE ENTRENAMIENTO DNN\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duración del entrenamiento: {results.get('training_duration', 'N/A')}\n")
                f.write(f"Épocas completadas: {results.get('epochs_completed', 'N/A')}\n\n")
                
                f.write("ESTADÍSTICAS DEL DATASET:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Muestras: {dataset_stats.get('n_samples', 'N/A')}\n")
                f.write(f"Características: {dataset_stats.get('n_features', 'N/A')}\n")
                f.write(f"Clases: {dataset_stats.get('n_classes', 'N/A')}\n")
                f.write(f"Distribución de clases: {dataset_stats.get('class_distribution', 'N/A')}\n\n")
                
                f.write("MÉTRICAS FINALES:\n")
                f.write("-" * 30 + "\n")
                for metric, value in results.items():
                    if metric not in ['training_duration', 'epochs_completed']:
                        f.write(f"{metric}: {value:.4f}\n")
                
                f.write("\n" + self.generate_model_summary())
            
            logger.info(f"Reporte guardado en {report_filename}")
            
        except Exception as e:
            logger.error(f"Error al guardar reporte: {str(e)}")


def main():
    """
    Función principal que ejecuta el proceso completo de entrenamiento.
    """
    try:
        print("=" * 80)
        print("ENTRENAMIENTO DE RED NEURONAL PROFUNDA (DNN)")
        print("=" * 80)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Paso 1: Cargar datos simulados
        print("1. Cargando datos simulados...")
        X_train, y_train = cargar_datos_simulados(n_samples=1000)
        print(f"   ✓ Datos cargados: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
        
        # Obtener estadísticas del dataset
        dataset_stats = get_data_statistics(X_train, y_train)
        print(f"   ✓ Distribución de clases: {dataset_stats['class_distribution']}")
        
        # Paso 2: Crear modelo
        print("\n2. Creando modelo DNN...")
        trainer = DNNTrainer(epochs=10, validation_split=0.2)
        print("   ✓ Entrenador inicializado")
        
        # Paso 3: Entrenar modelo
        print("\n3. Entrenando modelo...")
        results = trainer.train_model(X_train, y_train)
        print(f"   ✓ Entrenamiento completado en {results.get('training_duration', 'N/A')}")
        
        # Paso 4: Mostrar resumen del modelo
        print("\n4. Resumen del modelo:")
        print("-" * 50)
        model_summary = trainer.generate_model_summary()
        print(model_summary)
        
        # Paso 5: Mostrar métricas finales
        print("\n5. Métricas finales:")
        print("-" * 50)
        print(f"Precisión de entrenamiento: {results.get('final_train_accuracy', 0):.4f}")
        print(f"Precisión de validación: {results.get('final_val_accuracy', 0):.4f}")
        print(f"Mejor precisión de validación: {results.get('best_val_accuracy', 0):.4f}")
        print(f"Mejor pérdida de validación: {results.get('best_val_loss', 0):.4f}")
        
        # Paso 6: Generar gráficos
        print("\n6. Generando gráficos de entrenamiento...")
        trainer.plot_training_history(save_plots=True)
        print("   ✓ Gráficos generados y guardados")
        
        # Paso 7: Guardar reporte
        print("\n7. Guardando reporte de entrenamiento...")
        trainer.save_training_report(results, dataset_stats)
        print("   ✓ Reporte guardado")
        
        # Guardar dataset para referencia futura
        save_dataset_to_file(X_train, y_train, "models/dataset_entrenamiento.npz")
        
        print("\n" + "=" * 80)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    """
    Punto de entrada del script de entrenamiento.
    """
    success = main()
    
    if success:
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print("📊 Revisa los archivos generados:")
        print("   - plots/training_history.png (gráficos)")
        print("   - reports/training_report_*.txt (reporte)")
        print("   - models/best_model.h5 (modelo guardado)")
        print("   - models/dataset_entrenamiento.npz (dataset)")
        print("   - training.log (logs detallados)")
    else:
        print("\n❌ El entrenamiento falló. Revisa los logs para más detalles.")
        sys.exit(1)
