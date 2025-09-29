"""
Módulo 2: Preparación de Datos Simulados para Red Neuronal Profunda
=================================================================

Este módulo contiene funciones para generar y preparar conjuntos de datos
simulados para el entrenamiento de la DNN. Los datos generados incluyen:
- 8 características de entrada con patrones realistas
- 4 clases de salida codificadas en one-hot
- 1000 muestras de entrenamiento con distribución balanceada

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import logging
from typing import Tuple, Optional, Dict, Any
import warnings
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    """
    Clase para generar y gestionar datos simulados para la DNN.
    
    Esta clase proporciona métodos para crear datasets sintéticos
    con patrones realistas y distribuciones apropiadas para el
    entrenamiento de redes neuronales profundas.
    """
    
    def __init__(self, n_samples: int = 1000, n_features: int = 8, n_classes: int = 4):
        """
        Inicializa el generador de datos.
        
        Args:
            n_samples (int): Número de muestras a generar (default: 1000)
            n_features (int): Número de características (default: 8)
            n_classes (int): Número de clases (default: 4)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = 42
        
        logger.info(f"Generador de datos inicializado: {n_samples} muestras, {n_features} características, {n_classes} clases")
    
    def _generate_realistic_features(self) -> np.ndarray:
        """
        Genera características realistas con patrones complejos.
        
        Returns:
            np.ndarray: Matriz de características (n_samples, n_features)
        """
        try:
            np.random.seed(self.random_state)
            
            # Característica 1: Distribución normal con outliers
            feature_1 = np.random.normal(50, 15, self.n_samples)
            outliers = np.random.choice(self.n_samples, size=50, replace=False)
            feature_1[outliers] += np.random.normal(0, 30, 50)
            
            # Característica 2: Distribución exponencial
            feature_2 = np.random.exponential(scale=25, size=self.n_samples)
            
            # Característica 3: Distribución uniforme
            feature_3 = np.random.uniform(0, 100, self.n_samples)
            
            # Característica 4: Distribución gamma
            feature_4 = np.random.gamma(shape=2, scale=10, size=self.n_samples)
            
            # Característica 5: Correlacionada con feature_1
            feature_5 = 0.7 * feature_1 + np.random.normal(0, 5, self.n_samples)
            
            # Característica 6: Distribución log-normal
            feature_6 = np.random.lognormal(mean=3, sigma=0.5, size=self.n_samples)
            
            # Característica 7: Distribución beta
            feature_7 = np.random.beta(a=2, b=5, size=self.n_samples) * 100
            
            # Característica 8: Combinación de distribuciones
            feature_8 = 0.3 * feature_2 + 0.4 * feature_4 + np.random.normal(0, 8, self.n_samples)
            
            # Combinar todas las características
            features = np.column_stack([
                feature_1, feature_2, feature_3, feature_4,
                feature_5, feature_6, feature_7, feature_8
            ])
            
            logger.info(f"Características generadas: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error al generar características: {str(e)}")
            raise
    
    def _generate_realistic_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Genera etiquetas realistas basadas en las características.
        
        Args:
            features (np.ndarray): Matriz de características
            
        Returns:
            np.ndarray: Array de etiquetas (n_samples,)
        """
        try:
            labels = np.zeros(self.n_samples, dtype=int)
            
            # Clase 0: Características 1 y 5 altas
            mask_0 = (features[:, 0] > np.percentile(features[:, 0], 70)) & \
                    (features[:, 4] > np.percentile(features[:, 4], 60))
            labels[mask_0] = 0
            
            # Clase 1: Características 2 y 6 altas
            mask_1 = (features[:, 1] > np.percentile(features[:, 1], 65)) & \
                    (features[:, 5] > np.percentile(features[:, 5], 70))
            labels[mask_1] = 1
            
            # Clase 2: Características 3 y 7 bajas
            mask_2 = (features[:, 2] < np.percentile(features[:, 2], 40)) & \
                    (features[:, 6] < np.percentile(features[:, 6], 45))
            labels[mask_2] = 2
            
            # Clase 3: Características 4 y 8 medias
            mask_3 = (features[:, 3] > np.percentile(features[:, 3], 30)) & \
                    (features[:, 3] < np.percentile(features[:, 3], 70)) & \
                    (features[:, 7] > np.percentile(features[:, 7], 40)) & \
                    (features[:, 7] < np.percentile(features[:, 7], 80))
            labels[mask_3] = 3
            
            # Asignar clases restantes aleatoriamente
            unassigned = labels == 0
            remaining_classes = np.random.choice(
                range(self.n_classes), 
                size=np.sum(unassigned), 
                replace=True
            )
            labels[unassigned] = remaining_classes
            
            # Asegurar que todas las clases estén representadas
            for class_id in range(self.n_classes):
                if np.sum(labels == class_id) == 0:
                    # Asignar algunas muestras aleatorias a esta clase
                    random_indices = np.random.choice(
                        self.n_samples, 
                        size=max(50, self.n_samples // 20), 
                        replace=False
                    )
                    labels[random_indices] = class_id
            
            logger.info(f"Etiquetas generadas: distribución {np.bincount(labels)}")
            return labels
            
        except Exception as e:
            logger.error(f"Error al generar etiquetas: {str(e)}")
            raise
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normaliza las características usando StandardScaler.
        
        Args:
            features (np.ndarray): Matriz de características
            
        Returns:
            np.ndarray: Características normalizadas
        """
        try:
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            logger.info("Características normalizadas exitosamente")
            return features_normalized
            
        except Exception as e:
            logger.error(f"Error al normalizar características: {str(e)}")
            raise
    
    def _encode_labels_one_hot(self, labels: np.ndarray) -> np.ndarray:
        """
        Codifica las etiquetas en formato one-hot.
        
        Args:
            labels (np.ndarray): Array de etiquetas
            
        Returns:
            np.ndarray: Matriz de etiquetas codificadas en one-hot
        """
        try:
            # Usar pandas para codificación one-hot
            df_labels = pd.DataFrame(labels, columns=['class'])
            one_hot = pd.get_dummies(df_labels['class'], prefix='class')
            one_hot_array = one_hot.values.astype(np.float32)
            
            logger.info(f"Etiquetas codificadas en one-hot: {one_hot_array.shape}")
            return one_hot_array
            
        except Exception as e:
            logger.error(f"Error al codificar etiquetas: {str(e)}")
            raise
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera el dataset completo con características y etiquetas.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_train, y_train)
        """
        try:
            logger.info("Iniciando generación del dataset...")
            
            # Generar características realistas
            features = self._generate_realistic_features()
            
            # Generar etiquetas realistas
            labels = self._generate_realistic_labels(features)
            
            # Normalizar características
            features_normalized = self._normalize_features(features)
            
            # Codificar etiquetas en one-hot
            labels_one_hot = self._encode_labels_one_hot(labels)
            
            logger.info("Dataset generado exitosamente")
            return features_normalized, labels_one_hot
            
        except Exception as e:
            logger.error(f"Error al generar dataset: {str(e)}")
            raise


def cargar_datos_simulados(n_samples: int = 1000, n_features: int = 8, n_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función principal para cargar datos simulados para la DNN.
    
    Esta función genera un dataset sintético con patrones realistas
    que simula un problema de clasificación multiclase. Los datos
    incluyen características correlacionadas y distribuciones variadas
    para crear un conjunto de datos desafiante pero realista.
    
    Args:
        n_samples (int): Número de muestras a generar (default: 1000)
        n_features (int): Número de características (default: 8)
        n_classes (int): Número de clases (default: 4)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_train, y_train)
            - X_train: Matriz de características normalizadas (n_samples, n_features)
            - y_train: Matriz de etiquetas codificadas en one-hot (n_samples, n_classes)
    
    Raises:
        ValueError: Si los parámetros de entrada son inválidos
        RuntimeError: Si hay error en la generación de datos
        
    Example:
        >>> X_train, y_train = cargar_datos_simulados(n_samples=1000)
        >>> print(f"Forma de X_train: {X_train.shape}")
        >>> print(f"Forma de y_train: {y_train.shape}")
    """
    try:
        # Validar parámetros de entrada
        if n_samples <= 0:
            raise ValueError("n_samples debe ser un número positivo")
        if n_features <= 0:
            raise ValueError("n_features debe ser un número positivo")
        if n_classes <= 0:
            raise ValueError("n_classes debe ser un número positivo")
        
        logger.info(f"Generando datos simulados: {n_samples} muestras, {n_features} características, {n_classes} clases")
        
        # Crear generador de datos
        data_generator = DataGenerator(n_samples, n_features, n_classes)
        
        # Generar dataset
        X_train, y_train = data_generator.generate_dataset()
        
        # Validar formas de salida
        if X_train.shape != (n_samples, n_features):
            raise RuntimeError(f"Forma incorrecta de X_train: esperado {(n_samples, n_features)}, obtenido {X_train.shape}")
        
        if y_train.shape != (n_samples, n_classes):
            raise RuntimeError(f"Forma incorrecta de y_train: esperado {(n_samples, n_classes)}, obtenido {y_train.shape}")
        
        logger.info("Datos simulados cargados exitosamente")
        return X_train, y_train
        
    except ValueError as ve:
        logger.error(f"Error de validación: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado al cargar datos: {str(e)}")
        raise RuntimeError(f"Error al cargar datos simulados: {str(e)}")


def get_data_statistics(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Obtiene estadísticas detalladas del dataset generado.
    
    Args:
        X_train (np.ndarray): Matriz de características
        y_train (np.ndarray): Matriz de etiquetas codificadas en one-hot
        
    Returns:
        Dict[str, Any]: Diccionario con estadísticas del dataset
    """
    try:
        # Convertir one-hot a etiquetas numéricas
        labels = np.argmax(y_train, axis=1)
        
        stats = {
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_classes': y_train.shape[1],
            'feature_means': np.mean(X_train, axis=0).tolist(),
            'feature_stds': np.std(X_train, axis=0).tolist(),
            'class_distribution': np.bincount(labels).tolist(),
            'class_percentages': (np.bincount(labels) / len(labels) * 100).tolist(),
            'missing_values': np.isnan(X_train).sum(),
            'data_range': {
                'min': float(np.min(X_train)),
                'max': float(np.max(X_train))
            }
        }
        
        logger.info(f"Estadísticas calculadas: {stats['n_samples']} muestras, {stats['n_classes']} clases")
        return stats
        
    except Exception as e:
        logger.error(f"Error al calcular estadísticas: {str(e)}")
        return {}


def save_dataset_to_file(X_train: np.ndarray, y_train: np.ndarray, filename: str = "dataset_simulado.npz") -> bool:
    """
    Guarda el dataset generado en un archivo NumPy.
    
    Args:
        X_train (np.ndarray): Matriz de características
        y_train (np.ndarray): Matriz de etiquetas
        filename (str): Nombre del archivo de salida
        
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
    """
    try:
        np.savez_compressed(
            filename,
            X_train=X_train,
            y_train=y_train,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Dataset guardado en {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar dataset: {str(e)}")
        return False


def load_dataset_from_file(filename: str = "dataset_simulado.npz") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Carga un dataset desde un archivo NumPy.
    
    Args:
        filename (str): Nombre del archivo a cargar
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (X_train, y_train) o None si hay error
    """
    try:
        data = np.load(filename)
        X_train = data['X_train']
        y_train = data['y_train']
        
        logger.info(f"Dataset cargado desde {filename}")
        return X_train, y_train
        
    except Exception as e:
        logger.error(f"Error al cargar dataset: {str(e)}")
        return None


if __name__ == "__main__":
    """
    Script de prueba para el módulo datos.py
    """
    try:
        print("=" * 60)
        print("PRUEBA DEL MÓDULO DATOS.PY")
        print("=" * 60)
        
        # Cargar datos simulados
        X_train, y_train = cargar_datos_simulados(n_samples=1000)
        
        # Mostrar información básica
        print(f"\nINFORMACIÓN DEL DATASET:")
        print(f"Forma de X_train: {X_train.shape}")
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Tipo de datos X_train: {X_train.dtype}")
        print(f"Tipo de datos y_train: {y_train.dtype}")
        
        # Mostrar estadísticas
        stats = get_data_statistics(X_train, y_train)
        print(f"\nESTADÍSTICAS DETALLADAS:")
        print(f"Número de muestras: {stats['n_samples']}")
        print(f"Número de características: {stats['n_features']}")
        print(f"Número de clases: {stats['n_classes']}")
        print(f"Distribución de clases: {stats['class_distribution']}")
        print(f"Porcentajes de clases: {[f'{p:.1f}%' for p in stats['class_percentages']]}")
        print(f"Valores faltantes: {stats['missing_values']}")
        print(f"Rango de datos: [{stats['data_range']['min']:.2f}, {stats['data_range']['max']:.2f}]")
        
        # Probar guardar y cargar
        save_success = save_dataset_to_file(X_train, y_train, "test_dataset.npz")
        print(f"\nGuardado exitoso: {'SÍ' if save_success else 'NO'}")
        
        if save_success:
            loaded_data = load_dataset_from_file("test_dataset.npz")
            if loaded_data is not None:
                X_loaded, y_loaded = loaded_data
                print(f"Datos cargados correctamente: X={X_loaded.shape}, y={y_loaded.shape}")
            else:
                print("Error al cargar datos guardados")
        
        print("\n" + "=" * 60)
        print("PRUEBA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error en la prueba: {str(e)}")
        logger.error(f"Error en prueba del módulo: {str(e)}")
