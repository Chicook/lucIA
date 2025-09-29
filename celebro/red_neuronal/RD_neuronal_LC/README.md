# Red Neuronal Profunda (DNN) - Módulo LucIA

## Descripción General

Este módulo implementa una Red Neuronal Profunda (DNN) completamente conectada usando TensorFlow/Keras, diseñada específicamente para el proyecto LucIA. La arquitectura sigue las mejores prácticas de deep learning con una estructura modular y escalable.

## Arquitectura de la Red

```
Entrada (8 características) 
    ↓
Capa Densa (32 neuronas, ReLU)
    ↓
Capa Densa (16 neuronas, ReLU)
    ↓
Capa Densa (8 neuronas, ReLU)
    ↓
Salida (4 neuronas, Softmax)
```

## Estructura de Archivos

```
RD_neuronal_LC/
├── modelo.py          # Definición de la arquitectura DNN
├── datos.py           # Generación de datos simulados
├── entrenar.py        # Script principal de entrenamiento
├── requirements.txt   # Dependencias del proyecto
└── README.md         # Este archivo
```

## Características Principales

### 🧠 **Arquitectura Avanzada**
- **3 capas ocultas** con tamaños decrecientes (32 → 16 → 8)
- **Activación ReLU** en capas ocultas para evitar vanishing gradient
- **Activación Softmax** en salida para clasificación multiclase
- **Optimizador Adam** con configuración optimizada

### 📊 **Datos Simulados Realistas**
- **8 características** con distribuciones variadas (normal, exponencial, gamma, etc.)
- **4 clases** con patrones complejos y correlaciones
- **1000 muestras** balanceadas para entrenamiento
- **Normalización automática** de características

### 🔧 **Entrenamiento Robusto**
- **10 épocas** de entrenamiento con validación
- **20% de datos** reservados para validación
- **Callbacks avanzados**: Early Stopping, Reduce LR, Model Checkpoint
- **Métricas completas**: Accuracy, Precision, Recall

### 📈 **Monitoreo y Visualización**
- **Gráficos automáticos** del historial de entrenamiento
- **Reportes detallados** con métricas y estadísticas
- **Logs completos** de todo el proceso
- **Guardado automático** del mejor modelo

## Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Verificar Instalación

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import keras; print(f'Keras: {keras.__version__}')"
```

## Uso Rápido

### Ejecutar Entrenamiento Completo

```bash
python entrenar.py
```

### Uso Modular

```python
# Importar módulos
from modelo import crear_modelo
from datos import cargar_datos_simulados

# Cargar datos
X_train, y_train = cargar_datos_simulados()

# Crear modelo
model = crear_modelo(input_shape=8, num_classes=4)

# Mostrar resumen
model.summary()
```

## Ejemplos de Salida

### Resumen del Modelo

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_layer (Dense)         (None, 32)                288       
                                                                 
 hidden_layer_1 (Dense)      (None, 16)                528       
                                                                 
 hidden_layer_2 (Dense)      (None, 8)                 136       
                                                                 
 output_layer (Dense)        (None, 4)                 36        
                                                                 
=================================================================
Total params: 988
Trainable params: 988
Non-trainable params: 0
_________________________________________________________________
```

### Métricas Típicas

```
Precisión de entrenamiento: 0.8750
Precisión de validación: 0.8250
Mejor precisión de validación: 0.8500
Mejor pérdida de validación: 0.4123
```

## Configuración Avanzada

### Personalizar Arquitectura

```python
# Modificar tamaños de capas ocultas
dnn_arch = DNNArchitecture(input_shape=8, num_classes=4)
dnn_arch.hidden_layers_config = [64, 32, 16]  # Capas más grandes
model = dnn_arch.create_model()
```

### Personalizar Datos

```python
# Generar más muestras
X_train, y_train = cargar_datos_simulados(
    n_samples=5000,  # Más datos
    n_features=8,
    n_classes=4
)
```

### Personalizar Entrenamiento

```python
# Entrenamiento con más épocas
trainer = DNNTrainer(
    epochs=50,           # Más épocas
    validation_split=0.3, # Más validación
    batch_size=64        # Lotes más grandes
)
```

## Archivos Generados

Después del entrenamiento se generan los siguientes archivos:

```
plots/
├── training_history.png    # Gráficos de entrenamiento

models/
├── best_model.h5          # Mejor modelo guardado
└── dataset_entrenamiento.npz  # Dataset usado

reports/
└── training_report_YYYYMMDD_HHMMSS.txt  # Reporte detallado

logs/
└── training.log           # Logs del entrenamiento
```

## Monitoreo de Rendimiento

### Métricas Clave

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase (macro)
- **Recall**: Sensibilidad por clase (macro)
- **Loss**: Función de pérdida (categorical crossentropy)

### Indicadores de Overfitting

- Diferencia grande entre training y validation accuracy
- Validation loss que aumenta mientras training loss disminuye
- Early stopping activado frecuentemente

## Solución de Problemas

### Error: "CUDA out of memory"

```python
# Reducir batch size
trainer = DNNTrainer(batch_size=16)  # En lugar de 32
```

### Error: "Model not converging"

```python
# Aumentar learning rate o épocas
optimizer = Adam(learning_rate=0.01)  # En lugar de 0.001
```

### Error: "Data shape mismatch"

```python
# Verificar formas de datos
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
```

## Contribución

Para contribuir al módulo:

1. **Fork** el repositorio
2. **Crear** una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Crear** un Pull Request

## Licencia

Este módulo es parte del proyecto LucIA y está sujeto a la licencia del proyecto principal.

## Contacto

- **Equipo**: LucIA Development Team
- **Versión**: 1.0.0
- **Fecha**: 2025-01-11

---

## Changelog

### v1.0.0 (2025-01-11)
- ✅ Implementación inicial de DNN
- ✅ Módulos separados (modelo, datos, entrenar)
- ✅ Generación de datos simulados realistas
- ✅ Sistema de callbacks avanzado
- ✅ Visualización y reportes automáticos
- ✅ Documentación completa

---

**🎯 Objetivo**: Proporcionar una base sólida y escalable para el desarrollo de redes neuronales profundas en el ecosistema LucIA, siguiendo las mejores prácticas de machine learning y deep learning.
