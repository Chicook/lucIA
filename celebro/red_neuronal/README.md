# @red_neuronal - Sistema de Redes Neuronales Completo

**Versión:** 0.6.0  
**Autor:** LucIA Development Team  
**Fecha:** 20 de Septiembre de 2025

## 🧠 Descripción

@red_neuronal es un sistema completo de redes neuronales implementado desde cero en Python, diseñado para ser modular, eficiente y fácil de usar. Forma parte del ecosistema @celebro y está integrado con LucIA.

## 🎯 Características Principales

### 🏗️ Arquitectura Modular
- **Capas personalizables** (Dense, Conv, Pooling, Dropout, BatchNorm)
- **Funciones de activación** avanzadas (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Swish, GELU)
- **Optimizadores** modernos (Adam, SGD, RMSprop, Adagrad, AdamW, AdaDelta, Nadam, RAdam)
- **Funciones de pérdida** completas (MSE, MAE, Huber, CrossEntropy, FocalLoss, DiceLoss)

### 🚀 Rendimiento Avanzado
- **Entrenamiento eficiente** con mini-lotes
- **Early stopping** automático
- **Batch normalization** para estabilización
- **Dropout** para regularización
- **Gradient clipping** para estabilidad
- **Learning rate scheduling** dinámico

### 🔧 Facilidad de Uso
- **API simple** e intuitiva
- **Configuración declarativa** con NetworkConfig
- **Guardado/carga** de modelos
- **Visualización** de métricas
- **Logging** detallado

## 📁 Estructura del Proyecto

```
@red_neuronal/
├── __init__.py                 # Inicialización del módulo
├── neural_network.py          # Red neuronal principal
├── layers.py                  # Implementación de capas
├── optimizers.py              # Algoritmos de optimización
├── loss_functions.py          # Funciones de pérdida
├── neural_core.py             # Núcleo central del sistema
├── training.py                # Sistema de entrenamiento
├── evaluation.py              # Sistema de evaluación
├── visualization.py           # Herramientas de visualización
├── models/                    # Modelos guardados
├── logs/                      # Logs del sistema
├── data/                      # Datos de ejemplo
└── README.md                  # Documentación
```

## 🚀 Instalación y Uso

### Requisitos
```bash
pip install numpy matplotlib scikit-learn
```

### Uso Básico
```python
from celebro.red_neuronal import NeuralCore, NetworkConfig, TrainingConfig

# Crear núcleo de red neuronal
neural_core = NeuralCore()

# Configurar red
config = NetworkConfig(
    input_size=784,
    hidden_layers=[128, 64, 32],
    output_size=10,
    activation='relu',
    output_activation='softmax',
    learning_rate=0.001,
    dropout_rate=0.2
)

# Crear red
network = neural_core.create_network(config)

# Configurar entrenamiento
training_config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping=True,
    patience=10
)

# Entrenar
history = neural_core.train(X_train, y_train, config=training_config)

# Evaluar
metrics = neural_core.evaluate(X_test, y_test)

# Predecir
predictions = neural_core.predict(X_new)
```

## 🏗️ Componentes Principales

### 1. NeuralCore
Núcleo central que coordina todos los componentes:

```python
neural_core = NeuralCore()

# Crear red
network = neural_core.create_network(config)

# Entrenar
history = neural_core.train(X_train, y_train)

# Evaluar
metrics = neural_core.evaluate(X_test, y_test)

# Guardar modelo
neural_core.save_model("modelo.pkl")

# Cargar modelo
neural_core.load_model("modelo.pkl")
```

### 2. Capas de Red Neuronal

#### DenseLayer (Capa Densa)
```python
from celebro.red_neuronal import DenseLayer

dense = DenseLayer(
    units=128,
    activation='relu',
    kernel_initializer='he_normal',
    use_bias=True
)
```

#### ConvLayer (Capa Convolucional)
```python
from celebro.red_neuronal import ConvLayer

conv = ConvLayer(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='valid',
    activation='relu'
)
```

#### DropoutLayer (Regularización)
```python
from celebro.red_neuronal import DropoutLayer

dropout = DropoutLayer(rate=0.5)
```

#### BatchNormLayer (Normalización)
```python
from celebro.red_neuronal import BatchNormLayer

batch_norm = BatchNormLayer(momentum=0.9, epsilon=1e-5)
```

### 3. Optimizadores

#### Adam (Recomendado)
```python
from celebro.red_neuronal import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)
```

#### SGD con Momentum
```python
from celebro.red_neuronal import SGD

optimizer = SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

#### RMSprop
```python
from celebro.red_neuronal import RMSprop

optimizer = RMSprop(
    learning_rate=0.001,
    rho=0.9,
    epsilon=1e-8
)
```

### 4. Funciones de Pérdida

#### CrossEntropy (Clasificación)
```python
from celebro.red_neuronal import CrossEntropy

loss = CrossEntropy(
    from_logits=False,
    label_smoothing=0.0
)
```

#### MSE (Regresión)
```python
from celebro.red_neuronal import MSE

loss = MSE()
```

#### Huber (Robusta)
```python
from celebro.red_neuronal import Huber

loss = Huber(delta=1.0)
```

#### FocalLoss (Desbalanceado)
```python
from celebro.red_neuronal import FocalLoss

loss = FocalLoss(alpha=1.0, gamma=2.0)
```

## 📊 Ejemplos de Uso

### Clasificación de Imágenes
```python
# Crear red convolucional
network = neural_core.create_convolutional_network(
    input_shape=(28, 28, 1),
    num_classes=10,
    config={
        'filters': [32, 64, 128],
        'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
        'pool_sizes': [(2, 2), (2, 2), (2, 2)],
        'dense_units': [128, 64],
        'dropout_rate': 0.5
    }
)

# Entrenar
history = neural_core.train(X_train, y_train)
```

### Regresión
```python
# Configurar para regresión
config = NetworkConfig(
    input_size=10,
    hidden_layers=[64, 32, 16],
    output_size=1,
    activation='relu',
    output_activation='linear',
    learning_rate=0.001
)

# Crear y entrenar
network = neural_core.create_network(config)
history = neural_core.train(X_train, y_train)
```

### Clasificación Binaria
```python
# Configurar para clasificación binaria
config = NetworkConfig(
    input_size=20,
    hidden_layers=[64, 32],
    output_size=2,
    activation='relu',
    output_activation='softmax',
    dropout_rate=0.3
)

# Crear y entrenar
network = neural_core.create_network(config)
history = neural_core.train(X_train, y_train)
```

## 🔧 Configuración Avanzada

### Learning Rate Scheduling
```python
from celebro.red_neuronal import LearningRateScheduler

# Crear scheduler
scheduler = LearningRateScheduler(
    optimizer=optimizer,
    schedule_type='cosine',
    T_max=100,
    eta_min=1e-6
)

# Usar en entrenamiento
for epoch in range(100):
    new_lr = scheduler.step(epoch)
    # Entrenar época...
```

### Gradient Clipping
```python
from celebro.red_neuronal import GradientClipping

# Crear clipper
clipper = GradientClipping(max_norm=1.0, norm_type='l2')

# Aplicar a gradientes
clipped_gradients = clipper.clip_gradients(gradients)
```

### Callbacks Personalizados
```python
class CustomCallback:
    def on_epoch_end(self, epoch, logs):
        print(f"Época {epoch}: {logs}")
    
    def on_training_end(self, logs):
        print("Entrenamiento completado")

# Usar callback
callbacks = [CustomCallback()]
training_config = TrainingConfig(callbacks=callbacks)
```

## 📈 Monitoreo y Visualización

### Métricas de Entrenamiento
```python
# Obtener historial
history = neural_core.training_history

# Acceder a métricas
for metrics in history:
    print(f"Época {metrics.epoch}:")
    print(f"  Pérdida: {metrics.train_loss:.4f}")
    print(f"  Precisión: {metrics.train_accuracy:.4f}")
    print(f"  Val. Pérdida: {metrics.val_loss:.4f}")
    print(f"  Val. Precisión: {metrics.val_accuracy:.4f}")
```

### Resumen de la Red
```python
# Obtener resumen de arquitectura
summary = neural_core.get_network_summary()
print(summary)

# Obtener resumen de entrenamiento
training_summary = neural_core.get_training_summary()
print(training_summary)
```

## 💾 Guardado y Carga de Modelos

### Guardar Modelo
```python
# Guardar modelo completo
neural_core.save_model("mi_modelo.pkl")

# El archivo incluye:
# - Arquitectura de la red
# - Parámetros entrenados
# - Historial de entrenamiento
# - Mejores pesos
# - Configuración
```

### Cargar Modelo
```python
# Crear nueva instancia
new_neural_core = NeuralCore()

# Cargar modelo
new_neural_core.load_model("mi_modelo.pkl")

# Usar inmediatamente
predictions = new_neural_core.predict(X_new)
```

## 🧪 Testing y Validación

### Ejecutar Demostración
```bash
python demo_red_neuronal.py
```

### Tests Unitarios
```python
# Ejecutar tests específicos
python -m pytest celebro/red_neuronal/tests/
```

### Benchmark de Rendimiento
```python
# Ejecutar benchmark
python celebro/red_neuronal/benchmark.py
```

## 🔍 Casos de Uso

### 1. **Clasificación de Imágenes**
- Reconocimiento de dígitos (MNIST)
- Clasificación de objetos (CIFAR-10)
- Detección de rostros
- Análisis médico de imágenes

### 2. **Procesamiento de Texto**
- Clasificación de sentimientos
- Análisis de spam
- Categorización de documentos
- Traducción automática

### 3. **Análisis de Series Temporales**
- Predicción de precios
- Análisis de tendencias
- Detección de anomalías
- Pronósticos meteorológicos

### 4. **Sistemas de Recomendación**
- Filtrado colaborativo
- Filtrado basado en contenido
- Sistemas híbridos
- Personalización

### 5. **Análisis de Datos**
- Regresión multivariable
- Clasificación multiclase
- Clustering
- Reducción de dimensionalidad

## ⚡ Optimizaciones de Rendimiento

### 1. **Vectorización**
- Uso eficiente de NumPy
- Operaciones vectorizadas
- Minimización de bucles Python

### 2. **Gestión de Memoria**
- Liberación automática de memoria
- Reutilización de buffers
- Gestión eficiente de gradientes

### 3. **Paralelización**
- Entrenamiento por lotes
- Procesamiento paralelo de capas
- Optimización de operaciones matriciales

### 4. **Caching**
- Cache de activaciones
- Reutilización de cálculos
- Optimización de acceso a datos

## 🛠️ Desarrollo y Extensión

### Agregar Nueva Capa
```python
class MiCapaPersonalizada(Layer):
    def __init__(self, parametros, name=None):
        super().__init__(name)
        # Inicializar parámetros
    
    def initialize_parameters(self, input_shape):
        # Inicializar parámetros de la capa
        pass
    
    def forward(self, inputs):
        # Implementar propagación hacia adelante
        return output
    
    def backward(self, gradients):
        # Implementar propagación hacia atrás
        return input_gradients
```

### Agregar Nuevo Optimizador
```python
class MiOptimizador(Optimizer):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__(learning_rate)
        # Inicializar parámetros del optimizador
    
    def update(self, parameters, gradients):
        # Implementar algoritmo de actualización
        return updated_parameters
```

### Agregar Nueva Función de Pérdida
```python
class MiFuncionPérdida(LossFunction):
    def __init__(self, **kwargs):
        super().__init__()
        # Inicializar parámetros
    
    def compute(self, y_true, y_pred):
        # Calcular pérdida
        return loss
    
    def gradient(self, y_true, y_pred):
        # Calcular gradiente
        return gradients
```

## 📊 Métricas y Monitoreo

### Métricas de Entrenamiento
- **Pérdida de entrenamiento**: Error en datos de entrenamiento
- **Pérdida de validación**: Error en datos de validación
- **Precisión**: Porcentaje de predicciones correctas
- **F1-Score**: Media armónica de precisión y recall
- **AUC-ROC**: Área bajo la curva ROC

### Métricas de Regresión
- **MSE**: Error cuadrático medio
- **RMSE**: Raíz del error cuadrático medio
- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación
- **MAPE**: Error porcentual absoluto medio

### Monitoreo en Tiempo Real
- **Logging detallado** de cada época
- **Métricas en tiempo real** durante entrenamiento
- **Alertas automáticas** por overfitting
- **Visualización** de curvas de aprendizaje

## 🔒 Consideraciones de Seguridad

### Validación de Entrada
- **Verificación de tipos** de datos
- **Validación de rangos** de parámetros
- **Sanitización** de entradas del usuario
- **Límites de memoria** y procesamiento

### Protección de Modelos
- **Cifrado** de modelos guardados
- **Verificación de integridad** de archivos
- **Control de acceso** a modelos
- **Auditoría** de uso de modelos

### Privacidad de Datos
- **Anonimización** de datos sensibles
- **Cifrado** de datos en tránsito
- **Eliminación segura** de datos temporales
- **Cumplimiento** de regulaciones de privacidad

## 🚀 Roadmap Futuro

### Versión 0.7.0
- [ ] **GPU Support** con CUDA
- [ ] **Distributed Training** multi-GPU
- [ ] **AutoML** para selección automática de hiperparámetros
- [ ] **Neural Architecture Search** (NAS)

### Versión 0.8.0
- [ ] **Transfer Learning** pre-entrenado
- [ ] **Federated Learning** distribuido
- [ ] **Quantization** para modelos ligeros
- [ ] **Pruning** de redes neuronales

### Versión 0.9.0
- [ ] **Reinforcement Learning** integrado
- [ ] **Graph Neural Networks** (GNN)
- [ ] **Transformers** y atención
- [ ] **Generative Models** (GANs, VAEs)

### Versión 1.0.0
- [ ] **Production Ready** con optimizaciones
- [ ] **REST API** para servicios
- [ ] **Docker** containers
- [ ] **Kubernetes** orchestration

## 📞 Soporte y Contribución

### Reportar Issues
- Usar el sistema de issues del repositorio
- Incluir logs y contexto detallado
- Especificar versión y configuración

### Contribuir
- Fork del repositorio
- Crear rama para feature
- Pull request con descripción detallada
- Tests y documentación incluidos

### Documentación
- Mantener README actualizado
- Documentar cambios en CHANGELOG
- Incluir ejemplos de uso
- Tutoriales paso a paso

---

**@red_neuronal** - El sistema de redes neuronales más completo y modular para LucIA. 🧠⚡

*Desarrollado con ❤️ por el equipo de LucIA Development*
