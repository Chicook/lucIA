# 🚀 Inicio Rápido - WoldVirtual3DlucIA

## ❌ Problema Detectado
El error `FileNotFoundError: [Errno 2] No such file or directory: 'logs/lucIA_core.log'` indica que faltan directorios.

## ✅ Solución Rápida

### Opción 1: Instalación Básica (Recomendada)
```bash
python install_basic.py
python run_lucia.py
```

### Opción 2: Crear Directorios Manualmente
```bash
# Crear directorios necesarios
mkdir logs
mkdir data
mkdir models
mkdir cache
mkdir temp
mkdir config

# Ejecutar LucIA
python main.py
```

### Opción 3: Instalación Completa
```bash
python setup_lucia.py
python start_lucia.py
```

## 🔧 Solución del Error

El problema es que el sistema intenta escribir logs en `logs/lucIA_core.log` pero el directorio `logs` no existe.

**Archivo corregido**: `main.py` ahora crea automáticamente el directorio `logs` antes de configurar el logging.

## 📋 Verificación

Después de ejecutar cualquiera de las opciones anteriores, deberías ver:

```
🤖 WoldVirtual3DlucIA - Motor de IA Modular v0.6.0
============================================================
Iniciando sistema de inteligencia artificial...

🧠 Módulo 1: Sistema de Memoria Persistente
   ✅ Módulo inicializado correctamente

🧠 Módulo 2: Sistema de Aprendizaje y Adaptación
   ✅ Módulo inicializado correctamente

... (más módulos)
```

## 🆘 Si Aún Hay Problemas

1. **Verificar Python**: `python --version` (debe ser 3.8+)
2. **Instalar dependencias**: `pip install numpy aiohttp requests`
3. **Ejecutar test**: `python test_lucia_system.py`

## 📞 Soporte

- Revisar logs en `logs/lucIA_core.log`
- Ejecutar `python install_basic.py` para diagnóstico
- Ver `README.md` para documentación completa
