"""
Neurona de Entrada 01 - Módulo Independiente
===========================================

Primera neurona de la capa de entrada de la red neuronal modular.
Esta neurona recibe datos externos y los procesa de manera independiente,
comunicándose con las neuronas de la primera capa oculta.

Características:
- Tipo: INPUT
- Índice: 01
- Activación: LINEAR (sin transformación)
- Conexiones: 10 neuronas de hidden layer 1

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025-01-11
"""

import sys
import os
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional
import json
import time
from datetime import datetime

# Agregar el directorio padre al path para importar neurona_base
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neurona_base import (
    BaseNeuron, InputNeuron, NeuronConfig, NeuronType, 
    ActivationType, Connection, MessageQueue
)

# Configuración específica de esta neurona
class NeuronaInput01Config(NeuronConfig):
    """Configuración específica para la neurona de entrada 01"""
    
    def __init__(self):
        super().__init__(
            neuron_id="input_01",
            neuron_type=NeuronType.INPUT,
            layer_index=0,
            neuron_index=1,
            activation=ActivationType.LINEAR,
            bias=0.0,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0,
            max_connections=10,
            processing_timeout=1.0
        )
        
        # Configuraciones específicas
        self.input_range = (-1.0, 1.0)  # Rango esperado de entrada
        self.normalization_factor = 1.0  # Factor de normalización
        self.data_type = "continuous"    # Tipo de dato que procesa

class NeuronaInput01(InputNeuron):
    """
    Neurona de entrada 01 - Procesa la primera característica de entrada.
    
    Esta neurona es responsable de:
    - Recibir datos externos (primera característica)
    - Normalizar y validar los datos de entrada
    - Transmitir los datos procesados a la primera capa oculta
    - Mantener métricas de calidad de datos
    """
    
    def __init__(self):
        """Inicializa la neurona de entrada 01"""
        config = NeuronaInput01Config()
        super().__init__(config)
        
        # Métricas específicas de entrada
        self.input_metrics = {
            'total_inputs': 0,
            'valid_inputs': 0,
            'invalid_inputs': 0,
            'min_input': float('inf'),
            'max_input': float('-inf'),
            'average_input': 0.0,
            'input_variance': 0.0,
            'last_input_time': 0.0
        }
        
        # Estado específico
        self.current_external_input = None
        self.is_receiving_data = False
        self.data_quality_score = 1.0
        
        logger.info(f"Neurona de entrada 01 inicializada - ID: {self.config.neuron_id}")
    
    async def receive_external_data(self, data: float) -> bool:
        """
        Recibe datos externos y los procesa.
        
        Args:
            data (float): Dato de entrada externo
            
        Returns:
            bool: True si el dato fue procesado exitosamente
        """
        try:
            self.is_receiving_data = True
            start_time = time.time()
            
            # Validar y normalizar el dato
            processed_data = await self._validate_and_normalize_input(data)
            
            if processed_data is not None:
                # Actualizar métricas
                self._update_input_metrics(data, processed_data)
                
                # Procesar con la neurona base
                output = await self.process_input(processed_data, source_neuron="external")
                
                self.input_metrics['last_input_time'] = time.time()
                self.is_receiving_data = False
                
                logger.debug(f"Neurona {self.config.neuron_id} procesó entrada: {data} -> {output:.4f}")
                return True
            else:
                self.input_metrics['invalid_inputs'] += 1
                self.is_receiving_data = False
                logger.warning(f"Dato inválido recibido en neurona {self.config.neuron_id}: {data}")
                return False
                
        except Exception as e:
            self.is_receiving_data = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo datos en neurona {self.config.neuron_id}: {e}")
            return False
    
    async def _validate_and_normalize_input(self, data: float) -> Optional[float]:
        """
        Valida y normaliza el dato de entrada.
        
        Args:
            data (float): Dato a validar y normalizar
            
        Returns:
            Optional[float]: Dato normalizado o None si es inválido
        """
        try:
            # Verificar que sea un número válido
            if not isinstance(data, (int, float)) or np.isnan(data) or np.isinf(data):
                return None
            
            # Verificar rango esperado (opcional, puede ser flexible)
            min_range, max_range = self.config.input_range
            if data < min_range * 10 or data > max_range * 10:  # Permitir cierta flexibilidad
                logger.warning(f"Dato fuera de rango esperado: {data} (rango: {min_range}-{max_range})")
                # No rechazar, solo advertir
            
            # Normalizar usando el factor de normalización
            normalized_data = data * self.config.normalization_factor
            
            return float(normalized_data)
            
        except Exception as e:
            logger.error(f"Error validando entrada: {e}")
            return None
    
    def _update_input_metrics(self, raw_input: float, processed_input: float):
        """Actualiza las métricas específicas de entrada"""
        self.input_metrics['total_inputs'] += 1
        self.input_metrics['valid_inputs'] += 1
        
        # Actualizar min/max
        self.input_metrics['min_input'] = min(self.input_metrics['min_input'], raw_input)
        self.input_metrics['max_input'] = max(self.input_metrics['max_input'], raw_input)
        
        # Actualizar promedio
        total = self.input_metrics['total_inputs']
        current_avg = self.input_metrics['average_input']
        self.input_metrics['average_input'] = ((current_avg * (total - 1)) + raw_input) / total
        
        # Calcular varianza (simplificado)
        if total > 1:
            variance = ((raw_input - self.input_metrics['average_input']) ** 2) / total
            self.input_metrics['input_variance'] = ((self.input_metrics['input_variance'] * (total - 1)) + variance) / total
    
    def get_input_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de entrada"""
        return {
            'neuron_id': self.config.neuron_id,
            'input_metrics': self.input_metrics,
            'data_quality_score': self.data_quality_score,
            'is_receiving_data': self.is_receiving_data,
            'current_external_input': self.current_external_input,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_input_metrics(self):
        """Resetea las métricas de entrada"""
        self.input_metrics = {
            'total_inputs': 0,
            'valid_inputs': 0,
            'invalid_inputs': 0,
            'min_input': float('inf'),
            'max_input': float('-inf'),
            'average_input': 0.0,
            'input_variance': 0.0,
            'last_input_time': 0.0
        }
        logger.info(f"Métricas de entrada reseteadas para neurona {self.config.neuron_id}")
    
    async def set_normalization_factor(self, factor: float):
        """Establece un nuevo factor de normalización"""
        if factor > 0:
            self.config.normalization_factor = factor
            logger.info(f"Factor de normalización actualizado: {factor}")
        else:
            logger.warning(f"Factor de normalización inválido: {factor}")
    
    async def set_input_range(self, min_val: float, max_val: float):
        """Establece el rango esperado de entrada"""
        if min_val < max_val:
            self.config.input_range = (min_val, max_val)
            logger.info(f"Rango de entrada actualizado: {min_val} - {max_val}")
        else:
            logger.warning(f"Rango de entrada inválido: {min_val} - {max_val}")

# Función principal para ejecutar la neurona como módulo independiente
async def main():
    """Función principal para ejecutar la neurona como proceso independiente"""
    print("=" * 60)
    print("NEURONA DE ENTRADA 01 - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        # Crear instancia de la neurona
        neurona = NeuronaInput01()
        
        # Inicializar
        await neurona.initialize()
        
        print(f"✅ Neurona {neurona.config.neuron_id} inicializada")
        print(f"   Tipo: {neurona.config.neuron_type.value}")
        print(f"   Activación: {neurona.config.activation.value}")
        print(f"   Rango de entrada: {neurona.config.input_range}")
        
        # Simular recepción de datos
        print("\n📊 Simulando recepción de datos...")
        
        test_data = [0.5, 1.2, -0.3, 0.8, 2.1, -1.5, 0.0, 0.7]
        
        for i, data in enumerate(test_data):
            success = await neurona.receive_external_data(data)
            print(f"   Entrada {i+1}: {data} -> {'✅' if success else '❌'}")
            await asyncio.sleep(0.1)  # Simular tiempo de procesamiento
        
        # Mostrar estadísticas
        stats = neurona.get_input_statistics()
        print(f"\n📈 Estadísticas de entrada:")
        print(f"   Total entradas: {stats['input_metrics']['total_inputs']}")
        print(f"   Entradas válidas: {stats['input_metrics']['valid_inputs']}")
        print(f"   Entradas inválidas: {stats['input_metrics']['invalid_inputs']}")
        print(f"   Rango: {stats['input_metrics']['min_input']:.2f} - {stats['input_metrics']['max_input']:.2f}")
        print(f"   Promedio: {stats['input_metrics']['average_input']:.4f}")
        
        # Mostrar estado completo
        state = neurona.get_state()
        print(f"\n🔍 Estado completo de la neurona:")
        print(json.dumps(state, indent=2, default=str))
        
        # Cerrar neurona
        await neurona.shutdown()
        
        print("\n" + "=" * 60)
        print("✅ NEURONA DE ENTRADA 01 COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error en neurona de entrada 01: {e}")
        logger.error(f"Error en neurona de entrada 01: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar neurona
    asyncio.run(main())
