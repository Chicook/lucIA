"""
Script para crear las neuronas de entrada restantes (04-08)
==========================================================
"""

import os

def create_neuron_file(neuron_num):
    """Crea un archivo de neurona para el número especificado"""
    
    content = f'''"""
Neurona de Entrada {neuron_num:02d} - Módulo Independiente
===========================================

Neurona {neuron_num} de la capa de entrada de la red neuronal modular.
Esta neurona recibe datos externos y los procesa de manera independiente,
comunicándose con las neuronas de la primera capa oculta.

Características:
- Tipo: INPUT
- Índice: {neuron_num:02d}
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
class NeuronaInput{neuron_num:02d}Config(NeuronConfig):
    """Configuración específica para la neurona de entrada {neuron_num:02d}"""
    
    def __init__(self):
        super().__init__(
            neuron_id="input_{neuron_num:02d}",
            neuron_type=NeuronType.INPUT,
            layer_index=0,
            neuron_index={neuron_num},
            activation=ActivationType.LINEAR,
            bias=0.0,
            learning_rate=0.001,
            dropout_rate=0.0,
            batch_normalization=False,
            weight_decay=0.0,
            max_connections=10,
            processing_timeout=1.0
        )
        
        self.input_range = (-1.0, 1.0)
        self.normalization_factor = 1.0
        self.data_type = "continuous"

class NeuronaInput{neuron_num:02d}(InputNeuron):
    """Neurona de entrada {neuron_num:02d} - Procesa la característica {neuron_num} de entrada."""
    
    def __init__(self):
        config = NeuronaInput{neuron_num:02d}Config()
        super().__init__(config)
        
        self.input_metrics = {{
            'total_inputs': 0, 'valid_inputs': 0, 'invalid_inputs': 0,
            'min_input': float('inf'), 'max_input': float('-inf'),
            'average_input': 0.0, 'input_variance': 0.0, 'last_input_time': 0.0
        }}
        
        self.current_external_input = None
        self.is_receiving_data = False
        self.data_quality_score = 1.0
        
        logger.info(f"Neurona de entrada {neuron_num:02d} inicializada - ID: {{self.config.neuron_id}}")
    
    async def receive_external_data(self, data: float) -> bool:
        """Recibe datos externos y los procesa."""
        try:
            self.is_receiving_data = True
            processed_data = await self._validate_and_normalize_input(data)
            
            if processed_data is not None:
                self._update_input_metrics(data, processed_data)
                output = await self.process_input(processed_data, source_neuron="external")
                self.input_metrics['last_input_time'] = time.time()
                self.is_receiving_data = False
                logger.debug(f"Neurona {{self.config.neuron_id}} procesó entrada: {{data}} -> {{output:.4f}}")
                return True
            else:
                self.input_metrics['invalid_inputs'] += 1
                self.is_receiving_data = False
                logger.warning(f"Dato inválido recibido en neurona {{self.config.neuron_id}}: {{data}}")
                return False
                
        except Exception as e:
            self.is_receiving_data = False
            self.metrics['error_count'] += 1
            self.metrics['last_error'] = str(e)
            logger.error(f"Error recibiendo datos en neurona {{self.config.neuron_id}}: {{e}}")
            return False
    
    async def _validate_and_normalize_input(self, data: float) -> Optional[float]:
        """Valida y normaliza el dato de entrada."""
        try:
            if not isinstance(data, (int, float)) or np.isnan(data) or np.isinf(data):
                return None
            
            min_range, max_range = self.config.input_range
            if data < min_range * 10 or data > max_range * 10:
                logger.warning(f"Dato fuera de rango esperado: {{data}}")
            
            return float(data * self.config.normalization_factor)
            
        except Exception as e:
            logger.error(f"Error validando entrada: {{e}}")
            return None
    
    def _update_input_metrics(self, raw_input: float, processed_input: float):
        """Actualiza las métricas específicas de entrada"""
        self.input_metrics['total_inputs'] += 1
        self.input_metrics['valid_inputs'] += 1
        self.input_metrics['min_input'] = min(self.input_metrics['min_input'], raw_input)
        self.input_metrics['max_input'] = max(self.input_metrics['max_input'], raw_input)
        
        total = self.input_metrics['total_inputs']
        current_avg = self.input_metrics['average_input']
        self.input_metrics['average_input'] = ((current_avg * (total - 1)) + raw_input) / total
    
    def get_input_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas de entrada"""
        return {{
            'neuron_id': self.config.neuron_id,
            'input_metrics': self.input_metrics,
            'data_quality_score': self.data_quality_score,
            'is_receiving_data': self.is_receiving_data,
            'current_external_input': self.current_external_input,
            'timestamp': datetime.now().isoformat()
        }}

# Función principal
async def main():
    print("=" * 60)
    print(f"NEURONA DE ENTRADA {neuron_num:02d} - MÓDULO INDEPENDIENTE")
    print("=" * 60)
    
    try:
        neurona = NeuronaInput{neuron_num:02d}()
        await neurona.initialize()
        
        print(f"✅ Neurona {{neurona.config.neuron_id}} inicializada")
        print(f"   Tipo: {{neurona.config.neuron_type.value}}")
        print(f"   Activación: {{neurona.config.activation.value}}")
        
        # Datos de prueba específicos para cada neurona
        test_data = [0.1, 0.5, -0.2, 0.9, -0.6, 1.3, -0.1, 0.4]
        
        for i, data in enumerate(test_data):
            success = await neurona.receive_external_data(data)
            print(f"   Entrada {{i+1}}: {{data}} -> {{'✅' if success else '❌'}}")
            await asyncio.sleep(0.1)
        
        stats = neurona.get_input_statistics()
        print(f"\\n📈 Total entradas: {{stats['input_metrics']['total_inputs']}}")
        print(f"   Promedio: {{stats['input_metrics']['average_input']:.4f}}")
        
        await neurona.shutdown()
        print(f"\\n✅ NEURONA DE ENTRADA {neuron_num:02d} COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error en neurona de entrada {neuron_num:02d}: {{e}}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main())
'''
    
    return content

def main():
    """Crea las neuronas de entrada restantes"""
    print("Creando neuronas de entrada restantes (04-08)...")
    
    for neuron_num in range(4, 9):  # 04, 05, 06, 07, 08
        filename = f"neurona_input_{neuron_num:02d}.py"
        content = create_neuron_file(neuron_num)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Creada: {filename}")

if __name__ == "__main__":
    main()
