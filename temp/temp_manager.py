#!/usr/bin/env python3
"""
Sistema de Gestión de Archivos Temporales - LucIA
Versión: 0.6.0
Sistema avanzado para gestión de archivos temporales con limpieza automática
"""

import os
import shutil
import tempfile
import time
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio

logger = logging.getLogger('Temp_Manager')

class TempFileType(Enum):
    """Tipos de archivos temporales"""
    CACHE = "cache"
    PROCESSING = "processing"
    BACKUP = "backup"
    LOG = "log"
    DATA = "data"
    MODEL = "model"
    EXPORT = "export"

@dataclass
class TempFileInfo:
    """Información de un archivo temporal"""
    file_id: str
    file_path: str
    file_type: TempFileType
    created_at: datetime
    last_accessed: datetime
    size: int
    ttl: int  # Time to live en segundos
    metadata: Dict[str, Any]
    is_locked: bool = False

class TempFileManager:
    """
    Gestor avanzado de archivos temporales
    """
    
    def __init__(self, base_dir: str = "temp", 
                 max_age: int = 3600,  # 1 hora por defecto
                 max_size: int = 1024 * 1024 * 1024,  # 1GB por defecto
                 cleanup_interval: int = 300):  # 5 minutos
        self.base_dir = base_dir
        self.max_age = max_age
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        # Crear directorio base si no existe
        os.makedirs(base_dir, exist_ok=True)
        
        # Crear subdirectorios por tipo
        for file_type in TempFileType:
            os.makedirs(os.path.join(base_dir, file_type.value), exist_ok=True)
        
        # Registro de archivos temporales
        self.temp_files: Dict[str, TempFileInfo] = {}
        self.lock = threading.RLock()
        
        # Archivo de metadatos
        self.metadata_file = os.path.join(base_dir, "temp_metadata.json")
        
        # Cargar metadatos existentes
        self._load_metadata()
        
        # Iniciar limpieza automática
        self._start_cleanup_task()
        
        logger.info("Sistema de gestión de archivos temporales inicializado")
    
    def _start_cleanup_task(self):
        """Inicia la tarea de limpieza automática"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_expired_files()
                    self._cleanup_oversized_files()
                    self._save_metadata()
                except Exception as e:
                    logger.error(f"Error en limpieza automática: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def create_temp_file(self, file_type: TempFileType, 
                        content: Optional[Union[str, bytes]] = None,
                        extension: str = ".tmp",
                        ttl: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Crea un archivo temporal
        
        Args:
            file_type: Tipo de archivo temporal
            content: Contenido del archivo (opcional)
            extension: Extensión del archivo
            ttl: Tiempo de vida en segundos
            metadata: Metadatos adicionales
            
        Returns:
            ID del archivo temporal creado
        """
        try:
            with self.lock:
                # Generar ID único
                file_id = self._generate_file_id()
                
                # Crear nombre de archivo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{file_id}_{timestamp}{extension}"
                file_path = os.path.join(self.base_dir, file_type.value, filename)
                
                # Crear archivo
                if content is not None:
                    if isinstance(content, str):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        with open(file_path, 'wb') as f:
                            f.write(content)
                else:
                    # Crear archivo vacío
                    open(file_path, 'a').close()
                
                # Obtener información del archivo
                file_size = os.path.getsize(file_path)
                
                # Crear información del archivo
                file_info = TempFileInfo(
                    file_id=file_id,
                    file_path=file_path,
                    file_type=file_type,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    size=file_size,
                    ttl=ttl or self.max_age,
                    metadata=metadata or {}
                )
                
                # Registrar archivo
                self.temp_files[file_id] = file_info
                
                logger.debug(f"Archivo temporal creado: {file_id} ({file_type.value})")
                return file_id
                
        except Exception as e:
            logger.error(f"Error creando archivo temporal: {e}")
            raise
    
    def get_temp_file(self, file_id: str) -> Optional[TempFileInfo]:
        """
        Obtiene información de un archivo temporal
        
        Args:
            file_id: ID del archivo temporal
            
        Returns:
            Información del archivo o None si no existe
        """
        try:
            with self.lock:
                if file_id not in self.temp_files:
                    return None
                
                file_info = self.temp_files[file_id]
                
                # Verificar si ha expirado
                if self._is_expired(file_info):
                    self._delete_temp_file(file_id)
                    return None
                
                # Actualizar último acceso
                file_info.last_accessed = datetime.now()
                
                return file_info
                
        except Exception as e:
            logger.error(f"Error obteniendo archivo temporal: {e}")
            return None
    
    def read_temp_file(self, file_id: str, as_text: bool = True) -> Optional[Union[str, bytes]]:
        """
        Lee el contenido de un archivo temporal
        
        Args:
            file_id: ID del archivo temporal
            as_text: Si True, lee como texto; si False, como bytes
            
        Returns:
            Contenido del archivo o None si no existe
        """
        try:
            file_info = self.get_temp_file(file_id)
            if not file_info or not os.path.exists(file_info.file_path):
                return None
            
            if as_text:
                with open(file_info.file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                with open(file_info.file_path, 'rb') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Error leyendo archivo temporal: {e}")
            return None
    
    def write_temp_file(self, file_id: str, content: Union[str, bytes]) -> bool:
        """
        Escribe contenido en un archivo temporal existente
        
        Args:
            file_id: ID del archivo temporal
            content: Contenido a escribir
            
        Returns:
            True si se escribió correctamente
        """
        try:
            file_info = self.get_temp_file(file_id)
            if not file_info:
                return False
            
            # Bloquear archivo para escritura
            file_info.is_locked = True
            
            try:
                if isinstance(content, str):
                    with open(file_info.file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    with open(file_info.file_path, 'wb') as f:
                        f.write(content)
                
                # Actualizar tamaño
                file_info.size = os.path.getsize(file_info.file_path)
                file_info.last_accessed = datetime.now()
                
                return True
                
            finally:
                file_info.is_locked = False
                
        except Exception as e:
            logger.error(f"Error escribiendo archivo temporal: {e}")
            return False
    
    def delete_temp_file(self, file_id: str) -> bool:
        """
        Elimina un archivo temporal
        
        Args:
            file_id: ID del archivo temporal
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            with self.lock:
                return self._delete_temp_file(file_id)
                
        except Exception as e:
            logger.error(f"Error eliminando archivo temporal: {e}")
            return False
    
    def _delete_temp_file(self, file_id: str) -> bool:
        """Elimina un archivo temporal (método interno)"""
        if file_id not in self.temp_files:
            return False
        
        file_info = self.temp_files[file_id]
        
        # Verificar si está bloqueado
        if file_info.is_locked:
            logger.warning(f"Archivo temporal {file_id} está bloqueado, no se puede eliminar")
            return False
        
        try:
            # Eliminar archivo físico
            if os.path.exists(file_info.file_path):
                os.remove(file_info.file_path)
            
            # Eliminar del registro
            del self.temp_files[file_id]
            
            logger.debug(f"Archivo temporal eliminado: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando archivo temporal {file_id}: {e}")
            return False
    
    def cleanup_by_type(self, file_type: TempFileType) -> int:
        """
        Limpia archivos temporales de un tipo específico
        
        Args:
            file_type: Tipo de archivo a limpiar
            
        Returns:
            Número de archivos eliminados
        """
        try:
            with self.lock:
                files_to_delete = [
                    file_id for file_id, file_info in self.temp_files.items()
                    if file_info.file_type == file_type
                ]
                
                deleted_count = 0
                for file_id in files_to_delete:
                    if self._delete_temp_file(file_id):
                        deleted_count += 1
                
                logger.info(f"Limpieza por tipo {file_type.value}: {deleted_count} archivos eliminados")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error en limpieza por tipo: {e}")
            return 0
    
    def cleanup_expired(self) -> int:
        """Limpia archivos temporales expirados"""
        try:
            with self.lock:
                expired_files = [
                    file_id for file_id, file_info in self.temp_files.items()
                    if self._is_expired(file_info)
                ]
                
                deleted_count = 0
                for file_id in expired_files:
                    if self._delete_temp_file(file_id):
                        deleted_count += 1
                
                logger.info(f"Limpieza de expirados: {deleted_count} archivos eliminados")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error en limpieza de expirados: {e}")
            return 0
    
    def _cleanup_expired_files(self):
        """Limpia archivos expirados (método interno)"""
        self.cleanup_expired()
    
    def _cleanup_oversized_files(self):
        """Limpia archivos si se excede el tamaño máximo"""
        try:
            with self.lock:
                total_size = sum(file_info.size for file_info in self.temp_files.values())
                
                if total_size > self.max_size:
                    # Ordenar por fecha de creación (más antiguos primero)
                    sorted_files = sorted(
                        self.temp_files.items(),
                        key=lambda x: x[1].created_at
                    )
                    
                    # Eliminar archivos más antiguos hasta liberar espacio
                    freed_size = 0
                    target_size = self.max_size * 0.8  # Liberar hasta 80% del máximo
                    
                    for file_id, file_info in sorted_files:
                        if total_size - freed_size <= target_size:
                            break
                        
                        if self._delete_temp_file(file_id):
                            freed_size += file_info.size
                    
                    logger.info(f"Limpieza por tamaño: {freed_size} bytes liberados")
                    
        except Exception as e:
            logger.error(f"Error en limpieza por tamaño: {e}")
    
    def _is_expired(self, file_info: TempFileInfo) -> bool:
        """Verifica si un archivo ha expirado"""
        age = (datetime.now() - file_info.created_at).total_seconds()
        return age > file_info.ttl
    
    def _generate_file_id(self) -> str:
        """Genera un ID único para el archivo"""
        timestamp = str(time.time())
        random_data = os.urandom(16)
        hash_input = f"{timestamp}_{random_data}".encode()
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor de archivos temporales"""
        try:
            with self.lock:
                total_files = len(self.temp_files)
                total_size = sum(file_info.size for file_info in self.temp_files.values())
                
                # Estadísticas por tipo
                type_stats = {}
                for file_type in TempFileType:
                    type_files = [
                        f for f in self.temp_files.values() 
                        if f.file_type == file_type
                    ]
                    type_stats[file_type.value] = {
                        'count': len(type_files),
                        'size': sum(f.size for f in type_files)
                    }
                
                # Archivos expirados
                expired_count = sum(
                    1 for file_info in self.temp_files.values()
                    if self._is_expired(file_info)
                )
                
                return {
                    'total_files': total_files,
                    'total_size': total_size,
                    'expired_files': expired_count,
                    'type_distribution': type_stats,
                    'max_size': self.max_size,
                    'max_age': self.max_age,
                    'usage_percentage': (total_size / self.max_size) * 100 if self.max_size > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def _save_metadata(self):
        """Guarda metadatos de archivos temporales"""
        try:
            metadata = {
                'files': {
                    file_id: {
                        'file_path': file_info.file_path,
                        'file_type': file_info.file_type.value,
                        'created_at': file_info.created_at.isoformat(),
                        'last_accessed': file_info.last_accessed.isoformat(),
                        'size': file_info.size,
                        'ttl': file_info.ttl,
                        'metadata': file_info.metadata
                    }
                    for file_id, file_info in self.temp_files.items()
                },
                'config': {
                    'max_age': self.max_age,
                    'max_size': self.max_size,
                    'cleanup_interval': self.cleanup_interval
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error guardando metadatos: {e}")
    
    def _load_metadata(self):
        """Carga metadatos de archivos temporales"""
        try:
            if not os.path.exists(self.metadata_file):
                return
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Cargar archivos
            for file_id, file_data in metadata.get('files', {}).items():
                file_info = TempFileInfo(
                    file_id=file_id,
                    file_path=file_data['file_path'],
                    file_type=TempFileType(file_data['file_type']),
                    created_at=datetime.fromisoformat(file_data['created_at']),
                    last_accessed=datetime.fromisoformat(file_data['last_accessed']),
                    size=file_data['size'],
                    ttl=file_data['ttl'],
                    metadata=file_data['metadata']
                )
                
                # Solo cargar si el archivo físico existe
                if os.path.exists(file_info.file_path):
                    self.temp_files[file_id] = file_info
            
            logger.info(f"Cargados {len(self.temp_files)} archivos temporales")
            
        except Exception as e:
            logger.error(f"Error cargando metadatos: {e}")
    
    def cleanup_all(self) -> int:
        """Limpia todos los archivos temporales"""
        try:
            with self.lock:
                all_files = list(self.temp_files.keys())
                deleted_count = 0
                
                for file_id in all_files:
                    if self._delete_temp_file(file_id):
                        deleted_count += 1
                
                logger.info(f"Limpieza completa: {deleted_count} archivos eliminados")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error en limpieza completa: {e}")
            return 0

# Instancia global del gestor de archivos temporales
temp_file_manager = TempFileManager()
