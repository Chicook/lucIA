#!/usr/bin/env python3
"""
LucIA Intelligent Cache System
Version: 0.6.0

This module provides the main cache interface and exports core cache classes.
"""

from .intelligent_cache import (
    IntelligentCache,
    CacheStrategy,
    CacheItem,
    CacheStats,
)

__all__ = [
    "CacheItem",
    "CacheStats",
    "CacheStrategy",
    "IntelligentCache",
]
