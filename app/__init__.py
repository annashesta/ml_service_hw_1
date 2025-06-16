# app/__init__.py

# Версия приложения
__version__ = "1.0.0"

# Экспорт ключевых классов и функций
from .app import ProcessingService, main

__all__ = [
    "ProcessingService",
    "main"
]