# src/__init__.py

# Версия пакета
__version__ = "1.0.0"

# Экспорт ключевых функций для удобства импорта
from .preprocess import load_train_data, run_preproc
from .scorer import make_pred, initialize_threshold
from .feature_importance import save_feature_importance
from .plot_predictions import plot_predictions_distribution

__all__ = [
    "load_train_data"
    "run_preproc",
    "make_pred",
    "initialize_threshold",
    "save_feature_importance",
    "plot_predictions_distribution"
]