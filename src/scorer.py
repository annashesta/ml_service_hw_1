"""Модуль для выполнения предсказаний с помощью CatBoost модели."""
import json
import logging
from typing import Dict, List
import pandas as pd
from catboost import CatBoostClassifier
from src.feature_importance import save_feature_importance  # Импортируем функцию сохранения важности признаков

# Настройка логгера
logger = logging.getLogger(__name__)

# Глобальные переменные
MODEL: CatBoostClassifier = None
OPTIMAL_THRESHOLD: float = None
CATEGORICAL_FEATURES: List[str] = []

def load_model(model_path: str, categorical_features: List[str]) -> CatBoostClassifier:
    """
    Загружает предобученную модель CatBoost.
    Args:
        model_path: Путь к файлу модели.
        categorical_features: Список категориальных признаков.
    Returns:
        Загруженная модель CatBoost.
    Raises:
        RuntimeError: Если не удалось загрузить модель.
    """
    logger.info("Загрузка модели из %s...", model_path)
    try:
        model = CatBoostClassifier(cat_features=categorical_features)
        model.load_model(model_path)
        logger.info("Модель успешно загружена")
        return model
    except Exception as e:
        logger.error("Ошибка загрузки модели: %s", str(e))
        raise RuntimeError(f"Ошибка загрузки модели: {str(e)}") from e

def load_threshold(threshold_path: str) -> float:
    """
    Загружает порог классификации из файла.
    Args:
        threshold_path: Путь к файлу с порогом.
    Returns:
        Порог классификации (по умолчанию 0.5).
    Raises:
        RuntimeError: Если не удалось загрузить порог.
    """
    logger.info("Загрузка порога из %s...", threshold_path)
    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        threshold = threshold_data.get("threshold", 0.5)
        logger.info("Порог классификации: %.2f", threshold)
        return threshold
    except Exception as e:
        logger.error("Ошибка загрузки порога: %s", str(e))
        raise RuntimeError(f"Ошибка загрузки порога: {str(e)}") from e

def load_categorical_features(features_path: str) -> List[str]:
    """
    Загружает список категориальных признаков из JSON-файла.
    Args:
        features_path: Путь к файлу с категориальными признаками.
    Returns:
        Список категориальных признаков.
    Raises:
        RuntimeError: Если не удалось загрузить список категориальных признаков.
    """
    logger.info("Загрузка списка категориальных признаков из %s...", features_path)
    try:
        with open(features_path, "r", encoding="utf-8") as f:
            features_data = json.load(f)
        categorical_features = features_data.get("categorical_features", [])
        logger.info("Категориальные признаки: %s", categorical_features)
        return categorical_features
    except Exception as e:
        logger.error("Ошибка загрузки категориальных признаков: %s", str(e))
        raise RuntimeError(f"Ошибка загрузки категориальных признаков: {str(e)}") from e

def initialize_threshold(config: Dict) -> None:
    """
    Инициализирует модель, порог классификации и список категориальных признаков.
    Args:
        config: Конфигурационный словарь.
    """
    global MODEL, OPTIMAL_THRESHOLD, CATEGORICAL_FEATURES
    # Загрузка категориальных признаков
    CATEGORICAL_FEATURES = load_categorical_features(
        config["paths"]["categorical_features_path"]
    )
    # Загрузка модели
    MODEL = load_model(
        config["paths"]["model_path"],
        CATEGORICAL_FEATURES
    )
    # Проверка загрузки модели
    if MODEL is None:
        raise ValueError("Модель не загружена!")
    logger.info(f"Тип загруженной модели: {type(MODEL)}")
    if not hasattr(MODEL, "get_feature_importance"):
        raise AttributeError("Модель не поддерживает метод get_feature_importance")
    # Сохранение важности признаков
    try:
        save_feature_importance(
            MODEL,
            config["paths"]["output_dir"] + "/feature_importance.json",
            top_n=5
        )
        logger.info("Важность признаков успешно сохранена.")
    except Exception as e:
        logger.error(f"Ошибка сохранения важности признаков: {e}")
    # Загрузка порога
    OPTIMAL_THRESHOLD = load_threshold(config["paths"]["threshold_path"])

def make_pred(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Выполняет предсказания на основе загруженной модели.
    Args:
        data: DataFrame с предобработанными данными.
        config: Конфигурационный словарь.
    Returns:
        DataFrame с предсказаниями.
    Raises:
        ValueError: Если отсутствуют необходимые признаки.
    """
    logger.info("Выполнение предсказаний...")
    # Проверка наличия необходимых признаков
    required_features = MODEL.feature_names_
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        error_msg = f"Отсутствуют признаки: {missing_features}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    # Предсказания
    proba = MODEL.predict_proba(data)[:, 1]
    predictions = (proba >= OPTIMAL_THRESHOLD).astype(int)
    logger.info("Предсказания выполнены")
    return pd.DataFrame({
        "index": data.index,
        "prediction": predictions
    })