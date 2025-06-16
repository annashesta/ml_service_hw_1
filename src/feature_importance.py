"""Модуль для работы с важностью признаков модели CatBoost."""

import json
import logging
from typing import Dict, List

# Настройка логгера
logger = logging.getLogger(__name__)


def save_feature_importance(
    model: "CatBoostClassifier",
    output_path: str,
    top_n: int = 5
) -> None:
    """
    Сохраняет важность признаков модели в JSON файл.
    
    Args:
        model: Обученная модель CatBoost.
        output_path: Путь для сохранения JSON файла.
        top_n: Количество топ-признаков для сохранения.
    
    Raises:
        RuntimeError: Если произошла ошибка при сохранении.
    """
    logger.info("Сохранение важности признаков в %s...", output_path)
    
    try:
        # Получение важности признаков и их имен
        feature_importances = model.get_feature_importance()
        feature_names = model.feature_names_

        # Проверка совпадения размеров
        if len(feature_importances) != len(feature_names):
            raise ValueError("Размеры важности признаков и их имен не совпадают")

        # Сортировка и выбор топ-N признаков
        sorted_indices = sorted(
            range(len(feature_importances)),
            key=lambda i: feature_importances[i],
            reverse=True
        )
        top_features = {
            feature_names[i]: float(feature_importances[i])
            for i in sorted_indices[:top_n]
        }

        # Сохранение в JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(top_features, f, indent=4)

        logger.info("Сохранены топ-%d важных признаков в %s", top_n, output_path)
    
    except Exception as e:
        logger.error("Ошибка сохранения важности признаков: %s", str(e))
        raise RuntimeError(f"Ошибка сохранения важности признаков: {str(e)}") from e