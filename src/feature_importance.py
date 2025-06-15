import json
from catboost import CatBoostClassifier
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

def save_feature_importance(model, output_path, top_n=5):
    """
    Сохраняет важность признаков в JSON файл.

    :param model: обученная модель CatBoost.
    :param output_path: путь к выходному JSON файлу.
    :param top_n: количество топ-признаков для сохранения.
    """
    logger.info(f'Сохранение важности признаков в файл {output_path}...')
    try:
        # Получение важности признаков
        feature_importances = model.get_feature_importance()
        feature_names = model.feature_names_
        
        # Сортировка признаков по важности
        sorted_indices = np.argsort(feature_importances)[::-1]
        top_features = {
            feature_names[i]: float(feature_importances[i])
            for i in sorted_indices[:top_n]
        }
        
        # Сохранение в JSON
        with open(output_path, 'w') as f:
            json.dump(top_features, f, indent=4)
        
        logger.info('Важность признаков успешно сохранена.')
    except Exception as e:
        logger.error(f"Ошибка сохранения важности признаков: {str(e)}")
        raise