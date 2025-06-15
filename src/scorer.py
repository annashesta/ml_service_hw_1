import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier
import json

# Настройка логгера
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Загружает предобученную модель CatBoost.
    :param model_path: путь к файлу модели.
    :return: загруженная модель.
    """
    logger.info('Импортируется предобученная модель...')
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        logger.info('Предобученная модель успешно импортирована.')
        return model
    except Exception as e:
        logger.error('Ошибка при загрузке модели: %s', str(e))
        raise

# Глобальная переменная для модели
MODEL = None
# Глобальная переменная для оптимального порога
OPTIMAL_THRESHOLD = None

def load_threshold(threshold_path):
    """
    Загружает порог классификации из файла.

    :param threshold_path: путь к файлу с порогом.
    :return: порог классификации.
    """
    logger.info(f'Загрузка порога из файла {threshold_path}...')
    try:
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data.get('threshold', 0.5)  # Значение по умолчанию 0.5
        logger.info(f'Порог классификации загружен: {threshold}')
        return threshold
    except Exception as e:
        logger.error(f"Ошибка при загрузке порога из файла: {str(e)}")
        raise

def initialize_threshold(config):
    """
    Инициализирует оптимальный порог на основе конфигурации.
    :param config: конфигурационный словарь.
    """
    global OPTIMAL_THRESHOLD
    global MODEL

    model_path = config['paths']['model_path']
    threshold_path = config['paths']['threshold_path']
    logger.info(f'Загрузка модели из {model_path}...')
    MODEL = load_model(model_path)
    logger.info('Модель загружена.')

    # Загрузка порога из файла
    OPTIMAL_THRESHOLD = load_threshold(threshold_path)

def make_pred(data, config):
    """
    Выполняет предсказания на основе загруженной модели.

    :param data: DataFrame с предобработанными данными.
    :param config: конфигурационный словарь.
    :return: DataFrame с предсказаниями.
    """
    logger.info('Выполнение предсказаний...')
    global MODEL
    global OPTIMAL_THRESHOLD

    # Проверка наличия необходимых признаков
    required_features = MODEL.feature_names_
    if not all(feature in data.columns for feature in required_features):
        raise ValueError("Входные данные не содержат всех необходимых признаков.")

    # Получение вероятностей для положительного класса
    proba = MODEL.predict_proba(data)[:, 1]

    # Применение порога классификации
    predictions = (proba >= OPTIMAL_THRESHOLD).astype(int)

    # Формирование результата
    result = pd.DataFrame({
        'index': data.index,
        'prediction': predictions
    })
    logger.info('Предсказания выполнены.')
    return result