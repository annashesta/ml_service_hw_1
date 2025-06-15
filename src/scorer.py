import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import logging
from catboost import CatBoostClassifier

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


from sklearn.model_selection import cross_val_predict


from sklearn.model_selection import cross_val_predict

def find_optimal_threshold_with_cv(model, X, y):
    """
    Находит оптимальный порог для бинарной классификации на основе F1-score с использованием кросс-валидации.
    :param model: обученная модель CatBoost.
    :param X: DataFrame с признаками.
    :param y: истинные метки.
    :return: оптимальный порог.
    """
    logger.info('Поиск оптимального порога с использованием кросс-валидации...')
    
    # Получаем вероятности для положительного класса с помощью кросс-валидации
    y_proba = cross_val_predict(model, X, y, method='predict_proba')[:, 1]
    
    # Ищем оптимальный порог
    thresholds = np.linspace(0, 1, 100)  # Пробуем 100 значений порога
    best_threshold = 0
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)  # Преобразуем вероятности в метки
        f1 = f1_score(y, y_pred)  # Вычисляем F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f'Оптимальный порог найден: {best_threshold:.4f} (F1-score: {best_f1:.4f})')
    return best_threshold



# Глобальная переменная для модели
MODEL = load_model('./models/my_catboost.cbm')

# Глобальная переменная для оптимального порога
OPTIMAL_THRESHOLD = None


def initialize_threshold_with_cv(X, y):
    """
    Инициализирует оптимальный порог на основе кросс-валидации.
    :param X: DataFrame с признаками.
    :param y: истинные метки.
    """
    global OPTIMAL_THRESHOLD
    OPTIMAL_THRESHOLD = find_optimal_threshold_with_cv(MODEL, X, y)


def make_pred(dt, path_to_file, model_th=None):
    """
    Выполняет предсказания для входных данных.
    :param dt: DataFrame с обработанными данными.
    :param path_to_file: путь к исходному файлу для создания submission.
    :param model_th: порог для бинарной классификации (по умолчанию используется OPTIMAL_THRESHOLD).
    :return: DataFrame с предсказаниями.
    """
    try:
        # Если порог не передан, используем оптимальный
        threshold = model_th if model_th is not None else OPTIMAL_THRESHOLD

        # Проверка, что все необходимые признаки присутствуют
        required_features = MODEL.feature_names_
        if not all(feature in dt.columns for feature in required_features):
            raise ValueError("Входные данные не содержат всех необходимых признаков.")

        # Создание DataFrame с индексами и предсказаниями
        submission = pd.DataFrame({
            'index': pd.read_csv(path_to_file).index,  # Индексы из исходного файла
            'prediction': (MODEL.predict_proba(dt)[:, 1] > threshold) * 1  # Бинарные метки на основе порога
        })
        logger.info('Предсказания завершены для файла: %s', path_to_file)
        return submission

    except Exception as e:
        logger.error('Ошибка при выполнении предсказаний: %s', str(e))
        raise
    