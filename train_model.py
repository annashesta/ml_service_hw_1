import pandas as pd
import numpy as np
import json
import logging
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import f1_score
from geopy.distance import great_circle
from catboost import CatBoostClassifier

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

def add_time_features(df):
    """
    Добавляет временные признаки из столбца 'transaction_time'.
    Временные признаки: час, год, месяц, день месяца, день недели.
    Удаляет исходный столбец 'transaction_time'.
    Args:
        df (pd.DataFrame): Исходный DataFrame с колонкой 'transaction_time'.
    Returns:
        pd.DataFrame: DataFrame с новыми временными признаками.
    """
    logger.debug('Добавление временных признаков...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df

def add_distance_features(df):
    """
    Рассчитывает расстояние между клиентом и продавцом в километрах.
    Удаляет исходные колонки с координатами.
    Args:
        df (pd.DataFrame): DataFrame с координатами клиента и продавца.
    Returns:
        pd.DataFrame: DataFrame с новым признаком 'distance'.
    """
    logger.debug('Расчет расстояний...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']), 
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])

def load_train_data(train_data_path):
    """
    Загружает и предобрабатывает обучающий датасет.
    Выполняет следующие шаги:
    - Удаляет ненужные колонки.
    - Добавляет временные признаки.
    - Рассчитывает расстояния.
    Args:
        train_data_path (str): Путь к обучающему датасету.
    Returns:
        pd.DataFrame: Предобработанный обучающий датасет.
    """
    logger.info('Загрузка обучающих данных...')
    # Определение типов колонок
    target_col = 'target'
    categorical_cols = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
    continuous_cols = ['amount', 'population_city']
    drop_cols = ['name_1', 'name_2', 'street', 'post_code']

    # Загрузка обучающего датасета
    train = pd.read_csv(train_data_path).drop(columns=drop_cols)
    logger.info(f'Исходные данные загружены. Размер: {train.shape}')

    # Добавление временных признаков
    train = add_time_features(train)

    # Расчет расстояний
    train = add_distance_features(train)

    logger.info(f'Обработка обучающих данных завершена. Размер: {train.shape}')
    return train

def find_optimal_threshold_with_cv(model, X, y):
    """
    Находит оптимальный порог для бинарной классификации на основе F1-score с использованием кросс-валидации.
    Args:
        model: Обученная модель CatBoost.
        X: DataFrame с признаками.
        y: Истинные метки.
    Returns:
        float: Оптимальный порог.
    """
    logger.info('Поиск оптимального порога с использованием кросс-валидации...')
    y_proba = cross_val_predict(model, X, y, method='predict_proba')[:, 1]
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0
    best_f1 = 0
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    logger.info(f'Оптимальный порог найден: {best_threshold:.4f} (F1-score: {best_f1:.4f})')
    return best_threshold

if __name__ == "__main__":
    # Путь к обучающему датасету
    train_data_path = 'train_data/train.csv'

    # Загрузка и предобработка данных
    train = load_train_data(train_data_path)

    # Разделение данных на признаки и целевую переменную
    X = train.drop(columns=['target'])
    y = train['target']

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Определение категориальных признаков
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Обучение модели CatBoost
    logger.info('Обучение модели CatBoost...')
    model = CatBoostClassifier(
        iterations=1000, 
        depth=6, 
        learning_rate=0.1, 
        loss_function='Logloss', 
        verbose=100, 
        random_state=RANDOM_STATE, 
        cat_features=categorical_features,  # Передаем категориальные признаки
        task_type="CPU",  # Используем CPU
        thread_count=4    # Ограничение количества потоков
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    # Сохранение модели
    model_path = './model/catboost_model.cbm'
    model.save_model(model_path)
    logger.info(f'Модель сохранена в файл {model_path}')

    # Поиск оптимального порога
    optimal_threshold = find_optimal_threshold_with_cv(model, X_val, y_val)

    # Сохранение порога в файл
    threshold_path = './model/threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump({"threshold": optimal_threshold}, f, indent=4)
    logger.info(f'Порог сохранен в файл {threshold_path}')

    # Сохранение списка категориальных признаков
    categorical_features_path = './model/categorical_features.json'
    with open(categorical_features_path, 'w') as f:
        json.dump({"categorical_features": categorical_features}, f, indent=4)
    logger.info(f'Список категориальных признаков сохранен в файл {categorical_features_path}')