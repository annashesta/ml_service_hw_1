"""Модуль для предобработки данных перед подачей в модель."""
import logging
from typing import List
import pandas as pd
import numpy as np
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer

# Настройка логгера
logger = logging.getLogger(__name__)

# Глобальные переменные (будут инициализированы в load_train_data)
categorical_cols: List[str] = []
continuous_cols: List[str] = ["amount", "population_city"]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет временные признаки из столбца 'transaction_time'.
    Временные признаки: час, год, месяц, день месяца, день недели.
    Удаляет исходный столбец 'transaction_time'.
    Args:
        df (pd.DataFrame): DataFrame с колонкой 'transaction_time'.
    Returns:
        pd.DataFrame: DataFrame с новыми временными признаками.
    """
    logger.debug("Добавление временных признаков...")
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    dt = df["transaction_time"].dt
    df["hour"] = dt.hour
    df["year"] = dt.year
    df["month"] = dt.month
    df["day_of_month"] = dt.day
    df["day_of_week"] = dt.dayofweek
    return df.drop(columns="transaction_time")

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает расстояние между клиентом и продавцом в километрах.
    Удаляет исходные колонки с координатами.
    Args:
        df (pd.DataFrame): DataFrame с координатами клиента и продавца.
    Returns:
        pd.DataFrame: DataFrame с новым признаком 'distance'.
    """
    logger.debug("Расчет расстояний...")
    def calc_distance(row):
        try:
            return great_circle(
                (row["lat"], row["lon"]),
                (row["merchant_lat"], row["merchant_lon"])
            ).km
        except Exception as e:
            logger.warning(f"Ошибка расчета расстояния: {e}")
            return np.nan

    df["distance"] = df.apply(calc_distance, axis=1)
    return df.drop(columns=["lat", "lon", "merchant_lat", "merchant_lon"], errors="ignore")

def run_preproc(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет полный препроцессинг данных.
    Args:
        input_df (pd.DataFrame): Входной DataFrame для предобработки.
    Returns:
        pd.DataFrame: Предобработанный DataFrame.
    """
    logger.info("Начало препроцессинга данных...")
    # Проверка наличия необходимых колонок
    required_cols = (
        ["transaction_time", "lat", "lon", "merchant_lat", "merchant_lon"] +
        categorical_cols +
        continuous_cols
    )
    missing_cols = [col for col in required_cols if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_cols}")
    # 1. Добавление временных признаков
    input_df = add_time_features(input_df)
    logger.info("Добавлены временные признаки.")
    # 2. Расчет расстояний
    input_df = add_distance_features(input_df)
    logger.info("Расчет расстояний завершен.")
    # 3. Обработка пропущенных значений
    continuous_cols_with_distance = continuous_cols + ["distance"]
    imputer = SimpleImputer(strategy="mean")
    input_df[continuous_cols_with_distance] = imputer.fit_transform(input_df[continuous_cols_with_distance])
    logger.info("Обработка пропущенных значений завершена.")
    # 4. Логарифмическое преобразование числовых признаков
    for col in continuous_cols_with_distance:
        input_df[f"{col}_log"] = np.log(input_df[col] + 1)
    logger.info("Логарифмическое преобразование выполнено.")
    logger.info("Препроцессинг завершен.")
    return input_df



def load_train_data(train_data_path: str) -> pd.DataFrame:
    """
    Загружает и предобрабатывает обучающий датасет.
    Args:
        train_data_path (str): Путь к обучающему датасету.
    Returns:
        pd.DataFrame: Предобработанный обучающий датасет.
    """
    global categorical_cols

    logger.info("Загрузка обучающих данных...")
    # Определение типов колонок
    drop_cols = ["name_1", "name_2", "street", "post_code"]
    categorical_cols = ["gender", "merch", "cat_id", "one_city", "us_state", "jobs"]

    # Загрузка обучающего датасета
    train = pd.read_csv(train_data_path).drop(columns=drop_cols, errors="ignore")
    logger.info(f"Исходные данные загружены. Размер: {train.shape}")

    # Препроцессинг данных
    train = run_preproc(train)
    logger.info(f"Обработка обучающих данных завершена. Размер: {train.shape}")
    return train