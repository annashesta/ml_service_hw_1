# Импорт стандартных библиотек
import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle

# Настройка логирования
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


def cat_encode(train, input_df, col):
    """
    Кодирует категориальные переменные, заменяя их на числовые значения.
    Создает новую колонку <col>_cat, которая содержит закодированные категории.

    Args:
        train (pd.DataFrame): Обучающий датасет для создания таблицы соответствия.
        input_df (pd.DataFrame): Входной датасет для кодирования.
        col (str): Название категориальной колонки.

    Returns:
        pd.DataFrame: DataFrame с закодированными категориальными переменными.
    """
    logger.debug(f'Кодирование категории: {col}')
    new_col = col + '_cat'
    mapping = train[[col, new_col]].drop_duplicates()
    
    # Объединение с входным датасетом
    input_df = input_df.merge(mapping, how='left', on=col).drop(columns=col)
    return input_df


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


def load_train_data():
    """
    Загружает и предобрабатывает обучающий датасет.
    Выполняет следующие шаги:
    - Удаляет ненужные колонки.
    - Добавляет временные признаки.
    - Кодирует категориальные переменные.
    - Рассчитывает расстояния.

    Returns:
        pd.DataFrame: Предобработанный обучающий датасет.
    """
    logger.info('Загрузка обучающих данных...')

    # Определение типов колонок
    target_col = 'target'
    categorical_cols = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
    n_cats = 50

    # Загрузка обучающего датасета
    train = pd.read_csv('./train_data/train.csv').drop(columns=['name_1', 'name_2', 'street', 'post_code'])
    logger.info(f'Исходные данные загружены. Размер: {train.shape}')

    # Добавление временных признаков
    train = add_time_features(train)

    for col in categorical_cols:
        new_col = col + '_cat'

        # Создание таблицы категорий
        temp_df = (
            train.groupby(col, dropna=False)[[target_col]]
            .count()
            .sort_values(target_col, ascending=False)
            .reset_index()
            .set_axis([col, 'count'], axis=1)
            .reset_index()
        )
        temp_df['index'] = temp_df.apply(lambda x: np.nan if pd.isna(x[col]) else x['index'], axis=1)
        temp_df[new_col] = [
            'cat_NAN' if pd.isna(x) else f'cat_{x}' if x < n_cats else f'cat_{n_cats}+' 
            for x in temp_df['index']
        ]

        train = train.merge(temp_df[[col, new_col]], how='left', on=col)
    
    # Расчет расстояний
    train = add_distance_features(train)

    logger.info(f'Обработка обучающих данных завершена. Размер: {train.shape}')
    return train


def run_preproc(train, input_df):
    """
    Основная функция предобработки данных.
    Выполняет следующие шаги:
    - Кодирование категориальных переменных.
    - Добавление временных признаков.
    - Среднее кодирование категориальных переменных.
    - Расчет расстояний.
    - Обработка пропущенных значений.
    - Логарифмическое преобразование числовых признаков.

    Args:
        train (pd.DataFrame): Обучающий датасет для вычисления средних значений.
        input_df (pd.DataFrame): Входной датасет для предобработки.

    Returns:
        pd.DataFrame: Предобработанный датасет.
    """
    # Определение типов колонок
    target_col = 'target'
    categorical_cols = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
    continuous_cols = ['amount', 'population_city']
    
    # Кодирование категориальных переменных
    for col in categorical_cols:
        input_df = cat_encode(train, input_df, col)

    logger.info(f'Кодирование категориальных переменных завершено. Размер: {input_df.shape}')
    
    # Добавление временных признаков
    input_df = add_time_features(input_df)
    logger.info(f'Добавление временных признаков завершено. Размер: {input_df.shape}')

    categorical_cols = [x + '_cat' for x in categorical_cols]
    categorical_cols.extend(['hour', 'year', 'month', 'day_of_month', 'day_of_week'])
    
    # Среднее кодирование категориальных переменных
    for col in categorical_cols:
        # Заполнение пропущенных значений категориальных колонок
        input_df[col] = input_df[col].fillna('cat_NAN')
    
        # Создание таблицы средних значений
        means_tb = (
            train.groupby(col)[[target_col]]
            .mean()
            .reset_index()
            .rename(columns={target_col: f'{col}_mean_enc'})
        )
        
        # Объединение с входным датасетом
        input_df = input_df.merge(means_tb, how='left', on=col)

    logger.info(f'Среднее кодирование категориальных переменных завершено. Размер: {input_df.shape}')

    # Расчет расстояний
    input_df = add_distance_features(input_df)
    continuous_cols.extend(['distance'])

    # Обработка пропущенных значений числовых признаков
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    imputer = imputer.fit(train[continuous_cols])
    output_df = pd.concat([
        input_df.drop(columns=continuous_cols),
        pd.DataFrame(imputer.transform(input_df.copy()[continuous_cols]), columns=continuous_cols)
    ], axis=1)

    # Логарифмическое преобразование числовых признаков
    for col in continuous_cols:
        output_df[col + '_log'] = np.log(output_df[col] + 1)
        output_df.drop(columns=col, inplace=True)
        
    logger.info(f'Предобработка числовых признаков завершена. Размер: {output_df.shape}')
    
    # Масштабирование числовых признаков
    scaler = StandardScaler()
    output_df[continuous_cols] = scaler.fit_transform(output_df[continuous_cols])

    # Возвращение результирующего датасета
    return output_df