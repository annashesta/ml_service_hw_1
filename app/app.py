import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

# Добавление пути к модулям
sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, initialize_threshold, MODEL
from feature_importance import save_feature_importance
from plot_predictions import plot_predictions_distribution

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

import yaml

# Загрузка конфигурации
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config):
    """
    Проверяет корректность путей и параметров из конфигурационного файла.
    :param config: словарь с конфигурацией.
    """
    logger.info('Проверка конфигурации...')
    
    # Проверка существования директорий
    required_directories = ['input_dir', 'output_dir']
    for dir_key in required_directories:
        if not os.path.exists(config[dir_key]):
            raise FileNotFoundError(f"Директория {config[dir_key]} не найдена.")
    
    # Проверка существования файлов
    required_files = ['model_path', 'train_data_path']
    for file_key in required_files:
        if not os.path.exists(config[file_key]):
            raise FileNotFoundError(f"Файл {config[file_key]} не найден.")
    
    # Проверка наличия обязательных параметров
    required_params = ['random_state']
    for param in required_params:
        if param not in config:
            raise KeyError(f"Отсутствует обязательный параметр: {param}")
    
    logger.info('Конфигурация успешно проверена.')
    
    
# Загрузка конфигурации в начале программы
CONFIG = load_config('./config.yaml')


        
from sklearn.model_selection import train_test_split



class ProcessingService:
    def __init__(self):
        logger.info('Инициализация ProcessingService...')
        
        # Загрузка конфигурации
        self.config = load_config('./config.yaml')
        
        # Проверка конфигурации
        validate_config(self.config)
        
        # Инициализация переменных
        self.input_dir = self.config['input_dir']
        self.output_dir = self.config['output_dir']
        self.train = load_train_data(self.config['train_data_path'])
        
        # Разделение данных на обучающую и валидационную выборки
        X = self.train.drop(columns=['target'])
        y = self.train['target']
        
        # Инициализация оптимального порога с использованием кросс-валидации
        initialize_threshold_with_cv(X, y)
        
        logger.info('Сервис инициализирован.')
        
        
        

    def process_single_file(self, file_path):
        """
        Обрабатывает один файл: выполняет предобработку, предсказания и сохранение результатов.
        :param file_path: путь к входному файлу.
        """
        try:
            logger.info('Обработка файла: %s', file_path)

            # Проверка существования файла
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл {file_path} не найден.")

            # Загрузка данных
            input_df = pd.read_csv(file_path).drop(columns=['name_1', 'name_2', 'street', 'post_code'])

            # Предобработка данных
            logger.info('Начало предобработки данных.')
            processed_df = run_preproc(self.train, input_df)

            # Выполнение предсказаний
            logger.info('Выполнение предсказаний.')
            submission = make_pred(processed_df, file_path)

            # Сохранение результатов
            logger.info('Подготовка файла с предсказаниями.')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Предсказания сохранены в файл: %s', output_filename)

            # Сохранение важности признаков
            logger.info('Сохранение важности признаков.')
            feature_importance_path = os.path.join(self.output_dir, f"feature_importance_{timestamp}.json")
            save_feature_importance(MODEL, feature_importance_path)
            logger.info('Важность признаков сохранена в файл: %s', feature_importance_path)

            # Построение графика плотности предсказаний
            logger.info('Построение графика плотности предсказаний.')
            plot_path = os.path.join(self.output_dir, f"predictions_distribution_{timestamp}.png")
            plot_predictions_distribution(submission, plot_path)
            logger.info('График плотности предсказаний сохранён в файл: %s', plot_path)

        except Exception as e:
            logger.error('Ошибка при обработке файла %s: %s', file_path, e, exc_info=True)


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        """
        Обрабатывает создание нового файла.
        :param event: событие создания файла.
        """
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('Обнаружен новый файл: %s', event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info('Запуск сервиса ML scoring...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('Наблюдатель за файлами запущен.')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Сервис остановлен пользователем.')
        observer.stop()
    observer.join()