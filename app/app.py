import os
import sys
import pandas as pd
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path).drop(columns=['name_1', 'name_2', 'street', 'post_code'])

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}_{os.path.basename(file_path)}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)
            logger.info('Predictions saved to: %s', output_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
    
    
    ----
    ----
    ---
    ---
    
    
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
from scorer import make_pred

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


class ProcessingService:
    def __init__(self):
        logger.info('Инициализация ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
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
    
    # Загрузка валидационных данных
    validation_data = pd.read_csv('/path/to/validation_data.csv')
    X_val = validation_data.drop(columns=['target'])
    y_val = validation_data['target']

    # Инициализация оптимального порога
    from scorer import initialize_threshold
    initialize_threshold(X_val, y_val)

    # Запуск сервиса
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