"""Основной скрипт приложения."""
import logging
import os
import sys
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import yaml
# Локальные импорты
from src.preprocess import load_train_data, run_preproc
from src.scorer import make_pred, initialize_threshold, MODEL
from src.feature_importance import save_feature_importance
from src.plot_predictions import plot_predictions_distribution

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

def load_config(config_path):
    """Загрузка конфигурационного файла."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфига: {e}")
        raise

class ProcessingService:
    """Сервис для обработки файлов и выполнения предсказаний."""
    def __init__(self, config):
        logger.info('Инициализация ProcessingService...')
        self.config = config
        self._validate_config()
        self.input_dir = self.config['paths']['input_dir']
        self.output_dir = self.config['paths']['output_dir']
        self.train = load_train_data(self.config['paths']['train_data_path'])
        initialize_threshold(self.config)
        logger.info('Сервис инициализирован.')
        logger.info('Для продолжения')
        logger.info('разместите файл формата `test.csv` в директории `./input`.')

    def _validate_config(self):
        """Проверка корректности конфигурации."""
        required_paths = [
            'input_dir', 'output_dir', 'model_path',
            'train_data_path', 'threshold_path'
        ]
        for path_key in required_paths:
            if not os.path.exists(self.config['paths'][path_key]):
                raise FileNotFoundError(
                    f"Путь не найден: {self.config['paths'][path_key]}"
                )

    def process_single_file(self, file_path):
        """Обработка одного файла."""
        try:
            logger.info(f'Обработка файла: {file_path}')
            input_df = pd.read_csv(file_path)
            # Удаление ненужных колонок
            cols_to_drop = ['name_1', 'name_2', 'street', 'post_code']
            input_df = input_df.drop(columns=cols_to_drop, errors='ignore')
            processed_df = run_preproc(input_df)
            submission = make_pred(processed_df, self.config)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sample_submission{timestamp}.csv"
            submission.to_csv(
                os.path.join(self.output_dir, output_filename),
                index=False
            )
            # Дополнительные функции для оценки на 5
            self._save_prediction_plot(output_filename)
        except Exception as e:
            logger.error(f'Ошибка обработки файла: {e}', exc_info=True)

    def _save_prediction_plot(self, predictions_file):
        """Сохранение графика плотности."""
        try:
            # Передаем только часть конфигурации, связанную с графиками
            plot_config = self.config.get('plots', {}).get('density_plot', {})
            plot_predictions_distribution(
                os.path.join(self.output_dir, predictions_file),
                os.path.join(self.output_dir, 'predictions_distribution.png'),
                plot_config  # Передаем только density_plot
            )
            # Добавляем сообщение о сохранении файлов
            logger.info(
                "В директории /output находятся файлы:\n"
                "- sample_submission.csv с предсказаниями\n"
                "- feature_importance.json с топ-5 feature importances\n"
                "- predictions_distribution.png с графиком плотности распределения скоров"
            )
        except Exception as e:
            logger.error(f'Ошибка сохранения графика: {e}')

class FileHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы."""
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            self.service.process_single_file(event.src_path)

def main():
    """Основная функция запуска сервиса."""
    try:
        config = load_config('./config.yaml')
        service = ProcessingService(config)
        observer = Observer()
        observer.schedule(
            FileHandler(service),
            path=service.input_dir,
            recursive=False
        )
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Сервис остановлен')
    except Exception as e:
        logger.error(f'Ошибка в главном цикле: {e}', exc_info=True)
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()