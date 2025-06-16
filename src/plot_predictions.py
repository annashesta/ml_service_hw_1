"""Модуль для визуализации распределения предсказаний модели."""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore


# Настройка логгера
logger = logging.getLogger(__name__)


def plot_predictions_distribution(
    predictions_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> None:
    """
    Строит и сохраняет график плотности распределения предсказаний.
    
    Args:
        predictions_path: Путь к CSV файлу с предсказаниями.
        output_path: Путь для сохранения графика.
        config: Конфигурационный словарь с параметрами графика.
    
    Raises:
        RuntimeError: Если возникла ошибка при построении графика.
        FileNotFoundError: Если файл предсказаний не найден.
        ValueError: Если данные предсказаний некорректны.
    """
    logger.info("Построение графика плотности в %s", output_path)
    
    try:
        # Проверка существования файла
        if not Path(predictions_path).exists():
            raise FileNotFoundError(f"Файл {predictions_path} не найден")

        # Загрузка данных
        data = pd.read_csv(predictions_path)
        if 'prediction' not in data.columns:
            raise ValueError("Колонка 'prediction' отсутствует в файле")
        
        predictions = data['prediction']
        
        # Валидация данных
        if predictions.empty:
            raise ValueError("Файл предсказаний пуст")
        if not pd.api.types.is_numeric_dtype(predictions):
            raise ValueError("Предсказания должны быть числовыми")
        
        # Получение параметров из конфига
        plot_config = config.get('plots', {}).get('density_plot', {})
        width = plot_config.get('width', 10)
        height = plot_config.get('height', 6)
        color = plot_config.get('color', 'royalblue')
        title = plot_config.get('title', 'Распределение предсказаний')
        dpi = plot_config.get('dpi', 300)
        
        # Инициализация графика
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Построение графика плотности
        sns.kdeplot(
            data=predictions,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.6,
            linewidth=1.5
        )
        
        # Настройка оформления
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Значение предсказания', fontsize=12)
        ax.set_ylabel('Плотность вероятности', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Сохранение графика
        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches='tight',
            format='png'
        )
        plt.close(fig)
        
        logger.info("График успешно сохранен: %s", output_path)
        
    except Exception as e:
        logger.error("Ошибка построения графика: %s", str(e))
        raise RuntimeError(f"Ошибка построения графика: {str(e)}") from e