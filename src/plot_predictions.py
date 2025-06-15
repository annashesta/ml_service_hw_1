import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Настройка логгера
logger = logging.getLogger(__name__)

def plot_predictions_distribution(predictions_path, output_path, config):
    """
    Построение графика плотности предсказаний.

    :param predictions_path: путь к файлу с предсказаниями.
    :param output_path: путь к выходному файлу с графиком.
    :param config: конфигурационный словарь.
    """
    logger.info(f'Построение графика плотности предсказаний в файл {output_path}...')
    try:
        # Загрузка предсказаний
        predictions = pd.read_csv(predictions_path)['prediction']
        
        # Параметры графика
        plot_config = config['plots']['density_plot']
        width = plot_config['width']
        height = plot_config['height']
        color = plot_config['color']
        
        # Построение графика
        plt.figure(figsize=(width, height))
        sns.kdeplot(predictions, shade=True, color=color)
        plt.title('Density Plot of Predictions')
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        
        # Сохранение графика
        plt.savefig(output_path)
        plt.close()
        
        logger.info('График плотности предсказаний успешно сохранен.')
    except Exception as e:
        logger.error(f"Ошибка построения графика плотности предсказаний: {str(e)}")
        raise