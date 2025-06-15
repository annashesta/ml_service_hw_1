import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions_distribution(predictions_path, output_path):
    # Загрузка предсказаний
    predictions = pd.read_csv(predictions_path)['target']
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions, shade=True, color='blue')
    plt.title('Density Plot of Predictions')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    
    # Сохранение графика
    plt.savefig(output_path)
    plt.close()