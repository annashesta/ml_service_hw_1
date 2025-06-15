import json
from catboost import CatBoostClassifier

def save_feature_importance(model_path, output_path):
    # Загрузка модели
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # Получение важности признаков
    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_
    
    # Топ-5 важных признаков
    top_features = {
        feature_names[i]: float(feature_importances[i])
        for i in range(5)
    }
    
    # Сохранение в JSON
    with open(output_path, 'w') as f:
        json.dump(top_features, f, indent=4)