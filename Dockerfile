# Базовый образ
FROM python:3.9-slim

# Создание и настройка рабочей директории
WORKDIR /app

# Создание директорий
RUN mkdir -p /app/input /app/output /app/model /app/src /app/logs /app/train_data && \
    useradd -m appuser && \
    chown -R appuser:appuser /app

# Установка зависимостей (копируем отдельно для лучшего кэширования)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache /var/lib/apt/lists/*

# Копирование остальных файлов
COPY model/catboost_model.cbm /app/model/
COPY train_data/train.csv /app/train_data/
COPY src/ /app/src/
COPY app/app.py /app/

# Настройка прав доступа
RUN chmod -R 755 /app/logs

# Переключаемся на непривилегированного пользователя
USER appuser

# Команда для запуска сервиса
CMD ["python", "app/app.py"]

# Опционально: проверка здоровья контейнера
# Убедитесь, что сервис действительно имеет HTTP-сервер и endpoint /health
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

  