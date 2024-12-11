# Норникель PDF Ассистент

Multimodal RAG pipeline with VLM (ColPali) - система для интеллектуального анализа PDF документов с поддержкой мультимодального поиска и генерации ответов.

## Описание

Проект представляет собой систему для работы с PDF документами, которая позволяет:

- Загружать и индексировать PDF документы
- Извлекать текст и изображения из документов
- Осуществлять семантический поиск по содержимому
- Генерировать ответы на вопросы с учетом визуального контекста

## Технологии

- FastAPI
- Streamlit
- ColPali Engine (Qwen2-VL)
- Qdrant Vector Database
- LangChain
- PyTorch

## Установка

1. Клонируйте репозиторий:

```bash
git clone git@github.com:erdenirf/hackathon_nornikel_2024_team_hakuna_matata.git
```

2. Установите зависимости с помощью Poetry:

```bash
poetry install
```

3. Создайте файл `.env` в корневой директории и добавьте необходимые переменные окружения:
```
QDRANT_API_KEY=your_api_key
QDRANT_URL=your_qdrant_url
QDRANT_COLLECTION_NAME=your_collection_name
```

## Запуск

### Backend API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Streamlit Interface

```bash
streamlit run streamlit.py
```

## API Endpoints

- `GET /documents` - Получить список всех проиндексированных документов
- `POST /documents/upload` - Загрузить и проиндексировать PDF документ
- `DELETE /documents/{filename}` - Удалить документ из индекса
- `POST /chat` - Получить ответ от RAG системы

## Лицензия

[MIT License](LICENSE.MD)

## Авторы

- Erdeni Tsyrendashiev
- Alexandr Maximov
- Safina Gulyamova
- Nikita Mazein

## Требования к системе

- Python 3.10+
- CUDA-совместимая видеокарта (для оптимальной производительности)
- 6gb GPU памяти для RAG
- 14gb GPU памяти для RAG+LLM на Qwen2-VL