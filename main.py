from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Добавляем импорт
from typing import List, Dict
import shutil
from pathlib import Path
import tempfile
import logging
import io
import torch
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)

from models import ChatRequest, ChatResponse, SimilarityMapsRequest, RetrievalResponse, RetrievalRequest
from services import QdrantService, RAGService

app = FastAPI(title="Норникель PDF Ассистент API")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники (в продакшене лучше указать конкретные домены)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все HTTP методы
    allow_headers=["*"],  # Разрешаем все заголовки
)
qdrant_service = QdrantService()
rag_service = RAGService()

@app.get("/documents", response_model=List[str])
async def get_indexed_documents():
    """Получить список всех проиндексированных документов"""
    try:
        return await qdrant_service.get_all_documents()
    except Exception as e:
        logging.error(f"Ошибка при получении списка документов: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Удалить документ из индекса"""
    try:
        result = await qdrant_service.delete_document(filename)
        if result == 0:
            return JSONResponse(content={"message": f"Документ {filename} не найден в индексе"}, status_code=404)
        return JSONResponse(content={"message": f"Документ {filename} успешно удален"})
    except Exception as e:
        logging.error(f"Ошибка при удалении документа: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    """Загрузить и проиндексировать PDF документ"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Только PDF файлы разрешены")
    
    try:
        # Проверяем, существует ли документ в индексе
        existing_documents = await qdrant_service.get_all_documents()
        if file.filename in existing_documents:
            return JSONResponse(
                content={"message": f"Файл {file.filename} уже проиндексирован"},
                status_code=200
            )
        
        # Создаём временную директорию для сохранения файла с оригинальным именем
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / file.filename
            with open(tmp_path, 'wb') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
            
            await qdrant_service.index_document(tmp_path, file.filename)
            return JSONResponse(content={"message": f"Файл {file.filename} успешно загружен и проиндексирован"})
    
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrieve_context", response_model=RetrievalResponse)
async def retrieve_context(request: RetrievalRequest):
    """Получить контекст для ответа"""
    try:
        response = await rag_service.retrieve_context(request.message, request.top_k)
        return response
    except Exception as e:
        logging.error(f"Ошибка при получении контекста: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Получить ответ от RAG системы"""
    try:
        response = await rag_service.generate_response(request.message, request.base64_image)
        return response
    except Exception as e:
        logging.error(f"Ошибка при генерации ответа: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/similarity_maps")
async def similarity_maps(request: SimilarityMapsRequest):
    """Получить схожесть между токенами запроса и изображением"""
    try:
        image = rag_service._base64_to_image(request.image)
        similarity_maps, _ = rag_service.get_similarity_maps(request.query, image, request.pooling)
        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.save(similarity_maps, buffer)
        buffer.seek(0)
        # Return a Response object with proper content type
        return Response(
            content=buffer.getvalue(),
            media_type="application/octet-stream"
        )
    except Exception as e:
        logging.error(f"Ошибка при получении схожести: {e}")
        raise HTTPException(status_code=400, detail=str(e))