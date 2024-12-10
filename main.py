from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
import shutil
from pathlib import Path
import tempfile

from models import ChatRequest, ChatResponse
from services import QdrantService, RAGService

app = FastAPI(title="Норникель PDF Ассистент API")
qdrant_service = QdrantService()
rag_service = RAGService()

@app.get("/documents", response_model=List[str])
async def get_indexed_documents():
    """Получить список всех проиндексированных документов"""
    return await qdrant_service.get_all_documents()

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Удалить документ из индекса"""
    try:
        result = await qdrant_service.delete_document(filename)
        if result == 0:
            return JSONResponse(content={"message": f"Документ {filename} не найден в индексе"}, status_code=404)
        return JSONResponse(content={"message": f"Документ {filename} успешно удален"})
    except Exception as e:
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
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Получить ответ от RAG системы"""
    try:
        response = await rag_service.generate_response(request.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))