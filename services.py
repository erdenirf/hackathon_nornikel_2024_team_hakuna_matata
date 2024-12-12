from typing import List, Optional
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

from config_reader import config
from src.ColQwen2ForRAGLangchain import ColQwen2ForRAGLangchain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient
from src.pdf2image_loader import Pdf2ImageLoader
from models import ChatResponse  # Добавляем импорт моделей

SUPER_MODEL = ColQwen2ForRAGLangchain()

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=config.QDRANT_URL.get_secret_value(),
            api_key=config.QDRANT_API_KEY.get_secret_value(),
        )
        self.embeddings = SUPER_MODEL.ImageEmbeddings
        self.collection_name = config.QDRANT_COLLECTION_NAME.get_secret_value()
        self.list_documents = self._get_list_documents()

    async def get_all_documents(self) -> List[str]:
        """Получить список всех проиндексированных документов"""
        return self.list_documents

    async def delete_document(self, filename: str) -> int:
        """Удалить документ из индекса"""
        if filename not in self.list_documents:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
        )
        self.list_documents.remove(filename)
        return 1

    async def index_document(self, file_path: Path, original_filename: str):
        """Индексировать PDF документ"""
        loader = Pdf2ImageLoader(str(file_path))
        documents = loader.load()
        
        await QdrantVectorStore.afrom_documents(
            documents,
            self.embeddings,
            collection_name=self.collection_name,
            url=config.QDRANT_URL.get_secret_value(),
            api_key=config.QDRANT_API_KEY.get_secret_value()
        )
        self.list_documents.append(original_filename)

    def _get_list_documents(self) -> List[str]:
        unique_sources = set()
        offset = None

        while True:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            
            points, offset = search_result  # распаковываем результат
            
            # Если получили пустой список точек, выходим из цикла
            if not points:
                break
            
            # Добавляем source в множество
            for point in points:
                if 'metadata' in point.payload and 'source' in point.payload['metadata']:
                    unique_sources.add(point.payload['metadata']['source'])
            
            # Если offset None, значит это была последняя страница
            if offset is None:
                break
        return list(unique_sources)  # Changed from dict to list

class RAGService:
    def __init__(self):
        self.model = SUPER_MODEL
        self.embeddings = self.model.ImageEmbeddings
        self.collection_name = config.QDRANT_COLLECTION_NAME.get_secret_value()

    async def generate_response(self, message: str) -> ChatResponse:
        """Сгенерировать ответ на основе контекста"""
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=self.collection_name,
            embedding=self.embeddings,
            url=config.QDRANT_URL.get_secret_value(),
            api_key=config.QDRANT_API_KEY.get_secret_value()
        )

        results = vector_store.similarity_search_by_vector(
            self.model.TextEmbeddings.embed_query(message),
            k=5
        )

        # Получаем изображения и метаданные
        images = [result.page_content for result in results]  # Предполагается, что это base64
        sources = [result.metadata['source'] for result in results]
        pages = [result.metadata['page'] for result in results]

        # Генерируем ответ
        response = self.model.generate(message, image=self._base64_to_image(images[0]))[0]

        return ChatResponse(
            response=response,
            context_images=images,
            sources=sources,
            pages=pages
        )
    
    def get_similarity_maps(self, query: str, image: Image.Image, pooling: str = 'none'):
        """
        Generate similarity maps between query tokens and image patches.
        
        Args:
            query (str): The query text
            image (Image.Image): The input image
            pooling (str): Pooling strategy ('none', 'mean', 'max')
            
        Returns:
            tuple: Contains:
                - similarity_maps: Tensor of shape (query_length, n_patches_x, n_patches_y) if pooling='none'
                                or (n_patches_x, n_patches_y) if pooling='mean'/'max'
                - query_tokens: List of tokenized query terms
        """
        return self.model.get_similarity_maps(query, image, pooling)

    def _base64_to_image(self, base64_str: str) -> Image:
        """Конвертировать base64 в изображение"""
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        padding = 4 - (len(base64_str) % 4) if len(base64_str) % 4 else 0
        base64_str = base64_str + ('=' * padding)
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))