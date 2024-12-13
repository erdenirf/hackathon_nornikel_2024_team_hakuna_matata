from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    base64_image: str

class ChatResponse(BaseModel):
    response: str

class RetrievalRequest(BaseModel):
    message: str

class RetrievalResponse(BaseModel):
    context_images: List[str]  # base64 encoded images
    sources: List[str]
    pages: List[int]

class SimilarityMapsRequest(BaseModel):
    query: str
    image: str
    pooling: str = 'mean'
