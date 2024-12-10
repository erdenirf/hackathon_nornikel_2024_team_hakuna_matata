from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context_images: List[str]  # base64 encoded images
    sources: List[str]
    pages: List[int]