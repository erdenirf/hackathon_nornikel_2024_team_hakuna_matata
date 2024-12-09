try:
    from pdf2image import convert_from_path, convert_from_bytes
except ImportError:
    raise ImportError(
        "pdf2image package not found, please install it with `pip install pdf2image`"
    )

try:
    from langchain_core.document_loaders.base import BaseLoader
    from langchain_core.documents.base import Document
except ImportError:
    raise ImportError(
        "langchain package not found, please install it with `pip install langchain-core`"
    )

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError(
        "pypdf package not found, please install it with `pip install pypdf`"
    )

try:
    from PIL.Image import Image
except ImportError:
    raise ImportError(
        "PIL package not found, please install it with `pip install Pillow`"
    )

from typing import Any
from pathlib import Path


def convert_Image_to_base64(image: Image) -> str:
    import base64
    import io
    # Save image to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    # Get the bytes data and convert to base64
    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # Format as data URL
    base64_str = f"data:image/jpeg;base64,{base64_data}"
    return base64_str

def convert_Images_to_Documents(images: list[Image], metadata: dict) -> list[Document]:
    ret_docs: list[Document] = []
    for index, image in enumerate(images):
        page_content = convert_Image_to_base64(image.convert('RGB'))
        doc = Document(page_content, metadata=metadata | {"page": index+1})
        ret_docs.append(doc)
    return ret_docs

class Pdf2ImageLoader(BaseLoader):
    file_path: str
    parser: Any
    metadata: dict
    images: list[Image]

    def __init__(self, file_path: str):

        self.file_path = file_path
        self.parser = PdfReader(file_path)
        self.metadata = {
            "source": Path(self.file_path).name
        }
        if self.parser.metadata is not None:
            self.metadata |= self.parser.metadata

    def load(self) -> list[Document]:
        self.images = convert_from_path(self.file_path, fmt='jpeg')
        return convert_Images_to_Documents(self.images, self.metadata)
    
class Pdf2ImageLoaderBytes(BaseLoader):
    bytes: Any
    metadata: dict
    images: list[Image]

    def __init__(self, bytes: Any):
        self.bytes = bytes
        self.metadata = {
            "source": ""
        }

    def load(self) -> list[Document]:
        self.images = convert_from_bytes(self.bytes, fmt='jpeg')
        return convert_Images_to_Documents(self.images, self.metadata)