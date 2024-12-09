try:
    from pdf2image import convert_from_path
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
    from PIL import Image
except ImportError:
    raise ImportError(
        "PIL package not found, please install it with `pip install Pillow`"
    )

from pathlib import Path


class Pdf2ImageLoader(BaseLoader):
    file_path: str
    metadata: dict
    images: list[Image.Image]

    def __init__(self, file_path: str):

        self.file_path = file_path
        self.metadata = {
            "source": Path(self.file_path).name
        }

    def load(self) -> list[Document]:
        self.images = convert_from_path(self.file_path, fmt='jpeg', dpi=100, thread_count=4)
        return self.convert_Images_to_Documents(self.images, self.metadata)
    
    def convert_Image_to_base64(self, image: Image.Image) -> str:
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

    def convert_Images_to_Documents(self, images: list[Image.Image], metadata: dict) -> list[Document]:
        ret_docs: list[Document] = []
        for index, image in enumerate(images):
            page_content = self.convert_Image_to_base64(image.convert('RGB'))
            doc = Document(page_content, metadata=metadata | {"page": index+1})
            ret_docs.append(doc)
        return ret_docs