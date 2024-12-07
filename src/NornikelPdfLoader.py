try:
    from langchain_core.document_loaders.base import BaseLoader
    from langchain_core.documents.base import Document
except ImportError:
    raise ImportError(
        "langchain package not found, please install it with `pip install langchain-core`"
    )

class NornikelPdfLoader(BaseLoader):
    def __init__(
        self,
        file_path: str,
    ) -> None:
        """Initialize with a file path."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with `pip install pypdf`"
            )
        self.parser = PdfReader(file_path)
        self.file_path = file_path

    def load(self) -> list[Document]:

        def extract_text_from_page(page):
            import base64
            metadata = {
                "source": self.file_path,
                "page": page.page_number,
            } | self.parser.metadata

            images_base64 = []
            for image in page.images:
                # Конвертируем бинарные данные в base64
                base64_data = base64.b64encode(image.data).decode('utf-8')
                # Добавляем в формате data:image/jpeg;base64,{data}
                images_base64.append(f"data:image/jpeg;base64,{base64_data}")

            metadata["images_base64"] = images_base64

            return Document(page.extract_text(), metadata=metadata)

        return [extract_text_from_page(page) for page in self.parser.pages]