try:
    from colpali_engine.models import ColQwen2
except ImportError:
    raise ImportError(
        "colpali-engine package not found, please install it with `pip install colpali-engine`"
    )

class ColQwen2ForRAG(ColQwen2):
        """
        ColQwen2 model implementation that can be used both for retrieval and generation.
        Allows switching between retrieval and generation modes.
        """

        from typing import Any

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_retrieval_enabled = True

        def forward(self, *args, **kwargs) -> Any:
            """
            Forward pass that calls either Qwen2VLForConditionalGeneration.forward for generation
            or ColQwen2.forward for retrieval based on the current mode.
            """
            from colpali_engine.models import ColQwen2
            try:
                from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
            except ImportError:
                raise ImportError(
                    "transformers package not found, please install it with `pip install transformers`"
                )
            if self.is_retrieval_enabled:
                return ColQwen2.forward(self, *args, **kwargs)
            else:
                return Qwen2VLForConditionalGeneration.forward(self, *args, **kwargs)

        def generate(self, *args, **kwargs):
            """
            Generate text using Qwen2VLForConditionalGeneration.generate.
            """
            if not self.is_generation_enabled:
                raise ValueError(
                    "Set the model to generation mode by calling `enable_generation()` before calling `generate()`."
                )
            return super().generate(*args, **kwargs)

        @property
        def is_retrieval_enabled(self) -> bool:
            return self._is_retrieval_enabled

        @property
        def is_generation_enabled(self) -> bool:
            return not self.is_retrieval_enabled

        def enable_retrieval(self) -> None:
            """
            Switch to retrieval mode.
            """
            self.enable_adapters()
            self._is_retrieval_enabled = True

        def enable_generation(self) -> None:
            """
            Switch to generation mode.
            """
            self.disable_adapters()
            self._is_retrieval_enabled = False

class ColQwen2ForRAGLangchain:

    from typing import Any
    # Добавляем объявление полей класса
    model_name: str
    device_name: str
    model: Any
    processor_retrieval: Any
    device: Any
    lora_config: Any

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device_name: str = "auto"):

        from colpali_engine.utils.torch_utils import get_torch_device
        from colpali_engine.models import ColQwen2Processor

        self.model_name = model_name
        self.device_name = device_name
        
        try:
            from peft import LoraConfig
        except ImportError:
            raise ImportError(
                "peft package not found, please install it with `pip install peft`"
            )
        from typing import cast
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch package not found, please install it with `pip install torch`"
            )
        self.device = get_torch_device(device_name)
        # Get the LoRA config from the pretrained retrieval model
        self.lora_config = LoraConfig.from_pretrained(model_name)
        # Load the processors
        self.processor_retrieval = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))
        # Load the model with the loaded pre-trained adapter for retrieval
        self.model = cast(
            ColQwen2ForRAG,
            ColQwen2ForRAG.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ),
        )
        # Load the text embeddings
        self.TextEmbeddings = self.TextEmbeddingsClass(self.processor_retrieval, self.model)
        # Load the image embeddings
        self.ImageEmbeddings = self.ImageEmbeddingsClass(self.processor_retrieval, self.model)

    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.pydantic_v1 import BaseModel
    except ImportError:
        raise ImportError(
            "langchain-core package not found, please install it with `pip install langchain-core`"
        )
    
    class TextEmbeddingsClass(Embeddings, BaseModel):
        from typing import Any
        model: Any
        processor_retrieval: Any

        class Config:
            arbitrary_types_allowed = True  # Add this configuration

        def __init__(self, processor_retrieval, model):
            super().__init__()
            self.processor_retrieval = processor_retrieval
            self.model = model
        
        def embed_query(self, text: str) -> list[float]:
            
            import torch
            # Process the inputs
            batch_queries = self.processor_retrieval.process_queries([text]).to(self.model.device)
            # Forward pass
            self.model.enable_retrieval()
            result = []

            with torch.no_grad():
                query_embeddings = self.model.forward(**batch_queries)
                query_embeddings = query_embeddings.float()
                query_embeddings = torch.mean(query_embeddings, dim=1)
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
                # Сразу переносим на CPU и очищаем GPU память
                result = query_embeddings.cpu().numpy().tolist()
                del query_embeddings
                torch.cuda.empty_cache()
            
            return result
        
        def embed_documents(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
            
            import torch
            # Обработка текстов батчами
            all_text_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                if batch_texts:
                    batch_queries = self.processor_retrieval.process_queries(batch_texts).to(self.model.device)
                    self.model.enable_retrieval()
                    with torch.no_grad():
                        query_embeddings = self.model.forward(**batch_queries)
                        query_embeddings = query_embeddings.float()
                        query_embeddings = torch.mean(query_embeddings, dim=1)
                        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
                        # Сразу переносим на CPU и очищаем GPU память
                        all_text_embeddings.extend(query_embeddings.cpu().numpy().tolist())
                        del query_embeddings
                        torch.cuda.empty_cache()

            return all_text_embeddings

    class ImageEmbeddingsClass(Embeddings, BaseModel):
        from typing import Any
        model: Any
        processor_retrieval: Any

        class Config:
            arbitrary_types_allowed = True  # Add this configuration
        
        def __init__(self, processor_retrieval, model):
            super().__init__()
            self.processor_retrieval = processor_retrieval
            self.model = model
        
        from PIL import Image

        def embed_query(self, text: str) -> list[float]:

            import torch
            result = []

            img = self.base64_to_image(text)
            batch_images = self.processor_retrieval.process_images([img]).to(self.model.device)
            self.model.enable_retrieval()
            with torch.no_grad():
                image_embeddings = self.model.forward(**batch_images)
                image_embeddings = image_embeddings.float()
                image_embeddings = torch.mean(image_embeddings, dim=1)
                image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
                # Сразу переносим на CPU и очищаем GPU память
                result = image_embeddings.cpu().numpy().tolist()
                del image_embeddings
                torch.cuda.empty_cache()
            return result
        
        def embed_documents(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
            import torch
            # Convert base64 strings to PIL Images first
            images = [self.base64_to_image(text) for text in texts]
            # Process images in batches
            all_image_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i + batch_size]
                if batch_imgs:
                    batch_images = self.processor_retrieval.process_images(batch_imgs).to(self.model.device)
                    self.model.enable_retrieval()
                    with torch.no_grad():
                        image_embeddings = self.model.forward(**batch_images)
                        image_embeddings = image_embeddings.float()
                        image_embeddings = torch.mean(image_embeddings, dim=1)
                        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
                        # Сразу переносим на CPU и очищаем GPU память
                        all_image_embeddings.extend(image_embeddings.cpu().numpy().tolist())
                        del image_embeddings
                        torch.cuda.empty_cache()

            return all_image_embeddings
        
        def base64_to_image(self, text: str) -> Image.Image:
            try:
                import base64
                from io import BytesIO
                from PIL import Image
                
                image_base64 = text
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                padding = 4 - (len(image_base64) % 4) if len(image_base64) % 4 else 0
                image_base64 = image_base64 + ('=' * padding)
                img_data = base64.b64decode(image_base64)
                return Image.open(BytesIO(img_data))
            except Exception:
                raise ValueError(f"Ошибка обработки изображения. Введите изображение в формате base64. Ваш ввод: {text}")