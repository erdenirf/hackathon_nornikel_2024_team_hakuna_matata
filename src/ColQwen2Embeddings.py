try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.pydantic_v1 import BaseModel
except ImportError:
    raise ImportError(
        "langchain-core package not found, please install it with `pip install langchain-core`"
    )

class ColQwen2Embeddings(Embeddings, BaseModel):
    # Добавляем конфигурацию в начало класса
    class Config:
        arbitrary_types_allowed = True

    from typing import Any
    # Добавляем объявление полей класса
    model_name: str
    device_name: str
    model: Any
    processor_retrieval: Any
    device: Any
    lora_config: Any
    
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", device_name: str = "auto"):
        # Добавляем вызов конструктора BaseModel
        super().__init__(model_name=model_name, device_name=device_name)
        
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

        self.model_name = model_name
        self.device_name = device_name
        
        try:
            from colpali_engine.utils.torch_utils import get_torch_device
            from colpali_engine.models import ColQwen2Processor
        except ImportError:
            raise ImportError(
                "colpali-engine package not found, please install it with `pip install colpali-engine`"
            )
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
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL package not found, please install it with `pip install Pillow`"
        )

    def embed_query(self, text: str | Image.Image) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str | Image.Image]) -> list[list[float]]:
        from PIL import Image

        texts2 = [text for text in texts if not isinstance(text, Image.Image)]
        images = [text for text in texts if isinstance(text, Image.Image)]

        if len(texts2) > 0:
            batch_queries = self.processor_retrieval.process_queries(texts2).to(self.model.device)
        else:
            batch_queries = None
        if len(images) > 0:
            batch_images = self.processor_retrieval.process_images(images).to(self.model.device)
        else:
            batch_images = None

        # Forward pass
        self.model.enable_retrieval()
        import torch

        with torch.no_grad():
            if batch_queries is not None:
                query_embeddings = self.model.forward(**batch_queries)
                # Convert to float32 and take mean over sequence length
                query_embeddings = query_embeddings.float()
                query_embeddings = torch.mean(query_embeddings, dim=1)  # Now shape is [1, 128]
                # Normalize the embeddings
                query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
                ret_query_embeddings = query_embeddings.cpu().numpy().tolist()
            else:
                ret_query_embeddings = []

            if batch_images is not None:
                #image embeddings
                image_embeddings = self.model.forward(**batch_images)
                # Convert to float32
                image_embeddings = image_embeddings.float()
                # Take mean over sequence length
                image_embeddings = torch.mean(image_embeddings, dim=1)  # Now shape should be [batch_size, 128]
                # Normalize embeddings
                image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
                ret_image_embeddings = image_embeddings.cpu().numpy().tolist()
            else:
                ret_image_embeddings = []

        return ret_query_embeddings + ret_image_embeddings

    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL package not found, please install it with `pip install Pillow`"
        )
