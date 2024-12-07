from enum import Enum

class ModelType(Enum):
    qwen2 = "qwen2"

class SpecialTokens:
    _token_map = {
        ModelType.qwen2: {
            "system_header": "<|im_start|>system",
            "user_header": "<|im_start|>user",
            "assistant_header": "<|im_start|>assistant",
            "eot": "<|im_end|>"
        }
    }

    def __init__(self, model_type: ModelType):
        if model_type not in self._token_map:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(ModelType)}")
        
        tokens = self._token_map[model_type]
        self.system_header = tokens["system_header"]
        self.user_header = tokens["user_header"]
        self.assistant_header = tokens["assistant_header"]
        self.eot = tokens["eot"]