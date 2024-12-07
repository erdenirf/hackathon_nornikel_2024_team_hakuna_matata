from langchain_community.llms.vllm import VLLMOpenAI
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from utils import ModelType, SpecialTokens

st = SpecialTokens(ModelType.qwen2)

evaluator_llm = VLLMOpenAI(
    name="evaluator",
    model="/qwen-2.5-72b",
    streaming=False,
    temperature=0.01,
    top_p=0.95,
    max_tokens=4096,
    openai_api_key="token-abc123",
    openai_api_base="..."
)
evaluator_llm = evaluator_llm.bind(stop=[st.eot])

encoder = ...

embeddings = LangchainEmbeddingsWrapper(encoder)
eval_llm = LangchainLLMWrapper(evaluator_llm)