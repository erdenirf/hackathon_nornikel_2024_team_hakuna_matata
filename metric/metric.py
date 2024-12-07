from config import eval_llm, embeddings
from dataset import eval_dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (Faithfulness,
                           LLMContextPrecisionWithReference, LLMContextRecall,
                           MultiModalFaithfulness, MultiModalRelevance)

#single
faithfulness = Faithfulness(llm=eval_llm)
context_precision = LLMContextPrecisionWithReference(llm=eval_llm)
context_recall = LLMContextRecall(llm=eval_llm)

#multimodal
mm_faithfulness = MultiModalFaithfulness(llm=eval_llm)
mm_relevance = MultiModalRelevance(llm=eval_llm)

score = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        context_precision,
        context_recall 
    ],
    run_config=RunConfig(timeout=1800, max_wait=1800, thread_timeout=1800, max_retries=3),
)

