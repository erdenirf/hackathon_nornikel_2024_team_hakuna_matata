from config import eval_llm, embeddings
from dataset import eval_dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (ResponseRelevancy, Faithfulness,
                           ContextPrecision, ContextRecall,
                           MultiModalFaithfulness, MultiModalRelevance)

#single
response_relevancy = ResponseRelevancy(llm=eval_llm, embeddings=embeddings)
faithfulness = Faithfulness(llm=eval_llm, embeddings=embeddings)
context_precision = ContextPrecision(llm=eval_llm, embeddings=embeddings)
context_recall = ContextRecall(llm=eval_llm, embeddings=embeddings)

#multimodal
mm_faithfulness = MultiModalFaithfulness(llm=eval_llm, embeddings=embeddings)
mm_relevance = MultiModalRelevance(llm=eval_llm, embeddings=embeddings)

score = evaluate(
    eval_dataset,
    metrics=[
        response_relevancy,
        faithfulness,
        context_precision,
        context_recall 
    ],
    run_config=RunConfig(timeout=1800, max_wait=1800, thread_timeout=1800, max_retries=3),
)

