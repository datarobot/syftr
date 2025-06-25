import asyncio
import math
import typing as T

import numpy as np
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)
from llama_index.core.prompts.mixin import PromptDictType
from rapidfuzz.fuzz import partial_ratio
from rouge_score import rouge_scorer


class RougeLRecallRetrievalMetric(BaseRetrievalMetric):
    """Custom retrieval metric using RougeL precision between expected and retrieved texts.

    Given expected text strings and a set of retrieved texts which may contain some or all of the expected
    string(s), compute the ROUGE-L precision between the retrieved and expected texts.

    Precision is 1 when the entire expected text is contained in the retrieved text, 0.90 when 90% is contained, etc.

    For each expected text we find the retrieved text with the highest precision to estimate the recall score for that text.
    The recall is then the average recall score across the expected texts.

    e.g. if we have three expected texts and ten retrieved texts, and the best precisions for each are
        [0.95, 0.30, 0.85]
    then the recall is
        sum([0.95, 0.30, 0.85]) / 3.0

    The recall is 1 if all expected texts are retrieved, 0 if there is no overlap, etc.

    This is a useful metric when the following conditions hold:
    - The IDs of the expected texts are not known
    - The expected texts are shorter in length than the retrieved documents, but still around 1+ complete sentences
    - The expected texts may be cut off between two documents in the retrieval index.
    """

    metric_name = "RougeLRecall"

    def compute(
        self,
        query: T.Optional[str] = None,
        expected_ids: T.Optional[T.List[str]] = None,
        retrieved_ids: T.Optional[T.List[str]] = None,
        expected_texts: T.Optional[T.List[str]] = None,
        retrieved_texts: T.Optional[T.List[str]] = None,
        **kwargs: T.Any,
    ) -> RetrievalMetricResult:
        """Compute metric.

        Args:
            query (Optional[str]): Query string
            expected_ids (Optional[List[str]]): Expected ids
            retrieved_ids (Optional[List[str]]): Retrieved ids
            expected_texts (Optional[List[str]]): Expected texts
            retrieved_texts (Optional[List[str]]): Retrieved texts
            **kwargs: Additional keyword arguments
        """
        if expected_ids is None or retrieved_ids is None:
            raise NotImplementedError("Scoring by Id is not currently implemented.")
        expected_texts = expected_texts or []
        retrieved_texts = retrieved_texts or []
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        total: float = 0.0
        metadata: T.Dict[str, T.Dict[str, T.Dict[str, str | float]]] = {"hits": {}}
        for expected_text in expected_texts:
            scores = [
                (text, scorer.score(text, expected_text)["rougeL"].precision)
                for text in retrieved_texts
            ]
            best_text, best_score = max(scores, key=lambda x: x[1])
            total += best_score
            metadata["hits"][expected_text] = {
                "best_text": best_text,
                "best_score": best_score,
            }
        score = total / len(expected_texts)

        result = RetrievalMetricResult(score=score, metadata=metadata)
        return result


def acc_confidence(accuracy: float, n_samples: int, zscore: float) -> float:
    if n_samples == 0:
        return np.nan
    return zscore * np.sqrt(accuracy * (1 - accuracy) / n_samples)


def lognormal_confidence(values: T.List[float], zscore: float) -> float:
    if len(values) == 0:
        return np.nan
    return zscore * float(np.std(values, ddof=1)) / np.sqrt(len(values))


class ExactMatchEvaluator(BaseEvaluator):
    """
    Evaluator that calculates exact match by comparing reference contexts
    with retrieved contexts.
    """

    async def aevaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        """
        Evaluate exact match by computing the proportion of reference contexts
        that are present in the retrieved contexts.
        """
        reference = kwargs.get("reference")

        if not reference:
            raise ValueError("Reference contexts are empty.")
        if not contexts:
            raise ValueError("Retrieved contexts are empty.")

        matched = sum(any(ref in context for context in contexts) for ref in reference)
        recall = matched / len(reference) if reference else 0.0
        return EvaluationResult(
            passing=recall > 0,
            score=recall,
        )

    def evaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        """
        Synchronous version of the evaluation method for compatibility with base class.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.aevaluate(query, response, contexts, **kwargs)
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""
        pass


class FuzzyRecallEvaluator(BaseEvaluator):
    """
    Evaluator that calculates fuzzy recall by comparing reference contexts
    with retrieved contexts using partial_ratio from rapidfuzz.
    """

    def __init__(self, threshold: float = 90.0):
        self.threshold = threshold

    async def fuzzy_match_async(self, ref: str, doc: str) -> bool:
        return await asyncio.to_thread(partial_ratio, ref, doc) >= self.threshold

    async def fuzzy_contains_async(self, ref: str, docs: T.Sequence[str]) -> bool:
        tasks = [self.fuzzy_match_async(ref, doc) for doc in docs]
        for coro in asyncio.as_completed(tasks):
            if await coro:
                return True
        return False

    async def aevaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        """
        Evaluate fuzzy recall by computing the proportion of reference contexts
        that have a fuzzy match in the retrieved contexts.
        """
        reference = kwargs.get("reference")

        if not reference:
            raise ValueError("Reference contexts are empty.")
        if not contexts:
            raise ValueError("Retrieved contexts are empty.")

        tasks = [self.fuzzy_contains_async(ref, contexts) for ref in reference]
        results = await asyncio.gather(*tasks)
        matched = sum(results)
        recall = matched / len(reference) if reference else 0.0
        return EvaluationResult(
            passing=recall > 0,
            score=recall,
        )

    def evaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        """
        Synchronous version of the evaluation method for compatibility with base class.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.aevaluate(query, response, contexts, **kwargs)
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""
        pass


class MRREvaluator(BaseEvaluator):
    """
    Evaluator that calculates Mean Reciprocal Rank (MRR) for a single query by
    finding the first matching reference in the retrieved contexts.
    """

    async def aevaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        """
        Evaluate reciprocal rank of the first relevant document.
        Assumes `reference` is a list of correct answers and `contexts` is
        a list of retrieved documents ordered by relevance.
        """
        reference = kwargs.get("reference")

        if not reference:
            raise ValueError("Reference contexts are empty.")
        if not contexts:
            raise ValueError("Retrieved contexts are empty.")

        reciprocal_rank = 0.0
        for i, context in enumerate(contexts):
            if any(ref in context for ref in reference):
                reciprocal_rank = 1.0 / (i + 1)
                break

        return EvaluationResult(
            passing=reciprocal_rank > 0,
            score=reciprocal_rank,
        )

    def evaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.aevaluate(query, response, contexts, **kwargs)
        )

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass


class NDCGEvaluator(BaseEvaluator):
    """
    Evaluator that calculates Normalized Discounted Cumulative Gain (NDCG)
    based on relevance of retrieved contexts.
    """

    def __init__(self, k: int = 10):
        self.k = k

    def _dcg(self, relevance_scores: T.List[float]) -> float:
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))

    def _get_relevance(
        self, context: str, reference: T.Union[T.Sequence[str], T.Mapping[str, float]]
    ) -> float:
        if isinstance(reference, dict):
            # Graded relevance
            return max(
                (score for ref, score in reference.items() if ref in context),
                default=0.0,
            )
        else:
            # Binary relevance
            return 1.0 if any(ref in context for ref in reference) else 0.0

    async def aevaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        reference = kwargs.get("reference")

        if not reference:
            raise ValueError("Reference contexts are empty.")
        if not contexts:
            raise ValueError("Retrieved contexts are empty.")

        top_k_contexts = contexts[: self.k]
        relevance_scores = [self._get_relevance(c, reference) for c in top_k_contexts]
        dcg = self._dcg(relevance_scores)

        # Ideal DCG: sorted relevance scores
        if isinstance(reference, dict):
            ideal_scores = sorted(reference.values(), reverse=True)[: self.k]
        else:
            ideal_scores = [1.0] * min(len(reference), self.k)

        idcg = self._dcg(ideal_scores)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return EvaluationResult(
            passing=ndcg > 0,
            score=ndcg,
        )

    def evaluate(
        self,
        query: T.Optional[str] = None,
        response: T.Optional[str] = None,
        contexts: T.Optional[T.Sequence[str]] = None,
        **kwargs: T.Any,
    ) -> EvaluationResult:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.aevaluate(query, response, contexts, **kwargs)
        )

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        pass
