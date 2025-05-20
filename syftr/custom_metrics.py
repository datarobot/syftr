import typing as T

import numpy as np
from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
    RetrievalMetricResult,
)
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
