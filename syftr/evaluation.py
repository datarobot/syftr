# flake8: noqa: E402
from syftr.event_loop import fix_asyncio

fix_asyncio()

import asyncio
import itertools
import logging
import math
import random
import typing as T
from collections import Counter, defaultdict
from datetime import datetime, timezone

import anthropic
import azure
import google
import llama_index.core.instrumentation as instrument
import numpy as np
import openai
from aiolimiter import AsyncLimiter
from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation import BaseEvaluator, CorrectnessEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms import CompletionResponse
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import MetadataMode, NodeWithScore
from optuna import TrialPruned
from rapidfuzz.fuzz import partial_ratio
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from syftr import core, custom_metrics
from syftr.configuration import EVAL__RAISE_ON_EXCEPTION
from syftr.flows import Flow, RetrieverFlow
from syftr.helpers import get_exception_report
from syftr.instrumentation.tokens import LLMCallData
from syftr.llm import get_llm
from syftr.pruning import CostPruner, ParetoPruner, RuntimePruner
from syftr.studies import AgentStudyConfig, SearchSpace, StudyConfig

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("evaluation")
log.setLevel(logging.INFO)

dispatcher = instrument.get_dispatcher()
CorrectnessEvaluator.evaluate = dispatcher.span(CorrectnessEvaluator.evaluate)  # type: ignore
CorrectnessEvaluator.aevaluate = dispatcher.span(CorrectnessEvaluator.aevaluate)  # type: ignore

RETRY_ATTEMPTS = 10
RETRY_MIN_SLEEP = 120
RETRY_MAX_SLEEP = 1800

RETRIABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.ServiceUnavailable,
    azure.core.exceptions.HttpResponseError,
    azure.core.exceptions.ServiceRequestError,
    azure.core.exceptions.ServiceResponseError,
)


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


class SyftrEvaluationResult(EvaluationResult):
    class Config:
        arbitrary_types_allowed = True

    qa_pair: T.Optional[core.QAPair] = Field(default=None, description="Q&A pair")
    run_time: T.Optional[float] = Field(
        default=np.nan, description="Flow completion time"
    )
    generation_exception: T.Optional[Exception] = Field(
        default=None, description="Exception during generation"
    )
    evaluation_exception: T.Optional[Exception] = Field(
        default=None, description="Exception during evaluation"
    )
    llm_call_data: T.List[LLMCallData] = Field(
        default_factory=list,
        description="Token counts and latencies for all LLM calls made during flow",
    )
    retriever_context_length: T.Optional[float] = Field(
        default=None, description="Total length of retrieved contexts in tokens"
    )
    retriever_recall: T.Optional[float] = Field(
        default=None, description="Retriever recall score in [0, 1]"
    )


async def exception_catcher(
    func: T.Callable,
    return_values_on_exception: T.Tuple[T.Any, ...] | None,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
    **kwargs,
):
    try:
        results = await func(**kwargs)
        if isinstance(results, tuple):
            return *results, None
        return results, None
    except RETRIABLE_EXCEPTIONS:
        raise
    except Exception as ex:
        if raise_on_exception:
            log.exception(
                ">>>>>>>>>>>>>>>>> EXCEPTION <<<<<<<<<<<<<<<<<<<<\n %s for input: %s",
                ex,
                kwargs,
            )
            raise ex
        log.exception(">>> %s for input: %s", ex, kwargs)
        ret = return_values_on_exception or ()
        ret = (ret,) if not isinstance(ret, tuple) else ret
        return *ret, ex


def should_retry(exception):
    """Add any specific tests for exceptions which should not be retried here."""
    if isinstance(exception, Exception) and "draft tokens, too large for model" in str(
        exception
    ):
        msg = f"Trial pruned due to HttpResponseError 3051: {exception.message}"
        log.warning(msg)
        raise TrialPruned(msg)
    return True


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential_jitter(
        initial=RETRY_MIN_SLEEP, max=RETRY_MAX_SLEEP, exp_base=2, jitter=RETRY_MAX_SLEEP
    ),
    reraise=True,
    retry=(
        retry_if_exception_type(RETRIABLE_EXCEPTIONS)
        | retry_if_exception_cause_type(RETRIABLE_EXCEPTIONS)
    )
    & retry_if_exception(should_retry),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
async def agenerate_pair(
    qa_pair: core.QAPair,
    flow: Flow,
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[CompletionResponse | None, float, T.List[LLMCallData], Exception | None]:
    """Get flow's answer to a question from an Q&A pair asynchronously."""
    # random wait to avoid thundering herd
    await asyncio.sleep(random.uniform(0.1, 0.5))
    async with rate_limiter:
        return await exception_catcher(
            func=flow.agenerate,
            return_values_on_exception=(False, np.nan, []),
            raise_on_exception=raise_on_exception,
            query=qa_pair.question,
        )


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential_jitter(
        initial=RETRY_MIN_SLEEP, max=RETRY_MAX_SLEEP, exp_base=2, jitter=RETRY_MAX_SLEEP
    ),
    reraise=True,
    retry=(
        retry_if_exception_type(RETRIABLE_EXCEPTIONS)
        | retry_if_exception_cause_type(RETRIABLE_EXCEPTIONS)
    )
    & retry_if_exception(should_retry),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
async def aevaluate_pair(
    qa_pair: core.QAPair,
    response: CompletionResponse,
    evaluator: BaseEvaluator,
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[EvaluationResult | None, Exception | None]:
    """Evaluate a flow response asynchronously."""
    async with rate_limiter:
        return await exception_catcher(
            func=evaluator.aevaluate,
            return_values_on_exception=(None,),
            raise_on_exception=raise_on_exception,
            query=qa_pair.question,
            response=response.text,
            reference=qa_pair.answer,
        )


async def _aeval_pair(
    qa_pair: core.QAPair,
    flow: Flow,
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> SyftrEvaluationResult:
    """Evaluate single Q&A item asynchronously."""
    response, run_time, call_data, generation_exception = await agenerate_pair(
        qa_pair, flow, rate_limiter
    )
    eval_result, evaluation_exception = None, None
    if response:
        evaluator = evaluators[0]
        eval_result, evaluation_exception = await aevaluate_pair(
            qa_pair, response, evaluator, rate_limiter, raise_on_exception
        )
    return SyftrEvaluationResult(
        qa_pair=qa_pair,
        run_time=run_time,
        generation_exception=generation_exception,
        evaluation_exception=evaluation_exception,
        llm_call_data=call_data,
        **(eval_result.model_dump() if eval_result else {}),
    )


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential_jitter(
        initial=RETRY_MIN_SLEEP, max=RETRY_MAX_SLEEP, exp_base=2, jitter=RETRY_MAX_SLEEP
    ),
    reraise=True,
    retry=(
        retry_if_exception_type(RETRIABLE_EXCEPTIONS)
        | retry_if_exception_cause_type(RETRIABLE_EXCEPTIONS)
    )
    & retry_if_exception(should_retry),
    before_sleep=before_sleep_log(log, logging.WARNING),
)
async def aretrieve_pair(
    qa_pair: core.QAPair,
    flow: RetrieverFlow,
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[T.List[NodeWithScore] | None, float, Exception | None]:
    """Get flow's retrieved documents from an Q&A pair asynchronously."""
    result, run_time, exception = await exception_catcher(
        func=flow.aretrieve,
        return_values_on_exception=(None, np.nan),
        raise_on_exception=raise_on_exception,
        query=qa_pair.question,
    )
    return result, run_time, exception


async def _aeval_retriever_pair(
    qa_pair: core.QAPair,
    flow: RetrieverFlow,
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> SyftrEvaluationResult:
    """Evaluate retrieval performance on a single Q&A item."""
    if not qa_pair.gold_evidence:
        raise ValueError("QAPair gold_evidence is empty: %s", qa_pair)
    retrieval_results, run_time, retrieval_exception = await aretrieve_pair(
        qa_pair, flow, rate_limiter
    )
    if retrieval_results:
        retrieved_contexts: T.List[str] = [
            n.node.get_content(metadata_mode=MetadataMode.NONE)
            for n in retrieval_results or []
        ]
        # Approximate token length used here for speed of implementation.
        retrieved_contexts_length = sum(len(s) // 4 for s in retrieved_contexts)
        evaluator = evaluators[0]
        result = await evaluator.aevaluate(
            contexts=retrieved_contexts,
            reference=qa_pair.gold_evidence,
        )
    return SyftrEvaluationResult(
        qa_pair=qa_pair,
        run_time=run_time,
        generation_exception=retrieval_exception,
        evaluation_exception=None,
        llm_call_data=[],
        retriever_recall=result.score,
        retriever_context_length=retrieved_contexts_length,
        passing=result.passing,
    )


async def _aeval_pair_debias(
    qa_pair: core.QAPair,
    flow: Flow,
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> SyftrEvaluationResult:
    """Evaluate single Q&A item asynchronously with an evaluator chosen at random."""
    response, run_time, call_data, generation_exception = await agenerate_pair(
        qa_pair, flow, rate_limiter
    )
    eval_result, evaluation_exception = None, None
    if response:
        evaluator = random.choice(evaluators)
        eval_result, evaluation_exception = await aevaluate_pair(
            qa_pair, response, evaluator, rate_limiter, raise_on_exception
        )

    return SyftrEvaluationResult(
        qa_pair=qa_pair,
        run_time=run_time,
        generation_exception=generation_exception,
        evaluation_exception=evaluation_exception,
        llm_call_data=call_data,
        **(eval_result.model_dump() if eval_result else {}),
    )


async def _aeval_pair_consensus(
    qa_pair: core.QAPair,
    flow: Flow,
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> SyftrEvaluationResult:
    """Evaluate single Q&A item asynchronously using provided evaluators and average the results."""
    response, run_time, call_data, generation_exception = await agenerate_pair(
        qa_pair, flow, rate_limiter
    )
    eval_result, evaluation_exception = None, None
    if response is not None:
        eval_results: T.List[EvaluationResult] = []
        for evaluator in evaluators:
            eval_result, exception = await aevaluate_pair(
                qa_pair,
                response,
                evaluator,
                rate_limiter,
                raise_on_exception,
            )
            if not eval_result:
                evaluation_exception = exception
                continue
            eval_results.append(eval_result)
        eval_result = eval_results[0]
        eval_result.passing = Counter([r.passing for r in eval_results]).most_common(1)[
            0
        ][0]
    return SyftrEvaluationResult(
        qa_pair=qa_pair,
        run_time=run_time,
        generation_exception=generation_exception,
        evaluation_exception=evaluation_exception,
        llm_call_data=call_data,
        **(eval_result.model_dump() if eval_result else {}),
    )


async def _aeval_all_pair_runner(
    pair_eval_runner: T.Callable,
    dataset: T.List[core.QAPair],
    flow: Flow,
    evaluators: T.Sequence[BaseEvaluator],
    eval_timeout: int,
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.List[SyftrEvaluationResult]:
    """Helper function to run multiple pair_eval_runners in parallel."""
    tasks = []
    for pair in dataset:
        tasks.append(
            asyncio.create_task(
                pair_eval_runner(
                    pair, flow, evaluators, rate_limiter, raise_on_exception
                )
            )
        )

    await asyncio.wait(tasks, timeout=eval_timeout)

    all_results = []
    for t in tasks:
        try:
            r = t.result()
        except asyncio.exceptions.InvalidStateError as exc:
            # Providing empty result for proper reporting.
            exc.add_note(
                f"Eval of task {t} terminated due to timeout of {eval_timeout} seconds."
            )
            r = SyftrEvaluationResult(
                qa_pair=None,
                run_time=np.nan,
                generation_exception=exc,
                llm_call_data=[],
            )
        all_results.append(r)
    return all_results


def _async_eval_runner(
    pair_eval_runner: T.Callable,
    items: T.List[core.QAPair],
    flow: Flow,
    evaluators: T.Sequence[BaseEvaluator],
    study_config: T.Union[StudyConfig, AgentStudyConfig],
    rate_limiter: AsyncLimiter,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
    pruner: ParetoPruner | None = None,
    timeout_pruner: RuntimePruner | None = None,
    cost_pruner: CostPruner | None = None,
) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
    """Evaluate Q&A items asynchronously using provided pair_eval_runner."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    prune_reason = None
    results = []
    num_batches = math.ceil(len(items) / study_config.optimization.num_eval_batch)
    for i, batch in enumerate(
        itertools.batched(items, study_config.optimization.num_eval_batch)
    ):
        batch_result = loop.run_until_complete(
            _aeval_all_pair_runner(
                pair_eval_runner,
                list(batch),
                flow,
                evaluators,
                study_config.timeouts.single_eval_timeout,
                rate_limiter,
                raise_on_exception,
            )
        )
        results.extend(batch_result)
        run_times = [
            r.run_time for r in results if r and r.run_time and not np.isnan(r.run_time)
        ]
        # Compute stats and pruners if we have successful evals
        # or, if we are over the max fail rate, proceed to calculate_metrics, which will
        # error out if we don't have any successful trials.
        # max_eval_failure_rate only applies if there are zero successful evals so far
        if (
            run_times
            or i / num_batches > study_config.optimization.max_eval_failure_rate
        ):
            current_metrics = calculate_metrics(results, study_config)

            log.info(
                "Finished evaluation batch %s/%s with %s QA pairs. Metrics: %s, Flow: %s",
                i + 1,
                num_batches,
                len(batch),
                {
                    current_metrics["objective_1_name"]: current_metrics["obj1_value"],
                    current_metrics["objective_2_name"]: current_metrics["obj2_value"],
                    "num_errors": current_metrics["num_errors"],
                },
                flow,
            )

            if timeout_pruner:
                timeout_pruner.report_and_raise_on_prune(
                    step=(i + 1) * study_config.optimization.num_eval_batch,
                    p80_time=current_metrics["p80_time"],
                )

            if cost_pruner:
                cost_pruner.report_and_raise_on_prune(
                    step=(i + 1) * study_config.optimization.num_eval_batch,
                    total_cost=current_metrics["llm_cost_total"],
                    llm_cost_mean=current_metrics["llm_cost_mean"],
                )
            if pruner:
                if pruner.report_and_prune(
                    step=(i + 1) * study_config.optimization.num_eval_batch,
                    obj1=current_metrics["obj1_value"],
                    obj2=current_metrics["obj2_value"],
                    obj1_confidence=current_metrics["obj1_confidence"],
                    obj2_confidence=current_metrics["obj2_confidence"],
                ):
                    prune_reason = "pareto"
                    break
    return results, prune_reason


def async_eval(
    items: T.List[core.QAPair],
    flow: Flow,
    study_config: T.Union[StudyConfig, AgentStudyConfig],
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    pruner: ParetoPruner | None = None,
    cost_pruner: CostPruner | None = None,
    timeout_pruner: RuntimePruner | None = None,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
    """Evaluate Q&A items asynchronously with an evaluator chosen at random for each pair."""
    return _async_eval_runner(
        _aeval_pair,
        items,
        flow,
        evaluators,
        study_config,
        rate_limiter,
        pruner=pruner,
        timeout_pruner=timeout_pruner,
        raise_on_exception=raise_on_exception,
        cost_pruner=cost_pruner,
    )


def async_eval_debias(
    items: T.List[core.QAPair],
    flow: Flow,
    study_config: T.Union[StudyConfig, AgentStudyConfig],
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    pruner: ParetoPruner | None = None,
    cost_pruner: CostPruner | None = None,
    timeout_pruner: RuntimePruner | None = None,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
    """Evaluate Q&A items asynchronously with an evaluator chosen at random for each pair."""
    return _async_eval_runner(
        _aeval_pair_debias,
        items,
        flow,
        evaluators,
        study_config,
        rate_limiter,
        raise_on_exception=raise_on_exception,
        pruner=pruner,
        timeout_pruner=timeout_pruner,
        cost_pruner=cost_pruner,
    )


def async_eval_consensus(
    items: T.List[core.QAPair],
    flow: Flow,
    study_config: T.Union[StudyConfig, AgentStudyConfig],
    evaluators: T.Sequence[BaseEvaluator],
    rate_limiter: AsyncLimiter,
    pruner: ParetoPruner | None = None,
    cost_pruner: CostPruner | None = None,
    timeout_pruner: RuntimePruner | None = None,
    raise_on_exception: bool | None = EVAL__RAISE_ON_EXCEPTION,
) -> T.Tuple[T.List[SyftrEvaluationResult], T.Optional[str]]:
    """Evaluate Q&A items asynchronously with multiple evaluators and result averaging."""
    return _async_eval_runner(
        _aeval_pair_consensus,
        items,
        flow,
        evaluators,
        study_config,
        rate_limiter,
        raise_on_exception=raise_on_exception,
        pruner=pruner,
        timeout_pruner=timeout_pruner,
        cost_pruner=cost_pruner,
    )


def validate_evaluation_data(results: T.List[SyftrEvaluationResult]):
    generation_exceptions = []
    evaluation_exceptions = []
    run_times = [
        r.run_time for r in results if r and r.run_time and not np.isnan(r.run_time)
    ]
    call_data = [result.llm_call_data for result in results]
    costs = [sum(call.cost for call in calls) for calls in call_data]

    for ex_field in "generation_exception", "evaluation_exception":
        for result in results:
            ex: Exception = getattr(result, ex_field)
            if ex:
                if ex_field == "generation_exception":
                    generation_exceptions.append(ex)
                else:
                    evaluation_exceptions.append(ex)

    if generation_exceptions or evaluation_exceptions:
        exceptions = []
        if generation_exceptions:
            exceptions.append(
                ExceptionGroup(
                    "Exceptions during generation",
                    generation_exceptions,
                )
            )
        if evaluation_exceptions:
            exceptions.append(
                ExceptionGroup(
                    "Exceptions during evaluation",
                    evaluation_exceptions,
                )
            )
        exception_group = ExceptionGroup("Trial has failed evals", exceptions)
        exception_message = get_exception_report(exception_group)
        log.warning(exception_message)
        if not run_times or not costs:
            raise Exception(exception_message)
    else:
        logging.info("The evaluation finished without exceptions")


def eval_dataset(
    study_config: T.Union[StudyConfig, AgentStudyConfig],
    dataset_iter,
    flow: Flow,
    evaluation_mode: T.Literal["single", "random", "consensus", "retriever"] = "single",
) -> T.Dict[str, T.Any]:
    eval_start = datetime.now(timezone.utc).timestamp()
    dataset = list(dataset_iter.iter_examples())
    assert len(dataset) > 2, dataset
    dataset = dataset[: study_config.optimization.num_eval_samples]
    assert evaluation_mode in {
        "single",
        "random",
        "consensus",
        "retriever",
    }, "Evaluation mode should be 'single', 'random', 'consensus', or 'retriever'."
    eval_llms = [get_llm(name) for name in study_config.evaluation.llms]
    evaluators = [CorrectnessEvaluator(llm=llm) for llm in eval_llms]
    rate_limiter = AsyncLimiter(
        study_config.optimization.rate_limiter_max_coros,
        study_config.optimization.rate_limiter_period,
    )
    timeout, pruner, costout = None, None, None
    if not getattr(flow, "enforce_full_evaluation", False):
        if study_config.optimization.use_pareto_pruner:
            pruner = ParetoPruner(
                study_name=study_config.name,
                num_warmup_steps=study_config.optimization.num_warmup_steps_pareto,
                success_rate=study_config.optimization.pareto_pruner_success_rate,
            )
        if study_config.optimization.use_cost_pruner:
            costout = CostPruner(
                num_warmup_steps=study_config.optimization.num_warmup_steps_costout,
                num_total_steps=len(dataset),
                max_cost=study_config.optimization.max_trial_cost,
            )
        if study_config.optimization.use_runtime_pruner:
            timeout = RuntimePruner(
                num_warmup_steps=study_config.optimization.num_warmup_steps_timeout,
                num_total_steps=len(dataset),
                eval_timeout=study_config.timeouts.eval_timeout,
            )

    results: T.List[SyftrEvaluationResult] = []
    match evaluation_mode:
        case "single":
            results, prune_reason = async_eval(
                dataset,
                flow,
                study_config,
                evaluators=evaluators,
                raise_on_exception=study_config.evaluation.raise_on_exception,
                pruner=pruner,
                cost_pruner=costout,
                timeout_pruner=timeout,
                rate_limiter=rate_limiter,
            )
        case "random":
            results, prune_reason = async_eval_debias(
                dataset,
                flow,
                study_config,
                evaluators=evaluators,
                raise_on_exception=study_config.evaluation.raise_on_exception,
                pruner=pruner,
                cost_pruner=costout,
                timeout_pruner=timeout,
                rate_limiter=rate_limiter,
            )
        case "consensus":
            results, prune_reason = async_eval_consensus(
                dataset,
                flow,
                study_config,
                evaluators=evaluators,
                raise_on_exception=study_config.evaluation.raise_on_exception,
                pruner=pruner,
                cost_pruner=costout,
                timeout_pruner=timeout,
                rate_limiter=rate_limiter,
            )
        case "retriever":
            # Filter the dataset to remove any qa_pairs without gold_evidence
            filtered_dataset = [pair for pair in dataset if pair.gold_evidence]
            if not filtered_dataset:
                raise ValueError("No QAPairs with gold_evidence found in the dataset.")
            results, prune_reason = _async_eval_runner(
                pair_eval_runner=_aeval_retriever_pair,
                items=filtered_dataset,
                flow=flow,
                study_config=study_config,
                evaluators=[FuzzyRecallEvaluator()],
                raise_on_exception=study_config.evaluation.raise_on_exception,
                pruner=pruner,
                timeout_pruner=timeout,
                cost_pruner=costout,
                rate_limiter=rate_limiter,
            )

    metrics = calculate_metrics(results, study_config) if results else {}
    log.info("Number of evaluations: %d", metrics.get("num_total", 0))
    log.info("Number of successful evaluations: %d", metrics.get("num_success", 0))
    log.info("Number of errored evaluations: %d", metrics.get("num_errors", 0))
    eval_end = datetime.now(timezone.utc).timestamp()
    eval_duration = eval_end - eval_start
    metrics["is_pruned"] = prune_reason is not None
    metrics["eval_start"] = eval_start
    metrics["eval_end"] = eval_end
    metrics["eval_duration"] = eval_duration
    metrics["total_qa_pairs"] = len(dataset)
    metrics["prune_reason"] = prune_reason
    metrics["cardinality"] = (
        study_config.search_space.get_cardinality()
        if isinstance(study_config.search_space, SearchSpace)
        else None
    )
    return metrics


def calculate_metrics(
    results: T.List[SyftrEvaluationResult],
    study_config: T.Union[StudyConfig, AgentStudyConfig],
) -> T.Dict[str, T.Any]:
    validate_evaluation_data(results)
    objective_1: str = study_config.optimization.objective_1_name
    objective_2: str = study_config.optimization.objective_2_name
    if study_config.is_retriever_study:
        assert objective_1 == "retriever_recall"
        assert objective_2 == "retriever_context_length"

    passing = [r.passing for r in results if r.passing in [True, False]]

    if len(passing) / len(results) < study_config.evaluation.min_reporting_success_rate:
        raise TrialPruned(
            f"Too few successful evaluations: {len(passing)} out of {len(results)}"
        )

    num_total = len(passing)
    num_errors = sum(
        int(res.passing is None)
        or int(res.generation_exception is not None)
        or int(res.evaluation_exception is not None)
        for res in results
    )
    acc = sum(passing) / num_total
    passing_std = np.std(passing)

    eval_results = {
        res.qa_pair.id: {"passing": 1 if res.passing else 0, "raw_score": res.score}
        for res in results
        if res.qa_pair is not None
    }
    f1_scores = [
        core.f1_score(result.qa_pair.answer, result.response)
        for result in results
        if result and result.qa_pair and result.response
    ]
    f1_score = np.mean(f1_scores)
    retriever_recalls = [
        r.retriever_recall for r in results if r.retriever_recall is not None
    ]
    mean_retriever_recall = (
        float(np.mean(retriever_recalls)) if retriever_recalls else 0.0
    )
    retriever_context_lengths = [
        r.retriever_context_length for r in results if r.retriever_context_length
    ]
    mean_retriever_context_length = (
        float(np.mean(retriever_context_lengths)) if retriever_context_lengths else 0.0
    )
    run_times = [
        r.run_time for r in results if r and r.run_time and not np.isnan(r.run_time)
    ]
    min_time = float(np.min(run_times))
    max_time = float(np.max(run_times))
    mean_time = float(np.mean(run_times))
    median_time = float(np.median(run_times))
    p80_time = float(np.percentile(run_times, 80))
    run_times_std = float(np.std(run_times))
    run_times_p80 = [r for r in run_times if r <= p80_time]
    run_times_p80_std = float(np.std(run_times_p80))

    latency_data = extract_llm_latency_data(results)
    cost_data = extract_cost_data(results)
    token_data = extract_token_data(results)

    if objective_1 == "accuracy":
        obj1_value = acc
    elif objective_1 == "retriever_recall":
        obj1_value = mean_retriever_recall
    else:
        raise ValueError(f"Unknown objective_1: {objective_1}.")
    obj1_confidence = custom_metrics.acc_confidence(
        accuracy=obj1_value,
        n_samples=len(passing),
        zscore=study_config.optimization.obj1_zscore,
    )
    if objective_2 == "p80_time":
        obj2_values = run_times
        obj2_value = p80_time
    elif objective_2 == "llm_cost_mean":
        call_data = [result.llm_call_data for result in results]
        costs = [sum(call.cost for call in calls) for calls in call_data]
        obj2_values = costs
        obj2_value = float(np.mean(costs))
    elif objective_2 == "retriever_context_length":
        obj2_values = retriever_context_lengths
        obj2_value = mean_retriever_context_length
    else:
        raise ValueError(
            f"Unknown objective_2: {objective_2}. Valid options are: p80_time, llm_cost_mean, retriever_context_length."
        )
    obj2_confidence = custom_metrics.lognormal_confidence(
        values=obj2_values,
        zscore=study_config.optimization.obj2_zscore,
    )
    return {
        objective_1: obj1_value,
        objective_2: obj2_value,
        "min_time": min_time,
        "max_time": max_time,
        "mean_time": mean_time,
        "median_time": median_time,
        "passing_std": passing_std,
        "f1_score": f1_score,
        "num_total": num_total,
        "num_errors": num_errors,
        "num_success": num_total - num_errors,
        "p80_time": p80_time,
        "run_times_std": run_times_std,
        "run_times_p80_std": run_times_p80_std,
        "obj1_value": obj1_value,
        "obj1_confidence": obj1_confidence,
        "obj2_value": obj2_value,
        "obj2_confidence": obj2_confidence,
        "acc_confidence": obj1_confidence,
        "objective_1_name": objective_1,
        "objective_2_name": objective_2,
        "eval_results": eval_results,
        **cost_data,
        **token_data,
        **latency_data,
    }


def extract_llm_latency_data(
    all_results: T.List[SyftrEvaluationResult],
) -> T.Dict[str, float]:
    call_data = [result.llm_call_data for result in all_results]
    per_model_latency = defaultdict(list)
    for calls in call_data:
        for call in calls:
            per_model_latency[call.llm_name].append(call.llm_call_latency)

    latency_data = {}
    for model, latencies in per_model_latency.items():
        latency_data[f"llm_latency_mean_{model}"] = float(np.mean(latencies))
        latency_data[f"llm_latency_median_{model}"] = float(np.median(latencies))
        latency_data[f"llm_latency_total_{model}"] = sum(latencies)
    return latency_data


def extract_cost_data(
    all_results: T.List[SyftrEvaluationResult],
) -> T.Dict[str, float]:
    call_data = [result.llm_call_data for result in all_results]
    per_model_costs = defaultdict(list)
    for calls in call_data:
        for call in calls:
            per_model_costs[call.llm_name].append(call.cost)
    total_costs_per_model = {
        f"llm_cost_total_{model}": sum(costs)
        for model, costs in per_model_costs.items()
    }
    run_costs = [sum(call.cost for call in calls) for calls in call_data]
    return {
        "llm_cost_total": sum(run_costs),
        "llm_cost_min": float(np.min(run_costs)),
        "llm_cost_max": float(np.max(run_costs)),
        "llm_cost_mean": float(np.mean(run_costs)),
        "llm_cost_median": float(np.median(run_costs)),
        **total_costs_per_model,
    }


def extract_token_data(
    all_results: T.List[SyftrEvaluationResult],
) -> T.Dict[str, float]:
    call_data = [result.llm_call_data for result in all_results]
    per_model_input_tokens = defaultdict(list)
    for calls in call_data:
        for call in calls:
            per_model_input_tokens[call.llm_name].append(call.input_tokens)
    total_input_tokens_per_model = {
        f"llm_input_tokens_total_{model}": sum(input_tokens)
        for model, input_tokens in per_model_input_tokens.items()
    }
    run_input_tokens = [sum(call.input_tokens for call in calls) for calls in call_data]

    per_model_output_tokens = defaultdict(list)
    for calls in call_data:
        for call in calls:
            per_model_output_tokens[call.llm_name].append(call.output_tokens)
    total_output_tokens_per_model = {
        f"llm_output_tokens_total_{model}": sum(output_tokens)
        for model, output_tokens in per_model_output_tokens.items()
    }
    run_output_tokens = [
        sum(call.output_tokens for call in calls) for calls in call_data
    ]
    return {
        "llm_input_tokens_total": sum(run_input_tokens),
        "llm_input_tokens_min": float(np.min(run_input_tokens)),
        "llm_input_tokens_max": float(np.max(run_input_tokens)),
        "llm_input_tokens_mean": float(np.mean(run_input_tokens)),
        "llm_input_tokens_median": float(np.median(run_input_tokens)),
        "llm_output_tokens_total": sum(run_output_tokens),
        "llm_output_tokens_min": float(np.min(run_output_tokens)),
        "llm_output_tokens_max": float(np.max(run_output_tokens)),
        "llm_output_tokens_mean": float(np.mean(run_output_tokens)),
        "llm_output_tokens_median": float(np.median(run_output_tokens)),
        **total_input_tokens_per_model,
        **total_output_tokens_per_model,
    }
