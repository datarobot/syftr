import asyncio
import json
from pathlib import Path
from typing import List

import click
import pandas as pd
from aiolimiter import AsyncLimiter

from syftr.core import RandomTrial
from syftr.evaluation import (
    ExactMatchEvaluator,
    SyftrEvaluationResult,
    _aeval_retriever_pair,
)
from syftr.flows import RetrieverFlow
from syftr.instrumentation.arize import instrument_arize
from syftr.llm import get_llm
from syftr.retrievers.build import build_dummy_retriever
from syftr.studies import RetrieverStudyConfig
from syftr.tuner.qa_tuner import build_flow


@click.command()
@click.option(
    "-s",
    "--study-config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the study config yaml file",
)
@click.option("-d", "--dummy", is_flag=True, help="Use dummy retriever flow")
@click.option("-i", "--interactive", is_flag=True)
@click.option("-o", "--instrumentation", is_flag=True)
@click.option("-f", "--flow-json", type=str, help="JSON string of flow parameters")
def main(
    study_config_path: Path,
    instrumentation: bool,
    interactive: bool,
    flow_json: str,
    dummy: bool,
):
    study_config: RetrieverStudyConfig = RetrieverStudyConfig.from_file(
        study_config_path
    )
    click.secho(
        f"Loaded study config for study {study_config.name}", bold=True, fg="green"
    )

    if instrumentation:
        instrument_arize()

    if dummy:
        click.secho("Building dummy retriever flow", fg="yellow")
        retriever, docstore = build_dummy_retriever(study_config)
        flow = RetrieverFlow(
            response_synthesizer_llm=get_llm("gpt-4o-mini"),
            retriever=retriever,
            docstore=docstore,
            hyde_llm=None,
            additional_context_num_nodes=0,
            params={},
            enforce_full_evaluation=False,
        )
        click.secho("Built dummy retriever flow", fg="green")
    else:
        if flow_json:
            click.secho(f"Building flow from JSON: '{flow_json}'", fg="yellow")
            flow_params = json.loads(flow_json)

        else:
            search_space = study_config.search_space
            random_trial = RandomTrial()
            flow_params = search_space.sample(random_trial)
            flow_json = json.dumps(flow_params, ensure_ascii=False)
            click.secho(f"Building random flow '{flow_json}'", fg="yellow")

        f = build_flow(flow_params, study_config)
        assert isinstance(f, RetrieverFlow), "Flow must be a RetrieverFlow"
        flow = f
        click.secho(f"Built flow '{flow_json}'", fg="green")

    click.secho(f"Loading QA Pairs from {study_config.dataset.name}", fg="yellow")
    dataset = list(study_config.dataset.iter_examples())
    filtered_dataset = [pair for pair in dataset if pair.gold_evidence]
    if not filtered_dataset:
        raise ValueError("No QAPairs with gold_evidence found in the dataset.")
    click.secho(f"Loaded {len(filtered_dataset)} QA Pairs", fg="green")

    rate_limiter = AsyncLimiter(
        study_config.optimization.rate_limiter_max_coros,
        study_config.optimization.rate_limiter_period,
    )
    evaluator = ExactMatchEvaluator()

    results = []
    total_recall = 0.0
    total_context_length = 0.0
    for i, pair in enumerate(filtered_dataset):
        click.secho("Question: ", fg="blue", nl=False)
        click.echo(f"{pair.question}")
        click.secho("Reference Answer: ", fg="blue", nl=False)
        click.echo(f"{pair.answer}\n")
        if interactive:
            breakpoint()
        result = asyncio.run(
            _aeval_retriever_pair(
                pair,
                flow,
                [evaluator],
                rate_limiter,
                raise_on_exception=study_config.evaluation.raise_on_exception,
            )
        )
        recall = result.retriever_recall
        context_length = result.retriever_context_length
        passing = result.passing
        if result.generation_exception:
            click.secho(
                f"Generation Exception: {result.generation_exception}", fg="red"
            )
        if result.evaluation_exception:
            click.secho(
                f"Evaluation Exception: {result.evaluation_exception}", fg="red"
            )
        click.secho(
            f"Recall: {recall} with context length {context_length}",
            fg="green" if recall and recall >= 1.0 else "red",
        )
        click.secho(
            f"Evaluation Result: {'PASS' if passing else 'FAIL'}: {result.score}/5",
            fg="green" if passing else "red",
        )
        results.append(result)
        total_recall += recall if recall is not None else 0.0
        total_context_length += context_length if context_length is not None else 0.0

    avg_recall = total_recall / len(filtered_dataset)
    avg_context_length = total_context_length / len(filtered_dataset)
    click.echo(
        f"Average Recall: {avg_recall} Avg Context: {avg_context_length} in {len(filtered_dataset)} QA pairs"
    )


def write_results(results: List[SyftrEvaluationResult]):
    data = [result.model_dump() for result in results]
    df = pd.DataFrame(data)
    df.to_csv("flow-results.csv")
    return df


if __name__ == "__main__":
    main()
