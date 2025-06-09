import asyncio
import json
from pathlib import Path
from typing import List

import click
import pandas as pd
from aiolimiter import AsyncLimiter
from llama_index.core.evaluation import CorrectnessEvaluator

from syftr.core import RandomTrial
from syftr.evaluation import SyftrEvaluationResult, _aeval_pair
from syftr.instrumentation.arize import instrument_arize
from syftr.llm import get_llm
from syftr.studies import StudyConfig
from syftr.tuner.qa_tuner import build_flow


@click.command()
@click.option(
    "-s",
    "--study-config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the study config yaml file",
)
@click.option("-i", "--interactive", is_flag=True)
@click.option("-o", "--instrumentation", is_flag=True)
@click.option("-f", "--flow-json", type=str, help="JSON string of flow parameters")
def main(
    study_config_path: Path, flow_json: str, instrumentation: bool, interactive: bool
):
    study_config: StudyConfig = StudyConfig.from_file(study_config_path)
    click.secho(
        f"Loaded study config for study {study_config.name}", bold=True, fg="green"
    )

    if instrumentation:
        instrument_arize()

    if flow_json:
        click.secho(f"Building flow from JSON '{flow_json}'", fg="yellow")
        flow_params = json.loads(flow_json)
    else:
        search_space = study_config.search_space
        random_trial = RandomTrial()
        flow_params = search_space.sample(random_trial)
        flow_json = json.dumps(flow_params, ensure_ascii=False)
        click.secho(f"Building random flow '{flow_json}'", fg="yellow")

    flow = build_flow(flow_params, study_config)
    click.secho(f"Built flow '{flow_json}'", fg="green")

    click.secho(f"Loading QA Pairs from {study_config.dataset.name}", fg="yellow")
    dataset = list(study_config.dataset.iter_examples())
    click.secho(f"Loaded {len(dataset)} QA Pairs", fg="green")

    rate_limiter = AsyncLimiter(
        study_config.optimization.rate_limiter_max_coros,
        study_config.optimization.rate_limiter_period,
    )
    llm_name = study_config.evaluation.llms[0]
    llm = get_llm(llm_name)
    evaluator = CorrectnessEvaluator(llm=llm)

    results = []
    for i, pair in enumerate(dataset):
        click.secho("Question: ", fg="blue", nl=False)
        click.echo(f"{pair.question}")
        click.secho("Reference Answer: ", fg="blue", nl=False)
        click.echo(f"{pair.answer}\n")
        if interactive:
            breakpoint()
        result = asyncio.run(
            _aeval_pair(
                pair,
                flow,
                [evaluator],
                rate_limiter,
                raise_on_exception=study_config.evaluation.raise_on_exception,
            )
        )
        passing = result.passing
        if result.generation_exception:
            click.secho(
                f"Generation Exception: {result.generation_exception}", fg="red"
            )
        if result.evaluation_exception:
            click.secho(
                f"Evaluation Exception: {result.evaluation_exception}", fg="red"
            )

        click.echo(f"Generated answer: {result.response}")
        click.secho(
            f"Evaluation Result: {'PASS' if passing else 'FAIL'}: {result.score}/5",
            fg="green" if passing else "red",
        )
        click.secho(f"Evaluation Feedback: {result.feedback}", fg="blue")
        results.append(result)
        df = write_results(results)
        if len(df[~pd.isna(df.passing)]) > 0:
            accuracy = len(df[df.passing]) / len(df)
        else:
            accuracy = 0
        click.echo(f"Current accuracy: {accuracy} ({i + 1}/{len(dataset)} QA pairs)")
        click.echo("=" * 20)
        click.echo("\n")


def write_results(results: List[SyftrEvaluationResult]):
    data = [result.model_dump() for result in results]
    df = pd.DataFrame(data)
    df.to_csv("flow-results.csv")
    return df


if __name__ == "__main__":
    main()
