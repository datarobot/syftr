import typing as T
from pathlib import Path

import yaml
from syftr.ray.submit import get_client, start_study, tail
from syftr.storage import JudgeEvalHF
from syftr.studies import (ConsensusCorrectnessEvaluator, Evaluation,
                           JudgeSearchSpace, OptimizationConfig,
                           RandomCorrectnessEvaluator,
                           SingleCorrectnessEvaluator, StudyConfig,
                           get_llm_name_combinations)

JUDGE_LLMS: T.List[str] = [
    "gpt-4o-mini",
    "qwen-235b-a22b",
    "glm-4.5-air",
    "gpt-oss-120b-low",
    "gpt-oss-120b-medium",
    "gpt-oss-120b-high",
    "gpt-oss-20b-low",
    "gpt-oss-20b-medium",
    "gpt-oss-20b-high",
    "nemotron-super-49b",
    "qwen3-30b-a3b",
    "gemma3-27b-it",
    "phi-4-multimodal-instruct",
]


def main():
    # name = "judge-eval-consensus"
    name = "judge-eval-study-17"
    study_config = StudyConfig(
        name=name,
        reuse_study=False,
        recreate_study=True,
        dataset=JudgeEvalHF(),
        evaluation=Evaluation(mode="judge"),
        search_space=JudgeSearchSpace(
            judge_prompts=[
                "default",
                "simple",
                "out_of_ten",
                "detailed",
            ],
            single_correctness_evaluator=SingleCorrectnessEvaluator(
                response_synthesizer_llm_names=JUDGE_LLMS
            ),
            consensus_correctness_evaluator=ConsensusCorrectnessEvaluator(
                response_synthesizer_llm_combinations=get_llm_name_combinations(
                    JUDGE_LLMS, [3]
                ),
            ),
            random_correctness_evaluator=RandomCorrectnessEvaluator(
                response_synthesizer_llm_combinations=get_llm_name_combinations(
                    JUDGE_LLMS, [3]
                ),
            ),
        ),
        optimization=OptimizationConfig(
            num_trials=2000,
            baselines=[],
            num_random_trials=50,
            use_individual_baselines=False,
            use_agent_baselines=False,
            use_variations_of_baselines=False,
            max_concurrent_trials=50,
            num_eval_samples=1000,
            num_eval_batch=10,
            max_eval_failure_rate=0.05,
        ),
    )
    path = Path("studies") / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(study_config.dict(), f)

    client = get_client()
    job_id = start_study(
        client=client,
        study_config_file=path,
        study_config=study_config,
        delete_confirmed=True,
    )

    tail(client, job_id)


if __name__ == "__main__":
    main()
