from pathlib import Path

import yaml

from syftr.ray.submit import get_client, start_study, tail
from syftr.storage import JudgeEvalHF
from syftr.studies import (
    JUDGE_LLMS,
    ConsensusCorrectnessEvaluator,
    Evaluation,
    JudgeSearchSpace,
    OptimizationConfig,
    RandomCorrectnessEvaluator,
    SingleCorrectnessEvaluator,
    StudyConfig,
)


def main():
    name = "judge-eval-study-10-multi-prompt-2"
    study_config = StudyConfig(
        name=name,
        reuse_study=False,
        recreate_study=False,
        dataset=JudgeEvalHF(),
        evaluation=Evaluation(mode="judge"),
        search_space=JudgeSearchSpace(
            single_correctness_evaluator=SingleCorrectnessEvaluator(
                response_synthesizer_llms=JUDGE_LLMS
            ),
            consensus_correctness_evaluator=ConsensusCorrectnessEvaluator(
                response_synthesizer_llms=JUDGE_LLMS,
            ),
            random_correctness_evaluator=RandomCorrectnessEvaluator(
                response_synthesizer_llms=JUDGE_LLMS,
            ),
        ),
        optimization=OptimizationConfig(
            num_trials=128,
            baselines=[],
            num_random_trials=16,
            use_individual_baselines=False,
            use_agent_baselines=False,
            use_variations_of_baselines=False,
            max_concurrent_trials=4,
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
