import inspect
import typing as T
from typing import Iterator

import optuna
import pytest

import syftr.baselines as baselines
from syftr.llm import get_tokenizer
from syftr.studies import DEFAULT_LLMS, ParamDict, SearchSpace


@pytest.fixture(scope="session")
def optuna_study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.study.create_study(
        directions=[
            "maximize",
            "minimize",
        ],
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
        ),
    )
    return study


@pytest.fixture(params=[f"random_trial_{i}" for i in range(100)])
def search_space_and_trial_params(
    request, optuna_study
) -> T.Tuple[SearchSpace, T.Dict[str, T.Any]]:
    ss = SearchSpace()
    distributions = ss.build_distributions()
    trial = optuna_study.ask(distributions)
    params = ss.sample(trial)
    assert params
    return ss, params


@pytest.fixture(params=DEFAULT_LLMS)
def tokenizer(request):
    llm_name = request.param
    return get_tokenizer(llm_name)


def baseline_templates() -> Iterator[ParamDict]:
    for name, value in inspect.getmembers(baselines):
        if name.endswith("_TEMPLATE"):
            yield value
    for baseline in baselines.INDIVIDUAL_BASELINES:
        yield baseline


@pytest.fixture(params=baseline_templates())
def baseline_template(request):
    return request.param
