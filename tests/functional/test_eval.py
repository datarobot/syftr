import pytest
from optuna import TrialPruned

from syftr.evaluation import eval_dataset
from syftr.pruning import CostPruner, ParetoPruner


def test_eval_react_rag_agent_normal(react_agent_flow):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = False
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = False
    results = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode="single",
    )
    assert results["num_errors"] == 0
    assert results["num_success"] == results["num_total"]
    assert results["eval_duration"] > 0
    assert results["f1_score"] >= 0
    assert results["accuracy"] >= 0


def test_eval_react_rag_agent_bias_mitigation(react_agent_flow):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = False
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = False
    results = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode="random",
    )
    assert results["num_errors"] == 0
    assert results["num_success"] == results["num_total"]
    assert results["eval_duration"] > 0
    assert results["f1_score"] >= 0
    assert results["accuracy"] >= 0


def test_eval_react_rag_agent_variance_mitigation(react_agent_flow):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = False
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = False
    results = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode="consensus",
    )
    assert results["num_errors"] == 0
    assert results["num_success"] == results["num_total"]
    assert results["eval_duration"] > 0
    assert results["f1_score"] >= 0
    assert results["accuracy"] >= 0


def test_eval_react_rag_agent_bias_variance_mitigation(react_agent_flow):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = False
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = False
    with pytest.raises(AssertionError):
        eval_dataset(
            study_config=study_config,
            dataset_iter=study_config.dataset,
            flow=flow,
            evaluation_mode="normal",
        )


def test_eval_react_rag_agent_variance_mitigation_pruning(
    react_agent_flow, monkeypatch
):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = True
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = False
    study_config.optimization.num_warmup_steps_pareto = 1
    study_config.optimization.num_warmup_steps_costout = 1
    study_config.optimization.num_eval_batch = 5
    study_config.timeouts.single_eval_timeout = 100

    # This is needed when multiple tests are run to prevent pruner from accumulating data.
    def should_prune_pareto(self, *args, **kwargs):
        return True

    def should_prune_cost(self, *args, **kwargs):
        return False

    monkeypatch.setattr(ParetoPruner, "should_prune", should_prune_pareto)
    monkeypatch.setattr(CostPruner, "should_prune", should_prune_cost)
    results = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode="consensus",
    )
    assert results["num_errors"] == 0
    assert results["num_success"] == results["num_total"]
    assert results["is_pruned"] == True
    assert results["accuracy"] < study_config.optimization.pareto_pruner_success_rate


def test_eval_react_rag_agent_variance_mitigation_timeout(react_agent_flow):
    flow, study_config = react_agent_flow
    study_config.optimization.use_pareto_pruner = False
    study_config.optimization.use_cost_pruner = False
    study_config.optimization.use_runtime_pruner = True
    study_config.timeouts.eval_timeout = 1
    study_config.evaluation.raise_on_exception = True

    with pytest.raises(TrialPruned):
        eval_dataset(
            study_config=study_config,
            dataset_iter=study_config.dataset,
            flow=flow,
            evaluation_mode="consensus",
        )
