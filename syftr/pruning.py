import time
import typing as T
from abc import ABC, abstractmethod

import optuna
import pandas as pd
from optuna import Study, TrialPruned
from optuna.storages import BaseStorage
from paretoset import paretoset

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optuna_helper import get_completed_trials


class SyftrPruner(ABC):
    @abstractmethod
    def report(
        self,
        step: int,
        **kwargs,
    ):
        pass

    @abstractmethod
    def should_prune(self) -> bool:
        pass


class CostPruner(SyftrPruner):
    _step: int
    _process_time: float

    def __init__(
        self,
        num_warmup_steps: int,
        num_total_steps,
        max_cost: float,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_total_steps = num_total_steps
        self.max_cost = max_cost

    def report(
        self,
        step: int,
        total_cost: float = 0.0,
        **kwargs,
    ):
        self._total_cost = total_cost
        self._step = step

    def should_prune(self) -> bool:
        if self._step < self.num_warmup_steps:
            logger.debug("Not pruning because this is a warmup step")
            return False
        return (self._total_cost * self.num_total_steps / self._step) > self.max_cost

    def report_and_raise_on_prune(
        self, step: int, total_cost: float, llm_cost_mean: float
    ):
        self.report(
            step=step,
            total_cost=total_cost,
        )
        if self.should_prune():
            msg = f"Pruned after {step} evaluations because of expected cost-out with mean_cost={llm_cost_mean}"
            logger.warning(msg)
            raise TrialPruned(msg)


class RuntimePruner(SyftrPruner):
    _step: int
    _process_time: float

    def __init__(
        self,
        num_warmup_steps: int,
        num_total_steps,
        eval_timeout: int,
        eval_start: float | None = None,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_total_steps = num_total_steps
        self.eval_timeout = eval_timeout
        self.eval_start = eval_start or time.process_time()

    def report(
        self,
        step: int,
        **kwargs,
    ):
        self._process_time = time.process_time()
        self._step = step

    def should_prune(self) -> bool:
        if self._step < self.num_warmup_steps:
            logger.debug("Not pruning because this is a warmup step")
            return False
        return (
            (self._process_time - self.eval_start) * self.num_total_steps / self._step
        ) > self.eval_timeout

    def report_and_raise_on_prune(self, step: int, p80_time: float):
        self.report(step)
        if self.should_prune():
            msg = f"Pruned after {step} evaluations because of expected timeout with p80_time={p80_time}"
            logger.warning(msg)
            raise TrialPruned(msg)


class ParetoPruner(SyftrPruner):
    _obj1: float
    _obj2: float
    _obj1_confidence: float
    _obj2_confidence: float
    _step: int

    def __init__(
        self,
        study_name,
        num_warmup_steps: int,
        success_rate: float | None = None,
        storage: str | BaseStorage | None = None,
    ):
        self.study_name = study_name
        self.num_warmup_steps = num_warmup_steps
        self.directions = ["max", "min"]
        self.num_pareto_points = 1000
        self.success_rate = success_rate
        self.storage = storage if storage else cfg.database.get_optuna_storage()

    def get_pareto_points(self, trials: pd.DataFrame) -> pd.DataFrame:
        pareto_mask = paretoset(trials[["values_0", "values_1"]], sense=self.directions)
        return trials[pareto_mask].copy()

    def point_is_dominated(
        self, pareto_points: pd.DataFrame, point: T.Tuple[float, float]
    ) -> bool:
        point_x = point[0]
        point_y = point[1]
        for pareto_trial in pareto_points[["values_0", "values_1"]].itertuples():
            trial_x = pareto_trial.values_1  # latency
            trial_y = pareto_trial.values_0  # acc
            if trial_x < point_x and trial_y >= point_y:
                return True
            if trial_x <= point_x and trial_y > point_y:
                return True
        return False

    def prune(self, study: Study) -> bool:
        if self._step < self.num_warmup_steps:
            logger.debug("Not pruning because this is a warmup step")
            return False

        df_trials: pd.DataFrame = get_completed_trials(
            study, success_rate=self.success_rate
        )
        if df_trials.empty:
            logger.debug("Not pruning because there are no other trials")
            return False

        point = (
            self._obj2 - self._obj2_confidence,  # latency
            self._obj1 + self._obj1_confidence,  # acc
        )

        pareto_points = self.get_pareto_points(df_trials)

        if self.point_is_dominated(pareto_points, point):
            return True

        return False

    def report(
        self,
        step: int,
        **kwargs,
    ):
        assert "obj1" in kwargs, "obj1 is missing"
        assert "obj2" in kwargs, "obj2 is missing"
        assert "obj1_confidence" in kwargs, "'obj1_confidence' is missing"
        assert "obj2_confidence" in kwargs, "'obj2_confidence' is missing"
        assert step, "step is missing"

        self._step = step
        self._obj1 = kwargs["obj1"]
        self._obj2 = kwargs["obj2"]
        self._obj1_confidence = kwargs["obj1_confidence"]
        self._obj2_confidence = kwargs["obj2_confidence"]

    def should_prune(self) -> bool:
        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        return self.prune(study)

    def report_and_prune(
        self,
        step: int,
        obj1: float,
        obj2: float,
        obj1_confidence: float,
        obj2_confidence: float,
    ):
        self.report(
            step=step,
            obj1=obj1,
            obj2=obj2,
            obj1_confidence=obj1_confidence,
            obj2_confidence=obj2_confidence,
        )
        if self.should_prune():
            msg = f"Pruned after{step} batch because of lower-than-expected accuracy: {obj1}"
            logger.warning(msg)
            return True
        return False
