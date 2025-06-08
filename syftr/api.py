import asyncio
import json
import logging
import tempfile
import typing as T
from enum import Enum
from functools import cached_property
from pathlib import Path

import optuna
import pandas as pd
import yaml
from kneed import KneeLocator

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optuna_helper import get_completed_trials, get_pareto_df
from syftr.ray import submit
from syftr.studies import StudyConfig

WAIT_SECONDS = 5  # seconds


class SyftrUserAPIError(Exception):
    """An exception that is used to inform user about errors when iteracting with the API."""

    pass


class SyftrStudyStatus(Enum):
    """A status of a study on a cluster."""

    INITIALIZED = 1
    RUNNING = 2
    STOPPED = 3
    FAILED = 4
    COMPLETED = 5


_RAY_SYFTR_STATUS_MAP = {
    "PENDING": SyftrStudyStatus.INITIALIZED,
    "RUNNING": SyftrStudyStatus.RUNNING,
    "STOPPED": SyftrStudyStatus.STOPPED,
    "SUCCEEDED": SyftrStudyStatus.COMPLETED,
    "FAILED": SyftrStudyStatus.FAILED,
}


class Study:
    """Main Study object.

    The main entry point to the syftr API that carries out the interaction with studies
    and underlying infrastructure.
    """

    def __init__(
        self,
        study_config: StudyConfig,
        study_path: str | Path | None = None,
        remote: bool = False,
        debug: bool = False,
    ):
        """Intializes a study object from provided parameters locally or remotely."""
        self.study_config = study_config
        if not study_path:
            self.study_path = Path(
                cfg.paths.studies_dir / f"{self.study_config.name}.yaml"
            )
        else:
            self.study_path = Path(study_path)
        self.remote = remote
        if debug:
            logger.setLevel(logging.WARNING)

    @classmethod
    def from_file(cls, study_path: str | Path, remote: bool = False) -> T.Self:
        """Retrieves exusting study from a provided path in the filesystem."""
        study_config = StudyConfig.from_file(study_path)
        return cls(study_config, study_path, remote=remote)

    @classmethod
    def from_db(cls, study_name: str, remote: bool = False) -> T.Self:
        """Retrieves existing study specified by study_name from the database.

        In order to successfully retrieve the study metadata and build resulting object
        study should:
        1. Be stored in Optuna database
        2. Have study config saved as Optuna user attribute

        Then it will be retrieved and study config will be saved in the filesystem for
        further interaction with Ray.
        """
        try:
            study = optuna.load_study(
                study_name=study_name, storage=cfg.database.get_optuna_storage()
            )
        except KeyError:
            raise SyftrUserAPIError(f"Cannot find study {study_name} in the database.")
        if not study.user_attrs:
            raise SyftrUserAPIError(
                f"Study {study_name} has no study config in the database."
            )
        study_config = StudyConfig(**study.user_attrs)
        with tempfile.NamedTemporaryFile(
            mode="w", delete_on_close=False, delete=False
        ) as tmp_study_file:
            yaml.dump(study_config.dict(), tmp_study_file)
            study_path = tmp_study_file.name
        return cls(study_config, study_path, remote=remote)

    @property
    def status(self) -> T.Dict[str, T.Any]:
        """Return current status of the study.

        Checks Optuna storage for existing study record and then uses Ray client to check if
        the study is running.
        """
        job_status = None
        try:
            _ = optuna.load_study(
                study_name=self.study_config.name,
                storage=cfg.database.get_optuna_storage(),
            )
        except KeyError:
            pass
        else:
            job_status = SyftrStudyStatus.COMPLETED

        if hasattr(self, "job_id"):
            job_details = self.client.get_job_info(self.job_id)
            job_status = _RAY_SYFTR_STATUS_MAP[job_details.status]
        elif not job_status:
            job_status = SyftrStudyStatus.INITIALIZED

        return {
            "name": self.study_config.name,
            "dataset": self.study_config.dataset.xname,
            "remote": self.remote,
            "job_status": job_status,
        }

    @property
    def flows_df(self) -> pd.DataFrame:
        """Returns a Pandas dataframe containing all completed flows in the study."""
        try:
            df_flows = get_completed_trials(
                study=self.study_config.name, storage=cfg.database.get_optuna_storage()
            )
        except KeyError:
            raise SyftrUserAPIError(
                f"Cannot find this study in the database: `{self.study_config.name}`"
            )
        return df_flows

    @property
    def pareto_df(self) -> pd.DataFrame:
        """Return the Pareto front of a completed study as a Pandas dataframe."""
        try:
            df_pareto = get_pareto_df(self.study_config)
        except KeyError:
            raise SyftrUserAPIError(
                f"Cannot find this study in the database: `{self.study_config.name}`"
            )
        return df_pareto

    @property
    def knee_point(self) -> T.Dict[str, T.Any]:
        """Return the knee point of the Pareto front of a completed study."""
        df_pareto = self.pareto_df
        if len(df_pareto) < 3:
            raise SyftrUserAPIError(
                "Not enough points in the Pareto front to find a knee point."
            )
        knee = KneeLocator(
            df_pareto["values_1"],
            df_pareto["values_0"],
            curve="concave",
            direction="increasing",
        )
        knee_point = knee.knee
        if knee_point is None:
            raise SyftrUserAPIError("Unable to find knee point in the Pareto front.")
        df_knee = df_pareto[df_pareto["values_1"] == knee_point]
        flow_params = json.loads(df_knee["user_attrs_flow"].values[0])
        obj1_name = self.study_config.optimization.objective_1_name
        obj2_name = self.study_config.optimization.objective_2_name
        flow_metrics = {
            obj1_name: float(df_knee["values_0"].values[0]),
            obj2_name: float(df_knee["values_1"].values[0]),
        }
        return {"metrics": flow_metrics, "params": flow_params}

    def _extract_flows(self, df: pd.DataFrame) -> T.List[T.Dict[str, T.Any]]:
        """Helper method to extract flow dicts from a dataframe."""
        output = []
        obj1_name = self.study_config.optimization.objective_1_name
        obj2_name = self.study_config.optimization.objective_2_name
        df = df.sort_values(by="values_0", ascending=False)
        for _, row in df.iterrows():
            flow_params = json.loads(row["user_attrs_flow"])
            flow_metrics = {
                obj1_name: row["values_0"],
                obj2_name: row["values_1"],
            }
            output.append({"metrics": flow_metrics, "params": flow_params})
        return output

    @property
    def flows(self) -> T.List[T.Dict[str, T.Any]]:
        """Return all completed flows in the study."""
        return self._extract_flows(self.flows_df)

    @property
    def pareto_flows(self) -> T.List[T.Dict[str, T.Any]]:
        """Return the Pareto front of a completed study."""
        return self._extract_flows(self.pareto_df)

    @cached_property
    def client(self):
        return submit.get_client()

    def plot_pareto(self, save_path: str | Path | None = None):
        """Generates and optionally displays or saves the Pareto plot."""
        from syftr.plotting.insights import load_studies, pareto_plot_and_table

        study_names = [self.study_config.name]
        df, _, _ = load_studies(study_names)
        fig, _, _ = pareto_plot_and_table(df, study_names[0])

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved Pareto plot to {save_path}")

    def wait_for_completion(
        self, timeout: float | None = None, stream_logs: bool = False
    ):
        """Block until current study terminates or provided timeout reached."""
        current_status = self.status["job_status"]
        if current_status not in {
            SyftrStudyStatus.INITIALIZED,
            SyftrStudyStatus.RUNNING,
        }:
            raise SyftrUserAPIError(
                f"Study is not running. Current status: {current_status}"
            )

        async def _wait_for_status(flag: asyncio.Event):
            while True:
                if self.status["job_status"] != SyftrStudyStatus.COMPLETED:
                    await asyncio.sleep(WAIT_SECONDS)
                else:
                    flag.set()
                    break

        async def _iter_job_logs(job_logs: T.AsyncIterable, flag: asyncio.Event):
            async for lines in job_logs:
                print(lines, end="")
                if flag.is_set():
                    return

        async def _wait_for_completion():
            flag = asyncio.Event()
            tasks = [asyncio.create_task(_wait_for_status(flag))]

            if stream_logs:
                log_tailer = self.client.tail_job_logs(self.job_id)
                tasks.append(asyncio.create_task(_iter_job_logs(log_tailer, flag)))

            await asyncio.wait(tasks, timeout=timeout)
            for t in tasks:
                try:
                    t.result()
                except asyncio.exceptions.InvalidStateError:
                    self.stop()

        try:
            asyncio.get_running_loop()
            # In an interactive (async) context like Jupyter
            return _wait_for_completion()  # caller is responsible for awaiting
        except RuntimeError:
            # In a terminal or sync context
            asyncio.run(_wait_for_completion())

    def run(self):
        """Run current study."""
        if self.status["job_status"] == SyftrStudyStatus.RUNNING:
            raise SyftrUserAPIError(
                f"Study {self.study_config.name} is already running. Stop it first."
            )
        cfg.ray.local = False if self.remote else cfg.ray.local
        job_id = submit.start_study(
            self.client,
            self.study_path,
            self.study_config,
            delete_confirmed=True,
            agentic=False,
        )
        self.job_id = job_id
        dashboard_url = f"{self.client._address}/#/jobs/{job_id}"
        logger.info(f"Job started at: {dashboard_url}")

    def resume(self):
        """Resume the current study according to the configuration."""
        self.study_config.reuse_study = True
        self.run()

    def stop(self):
        """Stop running study."""
        if not hasattr(self, "job_id"):
            raise SyftrUserAPIError("This study is not running. Run it first.")
        try:
            self.client.stop_job(self.job_id)
            logger.info(f"Job {self.job_id} stopped.")
        except Exception as e:
            raise SyftrUserAPIError(f"Failed to stop job {self.job_id}. Error: {e}")

    def delete(self):
        """Remove study records and metadata from Optuna storage."""
        if self.status["job_status"] == SyftrStudyStatus.RUNNING:
            raise SyftrUserAPIError(
                "Cannot delete study %s that is running. Stop it first."
            )
        try:
            optuna.delete_study(
                study_name=self.study_config.name,
                storage=cfg.database.get_optuna_storage(),
            )
            logger.info(f"Study {self.study_config.name} deleted from the database.")
        except KeyError:
            raise SyftrUserAPIError(
                f"Study {self.study_config.name} has no study config in the database."
            )

    def __repr__(self):
        return f"Study(name={self.study_config.name}, remote={self.remote})"
