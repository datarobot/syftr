import argparse
import asyncio
import getpass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import git
import pytz
from ray.job_submission import JobSubmissionClient

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optimization import user_confirm_delete
from syftr.ray.runtime_env import get_runtime_env
from syftr.ray.utils import ray_init
from syftr.studies import AgentStudyConfig, StudyConfig, get_default_study_name


def get_client() -> JobSubmissionClient:
    address = cfg.ray.local_endpoint if cfg.ray.local else cfg.ray.remote_endpoint
    logger.info("Connecting to Ray job submission client at `%s`", address)
    if cfg.ray.local:
        ray_init()
    return JobSubmissionClient(address)


def _get_metadata(study_config: StudyConfig) -> Dict[str, Any]:
    try:
        repo = git.Repo(cfg.paths.root_dir)
        sha = repo.head.commit.hexsha
        short_sha = repo.git.rev_parse(sha, short=True)
    except git.exc.InvalidGitRepositoryError:
        # We are not in a git repo, syftr is used as a library,
        short_sha = None
    metadata = {
        "study_name": study_config.name,
        "submitted_by": getpass.getuser(),
        "study_config": study_config.model_dump_json(),
        "config": cfg.model_dump_json(),
    }
    if short_sha:
        metadata["git_short_sha"] = short_sha
    return metadata


def _get_submission_id(metadata: Dict[str, Any]):
    utc = pytz.utc
    current_time_utc = datetime.now(utc)
    t = current_time_utc.strftime("%Y-%m-%d-%H:%M:%S")
    s = metadata["study_name"]
    u = metadata["submitted_by"]
    submission_id = f"{t}--{u}--{s}"
    short_sha = metadata.get("git_short_sha")
    if short_sha:
        submission_id += f"--{short_sha}"
    return submission_id


def start_study(
    client: JobSubmissionClient,
    study_config_file: Path,
    study_config: StudyConfig,
    delete_confirmed: bool = False,
    agentic: bool = False,
) -> str:
    metadata = _get_metadata(study_config)
    submission_id = _get_submission_id(metadata)
    runtime_env = get_runtime_env(study_config_file, delete_confirmed)

    if not agentic:
        entrypoint = (
            f"python -m syftr.tuner.qa_tuner --study-config {study_config_file.name}"
        )
    else:
        entrypoint = (
            f"python -m syftr.tuner.agent_tuner --study-config {study_config_file.name}"
        )

    if not cfg.ray.local:
        logger.info(
            "Submitting job to remote Ray cluster at `%s`", cfg.ray.remote_endpoint
        )
    else:
        logger.info("Submitting job to local Ray instance")

    job_id = client.submit_job(
        submission_id=submission_id,
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        metadata=metadata,
    )
    return job_id


def tail(client: JobSubmissionClient, job_id: str):
    async def _tail():
        async for line in client.tail_job_logs(job_id):
            print(line)

    try:
        asyncio.run(_tail())
    except Exception:
        logger.exception("Exception raised during execution of job `%s`", job_id)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received for job %s", job_id)
    finally:
        logger.info("Done tailing logs from job `%s`", job_id)
        info = client.get_job_info(job_id)
        logger.info(info)
        if not cfg.ray.local:
            stop_command = f"ray job stop --address {cfg.ray.remote_endpoint} {job_id}"
            logger.info("Client disconnected but job may still be running.")
            logger.info("Job status: %s - %s", job_id, info.status)
            logger.info("To stop the job, run `%s`", stop_command)


def main():
    parser = argparse.ArgumentParser()
    default = get_default_study_name()
    parser.add_argument(
        "--study-config",
        help=f"Path to study config yaml (default: {default})",
        default=default,
    )
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--agent",
        help="Whether to run agent optimization",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    study_config_file = Path(args.study_config)
    if not args.agent:
        study_config = StudyConfig.from_file(study_config_file)
    else:
        study_config = AgentStudyConfig.from_file(study_config_file)
    delete_confirmed = user_confirm_delete(study_config)

    cfg.ray.local = False if args.remote else cfg.ray.local
    client = get_client()
    logger.info(
        "Starting study `%s` from file `%s`",
        study_config.name,
        study_config_file.as_posix(),
    )
    job_id = start_study(
        client,
        study_config_file,
        study_config,
        delete_confirmed=delete_confirmed,
        agentic=args.agent,
    )
    logger.info("Started study `%s` with job_id `%s`", study_config.name, job_id)
    tail(client, job_id)


if __name__ == "__main__":
    main()
