import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import List

import click
import optuna
from ray.job_submission import JobSubmissionClient

import syftr.scripts.system_check as system_check
from syftr.api import Study, SyftrUserAPIError
from syftr.configuration import cfg
from syftr.optuna_helper import get_study_names
from syftr.ray import submit

__version__ = version("syftr")

logging.basicConfig(level=logging.INFO, format="%(message)s")


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="syftr")
@click.pass_context
def main(ctx):
    """syftr command‐line interface for running and managing studies."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@main.command()
def check():
    """
    syftr check
    ---
    Checks the system for required dependencies and configurations.
    """
    system_check.check()


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--follow/--no-follow",
    default=False,
    help="Stream logs until the study completes.",
)
def run(config_path: str, follow: bool):
    """
    syftr run CONFIG_PATH [--follow]

    Launches a study from YAML configuration file at CONFIG_PATH,
    optionally following the logs until completion.
    """
    try:
        study = Study.from_file(Path(config_path))
        study.run()
        if follow:
            study.wait_for_completion(stream_logs=True)
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


def _get_ray_job_ids_from_name(
    client: JobSubmissionClient, study_name: str
) -> List[str]:
    """
    Helper function to look through Ray jobs and find the job IDs matching a job name.
    """
    try:
        jobs = client.list_jobs()
    except Exception as e:
        raise SyftrUserAPIError(f"Could not contact Ray: {e}")

    matches = [
        job.job_id
        for job in jobs
        if job.metadata.get("study_name") == study_name and job.status.name == "RUNNING"
    ]
    return matches


@main.command()
@click.argument("study_name", type=str)
def stop(study_name: str):
    """
    syftr stop STUDY_NAME
    ---
    Stop all running Ray jobs whose name (or metadata.study_name) matches.
    """
    try:
        client = submit.get_client()
        job_ids = _get_ray_job_ids_from_name(client, study_name)
        if not job_ids:
            raise SyftrUserAPIError(f"No running job found with name '{study_name}'")
        for jid in job_ids:
            client.stop_job(jid)
            click.echo(f"✓ Job {jid} stopped.")
    except Exception as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("study_name", type=str)
@click.option(
    "--follow/--no-follow",
    default=False,
    help="Stream logs until the study completes.",
)
def resume(study_name: str, follow: bool):
    """
    syftr resume STUDY_NAME
    ---
    Resume a previously stopped study.
    """
    try:
        study = Study.from_db(study_name)
        study.resume()
        if follow:
            study.wait_for_completion(stream_logs=True)
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("study_name", type=str)
def status(study_name: str):
    """
    syftr status STUDY_NAME
    ---
    Get the status of a study.
    """
    # Check if the study is running in Ray
    found_ray_job = False
    client = submit.get_client()
    job_ids = _get_ray_job_ids_from_name(client, study_name)
    for job_id in job_ids:
        try:
            job_details = client.get_job_info(job_id)
        except RuntimeError:
            continue
        if job_details.status.name == "RUNNING":
            dashboard_url = f"{client._address}/#/jobs/{job_id}"
            click.echo(
                f"✓ Study '{study_name}' is currently running in Ray at {dashboard_url}"
            )
            found_ray_job = True
    if not found_ray_job:
        click.echo(f"✗ Study '{study_name}' is not currently running in Ray.")

    # Also check the Optuna DB
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=cfg.database.get_optuna_storage(),
        )
        click.echo(
            f"✓ Study '{study_name}' found in Optuna DB with {len(study.trials)} completed trials."
        )
    except KeyError:
        click.echo(f"✗ Study '{study_name}' not found in Optuna DB.")


@main.command()
@click.option(
    "--include-regex",
    type=str,
    default=".*",
    help="Include only studies whose name matches this regex.",
)
@click.option(
    "--exclude-regex",
    type=str,
    default="",
    help="Exclude studies whose name matches this regex.",
)
def studies(include_regex: str, exclude_regex: str):
    """
    syftr studies [--include-regex REGEX] [--exclude-regex REGEX]
    ---
    List studies in the Optuna DB, filtering by name using regex.
    """
    try:
        studies = get_study_names(
            include_regex=include_regex, exclude_regex=exclude_regex
        )
        click.echo(f"Found {len(studies)} studies: {studies}")
    except Exception as e:
        click.echo(f"✗ Could not list studies: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("study_name_regex", type=str)
@click.option(
    "--exclude-regex",
    type=str,
    default="",
    help="Exclude studies whose name matches this regex.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def delete(study_name_regex: str, exclude_regex: str, yes: bool):
    """
    syftr delete STUDY_NAME_REGEX [--exclude-regex REGEX] [--yes]
    ---
    Delete existing studies by name or regex.

      • To delete a single study by exact name:
          syftr delete my_study_name

      • To delete multiple by regex:
          syftr delete 'foo.*' [--exclude-regex '.*_old'] [-y]
    """
    try:
        study_names = get_study_names(
            include_regex=study_name_regex, exclude_regex=exclude_regex
        )
    except AssertionError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()

    if not study_names:
        click.echo("No studies matched that pattern. Nothing to delete.")
        return

    click.echo(f"Found {len(study_names)} study(ies) to delete: {study_names}")

    if not yes:
        prompt = f"Are you sure you want to delete these {len(study_names)} study(ies)?"
        if not click.confirm(prompt, default=False):
            click.echo("Aborted. No studies were deleted.")
            return

    storage = cfg.database.get_optuna_storage()
    errors = []
    for name in study_names:
        try:
            optuna.delete_study(
                study_name=name,
                storage=storage,
            )
            click.echo(f"✓ Deleted `{name}`.")
        except Exception as e:
            errors.append((name, str(e)))

    if errors:
        click.echo("\nThe following errors occurred while deleting:")
        for name, msg in errors:
            click.echo(f"  ✗ {name}: {msg}")
        raise click.Abort()


@main.command()
@click.argument("study_name", type=str)
@click.option(
    "--results-dir",
    "results_dir",
    type=click.Path(file_okay=False, writable=True),
    default="results",
    help="Directory to save results (default: 'results').",
)
def analyze(study_name: str, results_dir: str):
    """
    syftr analyze STUDY_NAME [--results-dir RESULTS_DIR]

    Fetch Pareto/frontier data for STUDY_NAME and write:
      • {STUDY_NAME}_pareto_flows.parquet
      • {STUDY_NAME}_all_flows.parquet
      • {STUDY_NAME}_pareto_plot.png
    into RESULTS_DIR (default: ./results).
    """
    os.makedirs(results_dir, exist_ok=True)
    try:
        study = Study.from_db(study_name)
        study.pareto_df.to_parquet(
            Path(results_dir) / f"{study_name}_pareto_flows.parquet", index=False
        )
        study.flows_df.to_parquet(
            Path(results_dir) / f"{study_name}_all_flows.parquet", index=False
        )
        study.plot_pareto(Path(results_dir) / f"{study_name}_pareto_plot.png")
        click.echo(f"✓ Results saved to `{results_dir}`.")
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
