import asyncio
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import List

import click
import optuna
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray.job_submission import JobSubmissionClient

import syftr.scripts.system_check as system_check
from syftr.api import Study, SyftrUserAPIError
from syftr.configuration import cfg
from syftr.optuna_helper import get_study_names
from syftr.plotting.insights import create_pdf_report
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
) -> List[JobDetails]:
    """
    Helper function to look through Ray jobs and find the job IDs matching a job name.
    """
    try:
        jobs = client.list_jobs()
    except Exception as e:
        raise SyftrUserAPIError(f"Could not contact Ray: {e}")

    # Filter for jobs that have metadata with the study name and have valid job IDs
    matches = [job for job in jobs if job.metadata.get("study_name")]
    matches = [job for job in matches if job.job_id]
    return matches


@main.command()
@click.argument("study_name", type=str)
def stop(study_name: str):
    """
    syftr stop STUDY_NAME
    ---
    Stop all running Ray jobs with STUDY_NAME.
    """
    try:
        client = submit.get_client()
        jobs = _get_ray_job_ids_from_name(client, study_name)
        job_ids = [jid.job_id for jid in jobs if jid.status.name == "RUNNING"]
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
    client = submit.get_client()
    jobs = _get_ray_job_ids_from_name(client, study_name)
    running_jobs = [job for job in jobs if job.status.name == "RUNNING"]
    if running_jobs:
        for job in running_jobs:
            dashboard_url = f"{client._address}/#/jobs/{job.job_id}"
            click.echo(f"✓ Job '{study_name}' is running in Ray at {dashboard_url}")
    else:
        for job in jobs:
            dashboard_url = f"{client._address}/#/jobs/{job.job_id}"
            click.echo(
                f"✓ Job '{study_name}' found in Ray at {dashboard_url} with status {job.status.name}"
            )
    if not jobs:
        click.echo(f"✗ No Ray jobs found for '{study_name}'.")

    # Also check the Optuna DB
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=cfg.database.get_optuna_storage(),
        )
        from optuna.trial import TrialState

        state_counts = {state.name: 0 for state in TrialState}
        for t in study.trials:
            state_counts[t.state.name] += 1

        click.echo(
            f"✓ Study '{study_name}' found in Optuna with {len(study.trials)} total trials:"
        )
        for state_name, count in state_counts.items():
            if count > 0:
                click.echo(f"  • {count} {state_name} trial(s)")

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
@click.option(
    "-p",
    "--save-pareto-plot/--no-save-pareto-plot",
    default=False,
    help="Save pareto plot to {RESULTS_DIR}/{STUDY_NAME}_pareto_plot.png.",
)
@click.option(
    "-f",
    "--save-flows-df/--no-save-flows-df",
    default=False,
    help="Save flows dataframes to {RESULTS_DIR}/{STUDY_NAME}_flows.parquet.",
)
@click.option(
    "-r",
    "--save-report/--no-save-report",
    default=False,
    help="Save study report to {RESULTS_DIR}/{STUDY_NAME}_report.pdf.",
)
def analyze(
    study_name: str,
    results_dir: str,
    save_pareto_plot: bool,
    save_flows_df: bool,
    save_report: bool,
):
    """
    syftr analyze STUDY_NAME [--results-dir RESULTS_DIR] [--save-pareto-plot] [--save-flows-df] [--save-report]

    Fetch Pareto frontier data for STUDY_NAME and print out pareto-optimal flows.
    Optionally save the Pareto plot, flows dataframes, and a full report to RESULTS_DIR.
    """
    os.makedirs(results_dir, exist_ok=True)
    try:
        study = Study.from_db(study_name)
        click.echo("✓ Loaded study from database. Printing Pareto-optimal flows...")
        for flow in study.pareto_flows:
            metrics = ", ".join(f"{k} {v:.3f}" for k, v in flow["metrics"].items())
            parsed_params = (
                json.loads(flow["params"])
                if isinstance(flow["params"], str)
                else flow["params"]
            )
            click.echo(f"• {metrics}: {json.dumps(parsed_params)}")

        if save_flows_df:
            pareto_df_path = Path(results_dir) / f"{study_name}_pareto_flows.parquet"
            study.pareto_df.to_parquet(pareto_df_path, index=False)
            click.secho(
                f"✓ Saved pareto flow dataframe to `{pareto_df_path}`.", fg="green"
            )

            all_flows_df_path = Path(results_dir) / f"{study_name}_all_flows.parquet"
            study.flows_df.to_parquet(all_flows_df_path, index=False)
            click.secho(
                f"✓ Saved all flows dataframe to `{all_flows_df_path}`.", fg="green"
            )

        if save_pareto_plot:
            pareto_plot_path = Path(results_dir) / f"{study_name}_pareto_plot.png"
            study.plot_pareto(pareto_plot_path)
            click.secho(f"✓ Saved pareto plot to `{pareto_plot_path}`.", fg="green")

        if save_report:
            pdf_filename = Path(results_dir) / f"{study_name}_report.pdf"
            asyncio.run(
                create_pdf_report(
                    study_name,
                    all_study_names=[study_name],
                    pdf_filename=pdf_filename,
                    insights_prefix="",
                    show=False,
                )
            )
            click.secho(f"✓ Saved full report to {pdf_filename}", fg="green")
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
