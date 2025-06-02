import logging
import os
from pathlib import Path

import click

import syftr.scripts.system_check as system_check
from syftr.api import Study, SyftrUserAPIError, stop_ray_job

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _get_study(config_path: str, study_name: str):
    if (config_path and study_name) or (not config_path and not study_name):
        raise click.UsageError(
            "Provide exactly one of:\n"
            "  • a positional CONFIG_PATH to launch a new study,\n"
            "  • OR `--name STUDY_NAME` to attach to an existing study.\n"
            "(You passed: config_path=%r, study_name=%r)" % (config_path, study_name)
        )
    if config_path:
        return Study.from_file(Path(config_path))
    else:
        return Study.from_db(study_name)


@click.group()
def main():
    """syftr command‐line interface for running and managing studies."""
    pass


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
    required=False,
)
@click.option(
    "--name",
    "study_name",
    type=str,
    help=(
        "Name of an existing study to attach to or resume. "
        "If you supply a positional CONFIG_PATH, --name must be omitted."
    ),
)
@click.option(
    "--follow/--no-follow",
    default=False,
    help="Stream logs until the study completes.",
)
def run(config_path: str, study_name: str, follow: bool):
    """
    syftr run [CONFIG_PATH | --name STUDY_NAME] [--follow]

    • To launch a new study from YAML:
        syftr run path/to/config.yaml [--follow]

    • To attach to (or re‐run) an existing study:
        syftr run --name my_existing_study [--follow]
    """
    try:
        study = _get_study(config_path, study_name)
        study.run()
        if follow:
            study.wait_for_completion(stream_logs=True)
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
)
@click.option(
    "--name",
    "study_name",
    type=str,
    help=("Name of an existing study."),
)
def follow(config_path: str, study_name: str):
    """
    syftr follow [CONFIG_PATH | --name STUDY_NAME]

    Follows a running study, streaming logs until it completes.
    """
    try:
        study = _get_study(config_path, study_name)
        study.wait_for_completion(stream_logs=True)
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("job_id", type=int)
def stop(job_id: int):
    """
    syftr stop JOB_ID
    ---
    Stop a running Ray job by its numeric JOB_ID.
    """
    try:
        stop_ray_job(job_id)
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
)
@click.option(
    "--name",
    "study_name",
    type=str,
    help=("Name of an existing study."),
)
def delete(config_path: str, study_name: str):
    """
    syftr delete [CONFIG_PATH | --name STUDY_NAME]
    ---
    Delete an existing study (including all associated resources).
    """
    try:
        study = _get_study(config_path, study_name)
        study.delete()
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument(
    "results_dir",
    type=click.Path(file_okay=False, writable=True),
)
@click.option(
    "--config-path",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a YAML file to launch or resume a study.",
)
@click.option(
    "--name",
    "study_name",
    type=str,
    help="Name of an existing study to attach to or resume.",
)
def analyze(config_path: str, study_name: str, results_dir: str):
    """
    syftr analyze [--config-path CONFIG_PATH | --name STUDY_NAME] RESULTS_DIR

    Fetch Pareto/frontier data and save:
      • pareto_flows.parquet
      • all_flows.parquet
      • pareto_plot.png
    into RESULTS_DIR.
    """
    os.makedirs(results_dir, exist_ok=True)
    try:
        study = _get_study(config_path, study_name)
        study.pareto_df.to_parquet(
            Path(results_dir) / "pareto_flows.parquet", index=False
        )
        study.flows_df.to_parquet(Path(results_dir) / "all_flows.parquet", index=False)
        study.plot_pareto_frontier(Path(results_dir) / "pareto_plot.png")
        click.echo(f"Results saved under `{results_dir}`.")
    except SyftrUserAPIError as e:
        click.echo(f"✗ {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
