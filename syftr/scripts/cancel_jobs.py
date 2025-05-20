import argparse

from ray.job_submission import JobStatus

from syftr.configuration import cfg
from syftr.ray.submit import get_client


def cancel_jobs(
    substring: str = "<doesntmatch>",
    prefix: str = "<doesntmatch>",
    suffix: str = "<doesntmatch>",
    remote: bool = True,
):
    """Usage:

    from syftr.scripts.run_benchmarks import cancel_jobs
    cancel_jobs(prefix='bench3', remote=True)
    """
    cfg.ray.local = False if remote else cfg.ray.local

    client = get_client()
    job_details = client.list_jobs()
    jobs_to_stop = [
        job
        for job in job_details
        if job.submission_id is not None
        and (
            job.submission_id.startswith(prefix)
            or job.submission_id.endswith(suffix)
            or substring in job.submission_id
        )
        and job.status not in {JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.FAILED}
    ]

    for job in jobs_to_stop:
        client.stop_job(job.submission_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--substring",
        type=str,
        help="The substring to match in the job submission ID",
    )
    args = parser.parse_args()
    input_str = args.substring
    cancel_jobs(substring=input_str)
