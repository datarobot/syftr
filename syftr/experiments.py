import asyncio
import typing as T

from ray.job_submission import JobStatus

from syftr.ray.submit import get_client


def derived_representer(dumper, data):
    return dumper.represent_dict({"description": data.description})


async def iter_job_logs(job_logs: T.AsyncIterable):
    async for lines in job_logs:
        print(lines, end="")


async def iter_all_job_logs(tailers: T.List[T.AsyncIterable]):
    log_iters = [iter_job_logs(tailer) for tailer in tailers]
    await asyncio.gather(*log_iters)


def attach_logs(
    prefix: str = "<doesntmatch>",
    suffix: str = "<doesntmatch>",
    substring: str = "<doesntmatch>",
):
    """Connect to all jobs that match the given prefix, suffix, or substring"""
    client = get_client()
    job_details = client.list_jobs()
    jobs_to_tail = [
        job
        for job in job_details
        if (
            job.submission_id is not None
            and (
                job.submission_id.startswith(prefix)
                or job.submission_id.endwith(suffix)
                or substring in job.submission_id
            )
            and job.status
            not in {JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.FAILED}
        )
    ]
    log_tailers = [client.tail_job_logs(job.job_id) for job in jobs_to_tail]
    asyncio.run(iter_all_job_logs(log_tailers))
