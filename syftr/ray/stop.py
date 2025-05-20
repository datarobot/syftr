import argparse

from ray.job_submission import JobSubmissionClient

from syftr.configuration import cfg


def stop_job():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        help="Submission ID",
    )
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    endpoint = cfg.ray.remote_endpoint if args.remote else cfg.ray.local_endpoint
    client = JobSubmissionClient(endpoint)
    client.stop_job(args.id)


if __name__ == "__main__":
    stop_job()
