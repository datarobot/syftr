import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key

from syftr.configuration import REPO_ROOT, S3_TIMEOUT, cfg
from syftr.logger import logger
from syftr.utils.locks import distributed_lock


def get_parameter_value(client, name, decrypt) -> str:
    logger.info(f"Loading parameter value '{name}' with client")
    result = client.get_parameter(Name=name, WithDecryption=decrypt)
    return result["Parameter"]["Value"]


def get_aws_keypair():
    """Retrieve AWS keypair for octo-syftr user."""
    load_dotenv(REPO_ROOT / ".env")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        return {
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
            "AWS_DEFAULT_REGION": cfg.aws.region,
        }
    client = boto3.client("ssm")
    access_key = access_key or get_parameter_value(
        client, cfg.aws.access_key_ssm_path, True
    )
    secret_key = secret_key or get_parameter_value(
        client, cfg.aws.secret_key_ssm_path, True
    )
    return {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_DEFAULT_REGION": cfg.aws.region,
    }


def set_aws_key_pair(key_pair: dict):
    for k, v in key_pair.items():
        assert k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        set_key(REPO_ROOT / ".env", k, v)


def check_file_exists_s3(object_key):
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=cfg.storage.cache_bucket, Key=object_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def get_file_from_s3(object_key) -> Optional[str]:
    s3 = boto3.client("s3")

    with distributed_lock(
        f"{cfg.storage.cache_bucket}/{object_key}", timeout_s=S3_TIMEOUT
    ):
        if check_file_exists_s3(object_key):
            response = s3.get_object(Bucket=cfg.storage.cache_bucket, Key=object_key)
            # Read the file content into memory as a byte string
            file_content = response["Body"].read()
            return file_content

    logger.debug(
        f"File does not exist in bucket '{cfg.storage.cache_bucket}' with key '{object_key}'."
    )
    return None


def delete_file_from_s3(object_key) -> None:
    s3 = boto3.client("s3")

    with distributed_lock(
        f"{cfg.storage.cache_bucket}/{object_key}", timeout_s=S3_TIMEOUT
    ):
        if check_file_exists_s3(object_key):
            s3.delete_object(Bucket=cfg.storage.cache_bucket, Key=object_key)


if __name__ == "__main__":
    key_pair = get_aws_keypair()
    set_aws_key_pair(key_pair)
