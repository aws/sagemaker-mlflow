import os

import boto3


def get_default_region():
    return os.environ.get("REGION", "us-east-2")


def get_account_id():
    region = get_default_region()
    sts = boto3.client("sts", region_name=region)
    return sts.get_caller_identity()["Account"]


def get_s3_client():
    return boto3.client("s3")


def get_file_data_from_s3(bucket: str, key: str):
    s3_client = get_s3_client()
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"]
