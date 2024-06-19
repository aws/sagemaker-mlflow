# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os

import boto3
import mlflow
from sagemaker_mlflow.mlflow_sagemaker_helpers import validate_and_parse_arn


def get_presigned_url(url_expiration_duration=300, session_duration=5000) -> str:
    """ Creates a presigned url

    :param url_expiration_duration: First use expiration time of the presigned url
    :param session_duration: Session duration of the presigned url

    :returns: Authorized Url

    """
    arn = validate_and_parse_arn(mlflow.get_tracking_uri())
    custom_endpoint = os.environ.get("SAGEMAKER_ENDPOINT_URL", "")
    if not custom_endpoint:
       sagemaker_client = boto3.client("sagemaker", region_name=arn.region)
    else:
        sagemaker_client = boto3.client("sagemaker", endpoint_url=custom_endpoint, region_name=arn.region)

    config = {
        "TrackingServerName": arn.resource_id,
        "ExpiresInSeconds": url_expiration_duration,
        "SessionExpirationDurationInSeconds": session_duration
    }
    response = sagemaker_client.create_presigned_mlflow_tracking_server_url(**config)
    return response["AuthorizedUrl"]
