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

from typing import Optional
from sagemaker_mlflow.exceptions import MlflowSageMakerException
import os
import logging

TRACKING_SERVER_ARN_AND_ROLE_ARN_SEPERATOR = "#"

class Arn:

    """ Constructor for Arn Object
        Args:
            tracking_server_arn (str): Tracking Server Arn
    """
    def __init__(self, arn: str):

        self._assume_role_arn = None

        if self.arn_maybe_contains_role(arn):
            arn_parts = arn.split(TRACKING_SERVER_ARN_AND_ROLE_ARN_SEPERATOR)
            arn = arn_parts[0]
            self._assume_role_arn = arn_parts[1]

        splitted_arn = arn.split(":")
        self.partition = splitted_arn[1]
        self.service = splitted_arn[2]
        self.region = splitted_arn[3]
        self.account = splitted_arn[4]
        self.resource_type = splitted_arn[5].split("/")[0]
        self.resource_id = splitted_arn[5].split("/")[1]

    def arn_maybe_contains_role(self, tracking_server_arn: str) -> bool:
        tracking_server_arn = tracking_server_arn.split("#")
        return len(tracking_server_arn) == 2

    @property
    def maybe_assume_role_arn(self) -> Optional[str]:
        if self._assume_role_arn:
            arn = self.__class__(self._assume_role_arn)
            if (
                arn.service != "iam"
                or not arn.resource_type
                or not arn.resource_id
            ):
                raise MlflowSageMakerException(f"{self._assume_role_arn} is not a valid arn")
        return self._assume_role_arn

def validate_and_parse_arn(tracking_server_arn: str) -> Arn:
    """Validates and returns an arn from a string.

    Args:
        tracking_server_arn (str): Tracking Server Arn
    Returns:
        Arn: Arn Object
    """
    arn = Arn(tracking_server_arn)
    if (
        arn.service != "sagemaker"
        or not arn.resource_type
        or not arn.resource_id
    ):
        raise MlflowSageMakerException(f"{tracking_server_arn} is not a valid arn")
    return arn

def get_tracking_server_url(tracking_server_arn: str) -> str:
    """Returns the url used by SageMaker MLflow

    Args:
       tracking_server_arn (str): Tracking Server Arn
    Returns:
        str: Tracking Server URL.
    """
    custom_endpoint = os.environ.get("SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT", "")
    if custom_endpoint:
        logging.info(f"Using custom endpoint {custom_endpoint}")
        return custom_endpoint
    arn = validate_and_parse_arn(tracking_server_arn)
    dns_suffix = get_dns_suffix(arn.partition)
    endpoint = f"https://{arn.region}.experiments.sagemaker.{dns_suffix}"
    return endpoint


def get_dns_suffix(partition: str) -> str:
    """Returns a DNS suffix for a partition

    Args:
        partition (str): Partition that the tracking server resides in.
    Returns:
        str: DNS suffix of the partition
    """
    if partition == "aws":
        return "aws"
    else:
        raise MlflowSageMakerException(f"Partition {partition} Not supported.")
