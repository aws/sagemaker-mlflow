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

from sagemaker_mlflow.exceptions import MlflowSageMakerException, ResourceTypeUnsupportedException
import os
import logging
import warnings
from typing import Optional


class Arn:
    """Constructor for Arn Object

    .. deprecated::
        The Arn class is deprecated. Use SageMakerMLflowHostMetadataProvider instead.

    Args:
        tracking_server_arn (str): Tracking Server Arn
    """

    def __init__(self, arn: str):
        warnings.warn(
            "The Arn class is deprecated. Use SageMakerMLflowHostMetadataProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        splitted_arn = arn.split(":")
        self.partition = splitted_arn[1]
        self.service = splitted_arn[2]
        self.region = splitted_arn[3]
        self.account = splitted_arn[4]
        self.resource_type = splitted_arn[5].split("/")[0]
        self.resource_id = splitted_arn[5].split("/")[1]

    @property
    def maybe_assume_role_arn(self) -> Optional[str]:
        assume_role_arn = os.environ.get("SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN")
        if assume_role_arn:
            # Validate the assume role ARN
            try:
                arn = self.__class__(assume_role_arn)
                if arn.service != "iam" or not arn.resource_type or not arn.resource_id:
                    raise MlflowSageMakerException(f"{assume_role_arn} is not a valid arn")
            except (IndexError, AttributeError):
                raise MlflowSageMakerException(f"{assume_role_arn} is not a valid arn")
        return assume_role_arn


def validate_and_parse_arn(tracking_server_arn: str) -> Arn:
    """Validates and returns an arn from a string.

    .. deprecated::
        This function is deprecated. Use SageMakerMLflowHostMetadataProvider instead.

    Args:
        tracking_server_arn (str): Tracking Server Arn
    Returns:
        Arn: Arn Object
    """
    warnings.warn(
        "validate_and_parse_arn is deprecated. Use SageMakerMLflowHostMetadataProvider instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    arn = Arn(tracking_server_arn)
    if arn.service != "sagemaker" or not arn.resource_type or not arn.resource_id:
        raise MlflowSageMakerException(f"{tracking_server_arn} is not a valid arn")
    return arn


def get_tracking_server_url(tracking_server_arn: str) -> str:
    """Returns the url used by SageMaker MLflow

    .. deprecated::
        This function is deprecated. Use SageMakerMLflowHostMetadataProvider.construct_tracking_server_url instead.

    Args:
       tracking_server_arn (str): Tracking Server Arn
    Returns:
        str: Tracking Server URL.
    """
    warnings.warn(
        "get_tracking_server_url is deprecated. Use "
        "SageMakerMLflowHostMetadataProvider.construct_tracking_server_url instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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

    .. deprecated::
        This function is deprecated. Use SageMakerMLflowHostMetadataProvider._get_dns_suffix instead.

    Args:
        partition (str): Partition that the tracking server resides in.
    Returns:
        str: DNS suffix of the partition
    """
    warnings.warn(
        "get_dns_suffix is deprecated. Use SageMakerMLflowHostMetadataProvider._get_dns_suffix instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if partition == "aws":
        return "aws"
    elif partition == "aws-us-gov":
        return "aws"
    else:
        raise MlflowSageMakerException(f"Partition {partition} Not supported.")


class SageMakerMLflowHostMetadataProvider:
    """Provider for SageMaker MLflow host metadata and endpoint construction.

    This class handles parsing SageMaker MLflow ARNs and constructing appropriate
    tracking server URLs for both mlflow-tracking-server and mlflow-app resources.
    """

    def set_arn(self, sagemaker_mlflow_host_arn: str):
        """Parse and set the SageMaker MLflow host ARN.

        Args:
            sagemaker_mlflow_host_arn: The SageMaker MLflow ARN to parse

        Raises:
            MlflowSageMakerException: If the ARN is invalid
        """
        self.arn = sagemaker_mlflow_host_arn
        splitted_arn = sagemaker_mlflow_host_arn.split(":")
        self.partition = splitted_arn[1]
        self.service = splitted_arn[2]
        self.region = splitted_arn[3]
        self.account = splitted_arn[4]
        self.resource_type = splitted_arn[5].split("/")[0]
        self.resource_id = splitted_arn[5].split("/")[1]

        self._validate_arn()

    def construct_tracking_server_url(self):
        """Construct the tracking server URL for the configured ARN.

        Returns:
            str: The tracking server URL

        Raises:
            MlflowSageMakerException: If the ARN is invalid
            ResourceTypeUnsupportedException: If the resource type is unsupported
        """
        self._validate_arn()

        custom_endpoint = os.environ.get("SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT", "")
        if custom_endpoint:
            logging.info(f"Using custom endpoint {custom_endpoint}")
            return custom_endpoint

        return self._get_endpoint()

    def _validate_arn(self):
        """Validate that the parsed ARN is a valid SageMaker ARN.

        Raises:
            MlflowSageMakerException: If the ARN is invalid
        """
        if self.service != "sagemaker" or not self.resource_type or not self.resource_id:
            raise MlflowSageMakerException(f"{self.arn} is not a valid arn")

    def _get_dns_suffix(self) -> str:
        """Get the DNS suffix for the AWS partition.

        Returns:
            str: The DNS suffix ('aws' for standard and GovCloud partitions)

        Raises:
            MlflowSageMakerException: If the partition is not supported
        """
        if self.partition == "aws":
            return "aws"
        elif self.partition == "aws-us-gov":
            return "aws"
        else:
            raise MlflowSageMakerException(f"Partition {self.partition} Not supported.")

    def _get_endpoint(self) -> str:
        """Get the endpoint URL based on the resource type.

        Returns:
            str: The endpoint URL for the MLflow service

        Raises:
            ResourceTypeUnsupportedException: If the resource type is not supported
        """
        dns_suffix = self._get_dns_suffix()

        if self.resource_type == "mlflow-tracking-server":
            return f"https://{self.region}.experiments.sagemaker.{dns_suffix}"

        if self.resource_type == "mlflow-app":
            return f"https://mlflow.sagemaker.{self.region}.app.{dns_suffix}"

        raise ResourceTypeUnsupportedException(self.resource_type)

    @property
    def maybe_assume_role_arn(self) -> Optional[str]:
        """Get the assume role ARN from environment variable if configured.

        Returns:
            Optional[str]: The assume role ARN if configured, None otherwise

        Raises:
            MlflowSageMakerException: If the assume role ARN is invalid
        """
        assume_role_arn = os.environ.get("SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN")
        if assume_role_arn:
            # Validate the assume role ARN
            try:
                splitted_arn = assume_role_arn.split(":")
                service = splitted_arn[2]
                resource_type = splitted_arn[5].split("/")[0] if "/" in splitted_arn[5] else None
                resource_id = splitted_arn[5].split("/")[1] if "/" in splitted_arn[5] else None
                if service != "iam" or not resource_type or not resource_id:
                    raise MlflowSageMakerException(f"{assume_role_arn} is not a valid arn")
            except (IndexError, AttributeError):
                raise MlflowSageMakerException(f"{assume_role_arn} is not a valid arn")
        return assume_role_arn
