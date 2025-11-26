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

from sagemaker_mlflow.auth import AuthBoto
from mlflow import get_tracking_uri
import os

from sagemaker_mlflow.exceptions import ResourceTypeUnsupportedException
from sagemaker_mlflow.mlflow_sagemaker_helpers import SageMakerMLflowHostMetadataProvider

AWS_SIGV4_PLUGIN_NAME = "arn"


class AuthProvider:
    """Entry Point class to using the plugin. mlflow will call get_name
    to determine the name of the plugin. get_auth will be called
    when creating the request to put a callback class that will
    generate the Sig v4 token.
    """

    def __init__(self):
        self.host_metadata_provider = SageMakerMLflowHostMetadataProvider()

    def get_name(self) -> str:
        """Returns the name of the plugin"""
        return AWS_SIGV4_PLUGIN_NAME

    def get_auth(self) -> AuthBoto:
        """Returns the callback class(AuthBoto) used for generating the SigV4 header.

        Returns:
            AuthBoto: Callback Object which will calculate the header just before request submission.
        """

        self.host_metadata_provider.set_arn(get_tracking_uri())
        assume_role_arn = os.environ.get("SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN")
        return AuthBoto(self.host_metadata_provider.region, self._get_auth_service_name(), assume_role_arn)

    def _get_auth_service_name(self):
        resource_type = self.host_metadata_provider.resource_type

        if resource_type == "mlflow-tracking-server":
            return "sagemaker-mlflow"

        if resource_type == "mlflow-app":
            return "sagemaker"

        raise ResourceTypeUnsupportedException(resource_type)
