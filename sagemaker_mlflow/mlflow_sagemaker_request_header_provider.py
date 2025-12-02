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


from typing import Dict

from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow import get_tracking_uri

from sagemaker_mlflow.exceptions import ResourceTypeUnsupportedException
from sagemaker_mlflow.mlflow_sagemaker_helpers import SageMakerMLflowHostMetadataProvider


class MlflowSageMakerRequestHeaderProvider(RequestHeaderProvider):
    """RequestHeaderProvider provided through plugin system"""

    def __init__(self):
        self.host_metadata_provider = SageMakerMLflowHostMetadataProvider()
        super().__init__()

    def in_context(self):
        """Activates the plugin"""
        return True

    def request_headers(self):
        """Returns plugin headers used by SageMaker MLflow

        Returns:
            dict: Dictionary containing the headers that are needed for routing.
        """

        self.host_metadata_provider.set_arn(get_tracking_uri())
        return self._get_arn_header()

    def _get_arn_header(self) -> Dict[str, str]:
        resource_type = self.host_metadata_provider.resource_type
        arn = self.host_metadata_provider.arn

        if resource_type == "mlflow-tracking-server":
            return {"x-mlflow-sm-tracking-server-arn": arn}

        if resource_type == "mlflow-app":
            return {"x-sm-mlflow-app-arn": arn}

        raise ResourceTypeUnsupportedException(resource_type)
