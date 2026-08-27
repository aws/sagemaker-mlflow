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

import logging
from dataclasses import dataclass

from mlflow.utils import rest_utils
from sagemaker_mlflow.mlflow_sagemaker_helpers import SageMakerMLflowHostMetadataProvider

logger = logging.getLogger(__name__)

_SERVER_INFO_ENDPOINT = "/api/3.0/mlflow/server-info"
_PRESIGNED_UPLOAD_RUN_ID_SUPPORTED = "presigned_upload_run_id_supported"
_PRESIGNED_UPLOAD_MODEL_ID_SUPPORTED = "presigned_upload_model_id_supported"


@dataclass(frozen=True)
class PresignedUploadCapabilities:
    run_id_supported: bool
    model_id_supported: bool


def _get_host_creds(tracking_server_arn: str) -> rest_utils.MlflowHostCreds:
    provider = SageMakerMLflowHostMetadataProvider()
    provider.set_arn(tracking_server_arn)
    return rest_utils.MlflowHostCreds(
        host=provider.construct_tracking_server_url(),
        auth="arn",
    )


def get_presigned_upload_capabilities(tracking_server_arn: str) -> PresignedUploadCapabilities:
    """Read the presigned-upload request contracts advertised by the server."""
    try:
        host_creds = _get_host_creds(tracking_server_arn)
        response = rest_utils.http_request(
            host_creds,
            _SERVER_INFO_ENDPOINT,
            "GET",
            raise_on_status=False,
            max_retries=0,
        )
        if response.status_code != 200:
            return PresignedUploadCapabilities(False, False)
        server_info = response.json()
        return PresignedUploadCapabilities(
            run_id_supported=server_info.get(_PRESIGNED_UPLOAD_RUN_ID_SUPPORTED) is True,
            model_id_supported=server_info.get(_PRESIGNED_UPLOAD_MODEL_ID_SUPPORTED) is True,
        )
    except Exception as e:
        logger.warning("Failed to read presigned upload capabilities: %s", e)
        return PresignedUploadCapabilities(False, False)
