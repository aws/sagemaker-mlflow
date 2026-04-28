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

from mlflow.utils import rest_utils
from sagemaker_mlflow.mlflow_sagemaker_helpers import SageMakerMLflowHostMetadataProvider

logger = logging.getLogger(__name__)

_PRESIGNED_UPLOAD_ENDPOINT = "/api/2.0/mlflow/artifacts/presigned-upload-url"


def presigned_upload_supported(tracking_server_arn: str) -> bool:
    """Probe whether the tracking server supports the presigned upload endpoint."""
    provider = SageMakerMLflowHostMetadataProvider()
    provider.set_arn(tracking_server_arn)
    host_creds = rest_utils.MlflowHostCreds(
        host=provider.construct_tracking_server_url(),
        auth="arn",
    )
    try:
        response = rest_utils.http_request(
            host_creds,
            _PRESIGNED_UPLOAD_ENDPOINT,
            "POST",
            json={"run_id": "probe", "path": "probe.txt"},
            raise_on_status=False,
            max_retries=0,
        )
        # 404 = endpoint doesn't exist, 501 = not implemented
        # Anything else (400, 403, etc.) means the endpoint exists
        return response.status_code not in (404, 501)
    except Exception as e:
        logger.warning("Failed to probe presigned upload endpoint: %s", e)
        return False
