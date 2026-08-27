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
import os
import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils import rest_utils
from mlflow.utils.request_utils import cloud_storage_http_request

from sagemaker_mlflow.host_creds import get_host_creds as _get_host_creds

logger = logging.getLogger(__name__)

_SAGEMAKER_PRESIGNED_URL_UPLOAD_ENV_VAR = "SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED"

_PRESIGNED_UPLOAD_ENDPOINT = "/api/2.0/mlflow/artifacts/presigned-upload-url"


class _UploadResourceType(Enum):
    RUN = "run_id"
    LOGGED_MODEL = "model_id"


@dataclass(frozen=True)
class _UploadTarget:
    resource_type: _UploadResourceType
    resource_id: str


class S3PresignedArtifactRepository(S3ArtifactRepository):
    """S3 artifact repository with optional presigned URL upload support.

    Extends S3ArtifactRepository to optionally upload artifacts via presigned URLs
    obtained from the MLflow tracking server. This avoids the need for the client
    to have direct S3 write credentials.

    When the SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED environment variable is set to
    "true" and a tracking URI is available, recognized run and logged-model artifact
    uploads use presigned URLs. Unsupported targets and presigned upload failures
    propagate to the caller — there is no silent fallback to direct S3.

    When the environment variable is not set (the default), or when no tracking URI
    is available, behavior is identical to the parent S3ArtifactRepository.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_presigned: bool = os.environ.get(_SAGEMAKER_PRESIGNED_URL_UPLOAD_ENV_VAR, "").lower() == "true"

    def _should_use_presigned(self) -> bool:
        """Check whether presigned upload should be attempted for this call."""
        return self._use_presigned and self.tracking_uri is not None

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        if self._should_use_presigned():
            upload_target = self._extract_upload_target()
            self._upload_via_presigned_url(upload_target, local_file, artifact_path)
        else:
            super().log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        if not self._should_use_presigned():
            super().log_artifacts(local_dir, artifact_path)
            return

        upload_target = self._extract_upload_target()
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            if root == local_dir:
                rel_dir = ""
            else:
                rel_dir = os.path.relpath(root, local_dir)
            for filename in filenames:
                local_file = os.path.join(root, filename)
                if artifact_path and rel_dir:
                    file_artifact_path = posixpath.join(artifact_path, rel_dir.replace(os.sep, "/"))
                elif artifact_path:
                    file_artifact_path = artifact_path
                elif rel_dir:
                    file_artifact_path = rel_dir.replace(os.sep, "/")
                else:
                    file_artifact_path = None
                self._upload_via_presigned_url(upload_target, local_file, file_artifact_path)

    def _extract_upload_target(self) -> _UploadTarget:
        """Extract the owning resource type and ID from the artifact URI.

        Run pattern: s3://bucket/.../<run_id>/artifacts/...
        Logged-model pattern: s3://bucket/.../models/<model_id>/artifacts/...
        Uses reverse scan because 'artifacts' may appear in the URI prefix path
        (e.g., s3://bucket/ml-artifacts/<exp_id>/<run_id>/artifacts).
        """
        try:
            parts = urlparse(self.artifact_uri).path.strip("/").split("/")
        except Exception as exc:
            raise ValueError(f"Failed to parse upload target from artifact URI: {self.artifact_uri}") from exc

        for i in range(len(parts) - 1, 0, -1):
            if parts[i] != "artifacts":
                continue

            resource_id = parts[i - 1]
            if not resource_id:
                continue
            if i >= 2 and parts[i - 2] == "models" and resource_id.startswith("m-"):
                return _UploadTarget(_UploadResourceType.LOGGED_MODEL, resource_id)
            if i >= 2 and parts[i - 2] == "traces" and resource_id.startswith("tr-"):
                raise ValueError(
                    f"Presigned upload does not support trace artifact target '{resource_id}' from URI: "
                    f"{self.artifact_uri}"
                )
            return _UploadTarget(_UploadResourceType.RUN, resource_id)

        raise ValueError(
            "Could not extract a run or logged-model upload target from artifact URI "
            f"(expected .../<resource_id>/artifacts): {self.artifact_uri}"
        )

    def _get_tracking_host_creds(self) -> rest_utils.MlflowHostCreds:
        """Build MlflowHostCreds for the SageMaker tracking server."""
        return _get_host_creds(self.tracking_uri)

    def _build_upload_path(self, local_file: str, artifact_path: Optional[str]) -> str:
        """Construct the relative path sent to the server as the 'path' parameter.

        Must match how S3ArtifactRepository.log_artifact builds the S3 key:
        artifact_path (if any) + basename of local_file.
        """
        filename = os.path.basename(local_file)
        if artifact_path:
            return posixpath.join(artifact_path, filename)
        return filename

    def _request_presigned_url(self, upload_target: _UploadTarget, path: str, expiration: int = 900):
        """Request a presigned upload URL from the tracking server (SigV4-authenticated)."""
        host_creds = self._get_tracking_host_creds()
        return rest_utils.http_request(
            host_creds,
            _PRESIGNED_UPLOAD_ENDPOINT,
            "POST",
            json={upload_target.resource_type.value: upload_target.resource_id, "path": path, "expiration": expiration},
            raise_on_status=False,
            max_retries=0,
        )

    def _upload_via_presigned_url(
        self, upload_target: _UploadTarget, local_file: str, artifact_path: Optional[str]
    ) -> None:
        """Upload a file via a presigned URL.

        Two distinct HTTP paths:
        1. Tracking server API call (_request_presigned_url): SigV4-authenticated
           via rest_utils.http_request.
        2. S3 presigned URL PUT: NO auth — authorization is embedded in the
           presigned URL signature. Uses cloud_storage_http_request (same pattern
           as presigned_url_artifact_repo.py and optimized_s3_artifact_repo.py).

        Streams the file directly to avoid loading large artifacts into memory.
        """
        path = self._build_upload_path(local_file, artifact_path)
        response = self._request_presigned_url(upload_target, path)

        if not response.ok:
            raise Exception(f"Presigned upload URL request failed (HTTP {response.status_code})")

        response_json = response.json()
        presigned_url = response_json.get("presigned_url")
        headers = response_json.get("headers", {})

        with open(local_file, "rb") as f:
            put_response = cloud_storage_http_request(
                "put",
                presigned_url,
                data=f,
                headers=headers,
            )
            put_response.raise_for_status()

        logger.debug("Artifact uploaded via presigned URL: %s", path)
