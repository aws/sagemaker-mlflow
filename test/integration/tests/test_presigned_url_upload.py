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

"""Integration tests for presigned URL artifact uploads.

These tests require:
  - A SageMaker MLflow tracking server (set MLFLOW_TRACKING_SERVER_URI or
    MLFLOW_TRACKING_SERVER_NAME + REGION env vars)
  - The server must have the presigned-upload-url endpoint (mlflow/mlflow#21039)
  - AWS credentials with permission to call the tracking server and read S3

Tests are skipped when the server does not support the presigned upload
endpoint (HTTP 404), so they are safe to run against older servers.
"""

import logging
import os
import tempfile

import mlflow
import pytest

from utils.boto_utils import get_file_data_from_s3
from utils.random_utils import generate_uuid

logger = logging.getLogger(__name__)

_PRESIGNED_ENV_VAR = "SAGEMAKER_PRESIGNED_URL_UPLOAD"


def _presigned_upload_supported(tracking_server_arn):
    """Probe whether the tracking server supports the presigned upload endpoint."""
    from mlflow.utils import rest_utils
    from sagemaker_mlflow.mlflow_sagemaker_helpers import SageMakerMLflowHostMetadataProvider

    provider = SageMakerMLflowHostMetadataProvider()
    provider.set_arn(tracking_server_arn)
    host_creds = rest_utils.MlflowHostCreds(
        host=provider.construct_tracking_server_url(),
        auth="arn",
    )
    try:
        response = rest_utils.http_request(
            host_creds,
            "/api/2.0/mlflow/artifacts/presigned-upload-url",
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


def _parse_s3_uri(uri):
    """Parse s3://bucket/key into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    path = uri[len("s3://"):]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _create_temp_file(content=None, suffix=".txt"):
    """Create a temporary file with random or specified content. Returns (path, content)."""
    if content is None:
        content = generate_uuid(40) + generate_uuid(40)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    try:
        f.write(content)
    finally:
        f.close()
    return f.name, content


class PresignedEnvContext:
    """Context manager to set/unset the presigned upload environment variable."""

    def __init__(self, enabled=True):
        self._enabled = enabled
        self._original = None

    def __enter__(self):
        self._original = os.environ.get(_PRESIGNED_ENV_VAR)
        if self._enabled:
            os.environ[_PRESIGNED_ENV_VAR] = "true"
        else:
            os.environ.pop(_PRESIGNED_ENV_VAR, None)
        return self

    def __exit__(self, *args):
        if self._original is not None:
            os.environ[_PRESIGNED_ENV_VAR] = self._original
        else:
            os.environ.pop(_PRESIGNED_ENV_VAR, None)


class TestPresignedUrlUpload:
    """Integration tests for presigned URL artifact uploads.

    All tests in this class are skipped if the tracking server does not
    support the presigned-upload-url endpoint (mlflow/mlflow#21039).
    """

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        mlflow.set_tracking_uri(tracking_server)

        if not _presigned_upload_supported(tracking_server):
            pytest.skip(
                "Tracking server does not support presigned upload endpoint "
                "(requires mlflow/mlflow#21039)"
            )

    def test_presigned_upload_single_file(self, setup):
        """Upload a single file via presigned URL and verify it in S3."""
        file_path, file_contents = _create_temp_file()
        file_name = os.path.basename(file_path)

        try:
            with PresignedEnvContext():
                with mlflow.start_run() as run:
                    mlflow.log_artifact(file_path)

            artifact_uri = run.info.artifact_uri
            bucket, prefix = _parse_s3_uri(artifact_uri)
            key = f"{prefix}/{file_name}"

            data = get_file_data_from_s3(bucket, key).read().decode("utf-8")
            assert data == file_contents
        finally:
            os.remove(file_path)

    def test_presigned_upload_with_artifact_path(self, setup):
        """Upload a file to a subdirectory via presigned URL and verify in S3."""
        file_path, file_contents = _create_temp_file()
        file_name = os.path.basename(file_path)
        artifact_subdir = "models/v1"

        try:
            with PresignedEnvContext():
                with mlflow.start_run() as run:
                    mlflow.log_artifact(file_path, artifact_subdir)

            artifact_uri = run.info.artifact_uri
            bucket, prefix = _parse_s3_uri(artifact_uri)
            key = f"{prefix}/{artifact_subdir}/{file_name}"

            data = get_file_data_from_s3(bucket, key).read().decode("utf-8")
            assert data == file_contents
        finally:
            os.remove(file_path)

    def test_presigned_upload_directory(self, setup):
        """Upload a directory of files via presigned URL and verify all in S3."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            files = {}
            for name in ["data.csv", "config.json", "readme.txt"]:
                content = generate_uuid(30)
                with open(os.path.join(tmp_dir, name), "w") as f:
                    f.write(content)
                files[name] = content

            with PresignedEnvContext():
                with mlflow.start_run() as run:
                    mlflow.log_artifacts(tmp_dir, "output")

            artifact_uri = run.info.artifact_uri
            bucket, prefix = _parse_s3_uri(artifact_uri)

            for name, expected_content in files.items():
                key = f"{prefix}/output/{name}"
                data = get_file_data_from_s3(bucket, key).read().decode("utf-8")
                assert data == expected_content, f"Content mismatch for {name}"

    def test_presigned_upload_can_be_downloaded(self, setup):
        """File uploaded via presigned URL can be downloaded via mlflow client."""
        file_path, file_contents = _create_temp_file()
        file_name = os.path.basename(file_path)

        try:
            with PresignedEnvContext():
                with mlflow.start_run() as run:
                    mlflow.log_artifact(file_path)

            client = mlflow.MlflowClient()
            with tempfile.TemporaryDirectory() as download_dir:
                download_path = client.download_artifacts(
                    run.info.run_id, file_name, download_dir
                )
                with open(download_path, "r") as f:
                    downloaded = f.read()
                assert downloaded == file_contents
        finally:
            os.remove(file_path)


class TestPresignedUrlUploadDisabled:
    """Verify that artifact upload works normally when presigned is disabled."""

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        mlflow.set_tracking_uri(tracking_server)

    def test_upload_without_presigned_env_var(self, setup):
        """Without SAGEMAKER_PRESIGNED_URL_UPLOAD, uploads go through direct S3."""
        file_path, file_contents = _create_temp_file()
        file_name = os.path.basename(file_path)

        try:
            with PresignedEnvContext(enabled=False):
                with mlflow.start_run() as run:
                    mlflow.log_artifact(file_path)

            artifact_uri = run.info.artifact_uri
            bucket, prefix = _parse_s3_uri(artifact_uri)
            key = f"{prefix}/{file_name}"

            data = get_file_data_from_s3(bucket, key).read().decode("utf-8")
            assert data == file_contents
        finally:
            os.remove(file_path)
