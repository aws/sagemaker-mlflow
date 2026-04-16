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
import tempfile
import unittest
from unittest import mock, TestCase

from sagemaker_mlflow.s3_presigned_artifact_repo import (
    S3PresignedArtifactRepository,
    _SAGEMAKER_PRESIGNED_URL_UPLOAD_ENV_VAR,
)

TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test-server"
TEST_ARTIFACT_URI = "s3://test-bucket/123/abc456/artifacts"
TEST_TRACKING_URL = "https://us-west-2.experiments.sagemaker.aws"
TEST_PRESIGNED_URL = "https://test-bucket.s3.amazonaws.com/presigned-put?X-Amz-Signature=abc"

MODULE = "sagemaker_mlflow.s3_presigned_artifact_repo"


def _mock_response(status_code=200, json_data=None):
    """Build a mock HTTP response."""
    response = mock.Mock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = json_data or {
        "presigned_url": TEST_PRESIGNED_URL,
        "headers": {"Content-Type": "application/octet-stream"},
    }
    response.raise_for_status = mock.Mock()
    if status_code >= 400:
        from requests.exceptions import HTTPError

        response.raise_for_status.side_effect = HTTPError(
            f"HTTP {status_code}", response=response
        )
    return response


def _create_repo(artifact_uri=TEST_ARTIFACT_URI, tracking_uri=TEST_VALID_ARN, env_enabled=True):
    """Create an S3PresignedArtifactRepository with mocked parent init."""
    env = {_SAGEMAKER_PRESIGNED_URL_UPLOAD_ENV_VAR: "true" if env_enabled else ""}
    with mock.patch.dict(os.environ, env, clear=False):
        with mock.patch(f"{MODULE}.S3ArtifactRepository.__init__", return_value=None):
            repo = S3PresignedArtifactRepository(artifact_uri, tracking_uri=tracking_uri)
            repo.artifact_uri = artifact_uri
            repo.tracking_uri = tracking_uri
            return repo


class TestFeatureFlagAndInit(TestCase):
    """Tests #1, #16, #17: feature flag, constructor, tracking_uri=None."""

    def test_feature_disabled_uses_parent(self):
        """#1: No env var → super().log_artifact() called, no server call."""
        repo = _create_repo(env_enabled=False)

        with mock.patch.object(
            S3PresignedArtifactRepository.__bases__[0], "log_artifact"
        ) as mock_parent:
            repo.log_artifact("/tmp/model.pkl")
            mock_parent.assert_called_once_with("/tmp/model.pkl", None)

    def test_constructor_receives_tracking_uri(self):
        """#16: self.tracking_uri stored via parent ArtifactRepository.__init__."""
        repo = _create_repo()
        self.assertEqual(repo.tracking_uri, TEST_VALID_ARN)

    def test_tracking_uri_none_skips_presigned(self):
        """#17: tracking_uri=None → _should_use_presigned() returns False."""
        repo = _create_repo(tracking_uri=None)
        repo.tracking_uri = None
        self.assertFalse(repo._should_use_presigned())

        with mock.patch.object(
            S3PresignedArtifactRepository.__bases__[0], "log_artifact"
        ) as mock_parent:
            repo.log_artifact("/tmp/model.pkl")
            mock_parent.assert_called_once()


class TestRunIdExtraction(TestCase):
    """Tests #11-14: run_id extraction from artifact URI."""

    def test_run_id_extraction_standard(self):
        """#11: s3://bucket/123/abc456/artifacts → 'abc456'."""
        repo = _create_repo(artifact_uri="s3://bucket/123/abc456/artifacts")
        self.assertEqual(repo._extract_run_id(), "abc456")

    def test_run_id_extraction_with_subpath(self):
        """#12: URI with subpath still extracts correctly."""
        repo = _create_repo(artifact_uri="s3://bucket/123/abc456/artifacts/models/v1")
        self.assertEqual(repo._extract_run_id(), "abc456")

    def test_run_id_extraction_artifacts_in_prefix(self):
        """#13: Reverse scan: s3://bucket/data/artifacts/123/abc456/artifacts → 'abc456'."""
        repo = _create_repo(
            artifact_uri="s3://bucket/data/artifacts/123/abc456/artifacts"
        )
        self.assertEqual(repo._extract_run_id(), "abc456")

    def test_run_id_extraction_failure(self):
        """#14: Malformed URI → returns None → falls back to direct S3."""
        repo = _create_repo(artifact_uri="s3://bucket/no-artifacts-segment")
        self.assertIsNone(repo._extract_run_id())
        self.assertFalse(repo._should_use_presigned())


class TestBuildUploadPath(TestCase):
    """Tests #20, #21: _build_upload_path construction."""

    def test_build_upload_path_no_artifact_path(self):
        """#20: _build_upload_path('/tmp/model.pkl', None) → 'model.pkl'."""
        repo = _create_repo()
        self.assertEqual(repo._build_upload_path("/tmp/model.pkl", None), "model.pkl")

    def test_build_upload_path_with_artifact_path(self):
        """#21: _build_upload_path('/tmp/model.pkl', 'models/v1') → 'models/v1/model.pkl'."""
        repo = _create_repo()
        self.assertEqual(
            repo._build_upload_path("/tmp/model.pkl", "models/v1"), "models/v1/model.pkl"
        )


class TestPresignedUploadHappyPath(TestCase):
    """Tests #2, #3, #15, #18: successful presigned uploads."""

    def setUp(self):
        self.repo = _create_repo()

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_presigned_upload_happy_path(self, mock_provider_cls, mock_http, mock_cloud):
        """#2: Mock server returns URL → file PUT uploaded via cloud_storage_http_request."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response()
        mock_cloud.return_value = _mock_response()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"model_data")
            tmp_path = f.name

        try:
            self.repo.log_artifact(tmp_path)

            mock_http.assert_called_once()
            call_kwargs = mock_http.call_args
            self.assertEqual(call_kwargs[1]["json"]["run_id"], "abc456")

            mock_cloud.assert_called_once()
            cloud_call = mock_cloud.call_args
            self.assertEqual(cloud_call[0][0], "put")
            self.assertEqual(cloud_call[0][1], TEST_PRESIGNED_URL)
        finally:
            os.unlink(tmp_path)

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_presigned_upload_streams_file(self, mock_provider_cls, mock_http, mock_cloud):
        """Verify file is streamed (file handle passed) rather than read into memory."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response()
        mock_cloud.return_value = _mock_response()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"model_data")
            tmp_path = f.name

        try:
            self.repo.log_artifact(tmp_path)

            cloud_call = mock_cloud.call_args
            # data should be a file object, not bytes
            data_arg = cloud_call[1]["data"]
            self.assertTrue(hasattr(data_arg, "read"), "data should be a file-like object, not bytes")
        finally:
            os.unlink(tmp_path)

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_presigned_upload_with_artifact_path(self, mock_provider_cls, mock_http, mock_cloud):
        """#3: Server receives path='models/model.pkl' when artifact_path='models'."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response()
        mock_cloud.return_value = _mock_response()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"model_data")
            tmp_path = f.name

        try:
            self.repo.log_artifact(tmp_path, "models")

            call_kwargs = mock_http.call_args
            path_sent = call_kwargs[1]["json"]["path"]
            expected = "models/" + os.path.basename(tmp_path)
            self.assertEqual(path_sent, expected)
        finally:
            os.unlink(tmp_path)

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_presigned_upload_includes_headers(self, mock_provider_cls, mock_http, mock_cloud):
        """#15: Server response headers forwarded in PUT request."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL

        custom_headers = {
            "Content-Type": "application/octet-stream",
            "x-amz-server-side-encryption": "aws:kms",
        }
        mock_http.return_value = _mock_response(
            json_data={"presigned_url": TEST_PRESIGNED_URL, "headers": custom_headers}
        )
        mock_cloud.return_value = _mock_response()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            self.repo.log_artifact(tmp_path)

            cloud_call = mock_cloud.call_args
            self.assertEqual(cloud_call[1]["headers"], custom_headers)
        finally:
            os.unlink(tmp_path)


class TestPermanentFailures(TestCase):
    """Tests #5, #6, #9: permanent failure behavior (404, 501)."""

    def setUp(self):
        self.repo = _create_repo()

    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_server_404_raises(self, mock_provider_cls, mock_http):
        """#5: Old server returns 404 → exception propagates."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response(status_code=404)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            with self.assertRaises(Exception):
                self.repo.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)

    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_server_501_raises(self, mock_provider_cls, mock_http):
        """#6: Non-S3 backend → exception propagates."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response(status_code=501)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            with self.assertRaises(Exception):
                self.repo.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestTransientFailures(TestCase):
    """Tests #7, #8: transient failure handling (503, PUT failure)."""

    def setUp(self):
        self.repo = _create_repo()

    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_server_503_raises(self, mock_provider_cls, mock_http):
        """#7: 503 → exception propagates."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response(status_code=503)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            with self.assertRaises(Exception):
                self.repo.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_put_failure_raises(self, mock_provider_cls, mock_http, mock_cloud):
        """#8: PUT fails → exception propagates."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response()
        mock_cloud.side_effect = Exception("Connection reset")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            f.write(b"data")
            tmp_path = f.name

        try:
            with self.assertRaises(Exception):
                self.repo.log_artifact(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestDirectoryUploads(TestCase):
    """Tests #4, #19: log_artifacts directory handling."""

    def setUp(self):
        self.repo = _create_repo()

    def test_log_artifacts_disabled_calls_parent(self):
        """When presigned is disabled, log_artifacts delegates to parent directly."""
        repo = _create_repo(env_enabled=False)

        with mock.patch.object(
            S3PresignedArtifactRepository.__bases__[0], "log_artifacts"
        ) as mock_parent:
            repo.log_artifacts("/tmp/some_dir", "output")
            mock_parent.assert_called_once_with("/tmp/some_dir", "output")

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_presigned_upload_directory(self, mock_provider_cls, mock_http, mock_cloud):
        """#4: log_artifacts() calls self.log_artifact() per file, each via presigned URL."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response()
        mock_cloud.return_value = _mock_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in ["file1.txt", "file2.txt", "file3.txt"]:
                with open(os.path.join(tmp_dir, name), "w") as f:
                    f.write("content")

            self.repo.log_artifacts(tmp_dir, "output")

        self.assertEqual(mock_http.call_count, 3)
        self.assertEqual(mock_cloud.call_count, 3)

        paths_sent = sorted(
            call[1]["json"]["path"] for call in mock_http.call_args_list
        )
        self.assertEqual(
            paths_sent,
            ["output/file1.txt", "output/file2.txt", "output/file3.txt"],
        )

    @mock.patch(f"{MODULE}.cloud_storage_http_request")
    @mock.patch(f"{MODULE}.rest_utils.http_request")
    @mock.patch(f"{MODULE}.SageMakerMLflowHostMetadataProvider")
    def test_log_artifacts_failure_propagates(
        self, mock_provider_cls, mock_http, mock_cloud
    ):
        """#19: If presigned upload fails for a file, exception propagates."""
        mock_provider = mock_provider_cls.return_value
        mock_provider.construct_tracking_server_url.return_value = TEST_TRACKING_URL
        mock_http.return_value = _mock_response(status_code=503)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in ["a.txt", "b.txt"]:
                with open(os.path.join(tmp_dir, name), "w") as f:
                    f.write("content")

            with self.assertRaises(Exception):
                self.repo.log_artifacts(tmp_dir)


if __name__ == "__main__":
    unittest.main()
