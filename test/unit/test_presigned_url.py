import unittest
from unittest import mock, TestCase
import os

from sagemaker_mlflow.presigned_url import get_presigned_url
from moto import mock_aws

TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_URL = "https://test-site.com"


@mock.patch.dict(
    os.environ,
    {"MLFLOW_TRACKING_URI": TEST_VALID_ARN},
)
class PresignedUrlTests(TestCase):

    @mock.patch("boto3.Session")
    def test_presigned_url(self, mock_session):
        mock_client = mock.Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_response = {"AuthorizedUrl": TEST_VALID_URL}
        mock_client.create_presigned_mlflow_tracking_server_url.return_value = mock_response
        function_response = get_presigned_url()
        assert function_response == TEST_VALID_URL

    @mock.patch("boto3.Session")
    def test_presigned_url_with_fields(self, mock_session):
        mock_client = mock.Mock()
        mock_session.return_value.client.return_value = mock_client
        mock_response = {"AuthorizedUrl": TEST_VALID_URL}

        create_presigned_api_request = {
            "TrackingServerName": "xw",
            "ExpiresInSeconds": 200,
            "SessionExpirationDurationInSeconds": 1800,
        }

        mock_client.create_presigned_mlflow_tracking_server_url.return_value = mock_response
        function_response = get_presigned_url(200, 1800)

        mock_client.create_presigned_mlflow_tracking_server_url.assert_called_with(**create_presigned_api_request)

        assert function_response == TEST_VALID_URL


if __name__ == "__main__":
    unittest.main()
