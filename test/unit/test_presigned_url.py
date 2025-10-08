import unittest
from unittest import mock, TestCase
import os

from sagemaker_mlflow.presigned_url import get_presigned_url
from moto import mock_aws

TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_ROLE_ARN = "arn:aws:iam::0123456789:role/role-name-with-path"
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

    @mock.patch("boto3.Session")
    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": TEST_VALID_ROLE_ARN},
    )
    def test_presigned_url_with_assume_role(self, mock_session):
        mock_session_instance = mock_session.return_value
        mock_sts_client = mock.Mock()
        mock_sagemaker_client = mock.Mock()

        # Mock STS assume role response
        mock_assume_role_response = {
            "Credentials": {
                "AccessKeyId": "assumed_access_key",
                "SecretAccessKey": "assumed_secret_key",
                "SessionToken": "assumed_session_token"
            }
        }
        mock_sts_client.assume_role.return_value = mock_assume_role_response

        # Mock session.client to return appropriate clients
        def mock_client(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "sagemaker":
                return mock_sagemaker_client
            return None

        mock_session_instance.client = mock_client

        # Mock SageMaker response
        mock_response = {"AuthorizedUrl": TEST_VALID_URL}
        mock_sagemaker_client.create_presigned_mlflow_tracking_server_url.return_value = mock_response

        function_response = get_presigned_url()

        # Verify STS assume role was called
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn=TEST_VALID_ROLE_ARN,
            RoleSessionName="AuthBotoSagemakerMlFlow"
        )

        # Verify new session was created with assumed role credentials
        mock_session.assert_any_call(
            aws_access_key_id="assumed_access_key",
            aws_secret_access_key="assumed_secret_key",
            aws_session_token="assumed_session_token"
        )

        assert function_response == TEST_VALID_URL


if __name__ == "__main__":
    unittest.main()
