import unittest
from unittest import mock, TestCase

from sagemaker_mlflow.auth_provider import AuthProvider
from sagemaker_mlflow.exceptions import ResourceTypeUnsupportedException


class AuthProviderTest(TestCase):
    def test_auth_provider_returns_correct_name(self):
        auth_provider = AuthProvider()
        auth_provider_name = auth_provider.get_name()
        self.assertEqual(auth_provider_name, "arn")

    def test_auth_provider_returns_correct_sigv4_tracking_server(self):
        region = "us-east-2"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "mlflow-tracking-server"
        metadata_provider.region = region
        metadata_provider.maybe_assume_role_arn = None

        auth_provider = AuthProvider()
        auth_provider.host_metadata_provider = metadata_provider
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-east-2")
        self.assertEqual(result.sigv4._service_name, "sagemaker-mlflow")

    def test_auth_provider_returns_correct_sigv4_mlflow_app(self):
        region = "us-east-2"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "mlflow-app"
        metadata_provider.region = region
        metadata_provider.maybe_assume_role_arn = None

        auth_provider = AuthProvider()
        auth_provider.host_metadata_provider = metadata_provider
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-east-2")
        self.assertEqual(result.sigv4._service_name, "sagemaker")

    def test_auth_provider_returns_correct_sigv4_unknown(self):
        region = "us-east-2"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "wee"
        metadata_provider.region = region

        auth_provider = AuthProvider()
        auth_provider.host_metadata_provider = metadata_provider
        self.assertRaises(ResourceTypeUnsupportedException, auth_provider.get_auth)

    @mock.patch.dict("os.environ", {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role"})
    @mock.patch("sagemaker_mlflow.auth.boto3.Session")
    def test_auth_provider_with_assume_role(self, mock_session):
        region = "us-east-2"

        # Mock STS client and assume role response
        mock_sts_client = mock.Mock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "test-access-key",
                "SecretAccessKey": "test-secret-key",
                "SessionToken": "test-session-token",
            }
        }

        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "mlflow-tracking-server"
        metadata_provider.region = region
        metadata_provider.maybe_assume_role_arn = None

        auth_provider = AuthProvider()
        auth_provider.host_metadata_provider = metadata_provider
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-east-2")
        self.assertEqual(result.sigv4._service_name, "sagemaker-mlflow")
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/test-role", RoleSessionName="AuthBotoSagemakerMlFlow"
        )


if __name__ == "__main__":
    unittest.main()
