import unittest
import os
from unittest.mock import ANY, call, patch, Mock
from botocore.awsrequest import AWSRequest
from requests import PreparedRequest

from sagemaker_mlflow.auth import AuthBoto, EMPTY_SHA256_HASH, DEFAULT_CREDENTIAL_TTL_SECONDS


class TestAuthBoto(unittest.TestCase):

    def setUp(self):
        # Clear the credential cache before each test
        AuthBoto._credential_cache.clear()

    @patch("boto3.Session")
    def test_init(self, mock_session):
        # Arrange
        mock_session_instance = mock_session.return_value
        mock_get_credentials = mock_session_instance.get_credentials
        mock_credentials = Mock()
        mock_get_credentials.return_value = mock_credentials
        region = "us-west-2"

        # Act
        auth_boto = AuthBoto(region)

        # Assert
        self.assertEqual(auth_boto.region, region)
        self.assertEqual(auth_boto.creds, mock_credentials)
        mock_session.assert_called_once()
        mock_get_credentials.assert_called_once()

    @patch("boto3.Session")
    def test_init_with_assume_role_arn(self, mock_session):
        # Arrange
        mock_session_instance = mock_session.return_value
        mock_get_credentials = mock_session_instance.get_credentials
        mock_credentials = Mock()
        mock_get_credentials.return_value = mock_credentials
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/role-name-with-path"

        # Act
        auth_boto = AuthBoto(region, assume_role_arn)

        # Assert
        calls = [
            call(),
            call()
            .client()
            .assume_role(
                RoleArn="arn:aws:iam::0123456789:role/role-name-with-path", RoleSessionName="AuthBotoSagemakerMlFlow"
            ),
            call(aws_access_key_id=ANY, aws_secret_access_key=ANY, aws_session_token=ANY),
        ]
        self.assertEqual(auth_boto.region, region)
        self.assertEqual(auth_boto.creds, mock_credentials)
        mock_session.assert_called()
        mock_session.assert_has_calls(calls, any_order=True)
        mock_get_credentials.assert_called_once()

    def test_call(self):
        # Arrange
        region = "us-west-2"
        auth_boto = AuthBoto(region)

        mock_sigv4 = Mock()
        auth_boto.sigv4 = mock_sigv4
        auth_boto.creds = Mock()

        url = "https://example.com/path"
        method = "GET"
        header_value = "test-value"
        headers = {"Connection": "keep-alive", "x-sagemaker": header_value}
        body = None
        prepared_request = PreparedRequest()
        prepared_request.prepare(url=url, method=method, headers=headers, data=body)

        expected_headers = {
            "X-Amz-Content-SHA256": EMPTY_SHA256_HASH,
            "Connection": "keep-alive",
            "x-sagemaker": header_value,
        }
        expected_aws_request = AWSRequest(
            method=method,
            url=url.replace("+", "%20"),
            headers=expected_headers,
            data=body,
        )

        # Act
        result = auth_boto(prepared_request)

        # Assert
        for header in result.headers:
            self.assertTrue(header in expected_headers)

        self.assertEqual(result.body, expected_aws_request.data)
        self.assertEqual(result.method, method)
        self.assertEqual(result.url, url.replace("+", "%20"))

    def test_get_request_body_header(self):
        # Arrange
        region = "us-west-2"
        auth_boto = AuthBoto(region)
        request_body = b"test_body"
        expected_hash = "4443c6a8412e6c11f324c870a8366d6ede75e7f9ed12f00c36b88d479df371d6"

        # Act
        result = auth_boto.get_request_body_header(request_body)

        # Assert
        self.assertEqual(result, expected_hash)

    def test_get_request_body_header_empty(self):
        # Arrange
        region = "us-west-2"
        auth_boto = AuthBoto(region)
        request_body = b""

        # Act
        result = auth_boto.get_request_body_header(request_body)

        # Assert
        self.assertEqual(result, EMPTY_SHA256_HASH)

    @patch("boto3.Session")
    def test_credential_caching_first_call(self, mock_session):
        # Test that first call to assume role caches credentials
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}

        # Mock session credentials
        mock_session_instance.get_credentials.return_value = Mock()

        # Act
        auth_boto = AuthBoto(region, assume_role_arn)

        # Assert - STS should be called once
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn=assume_role_arn,
            RoleSessionName="AuthBotoSagemakerMlFlow"
        )

    @patch("boto3.Session")
    def test_credential_caching_second_call_uses_cache(self, mock_session):
        # Test that second call uses cached credentials without calling STS
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response for first call
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}
        mock_session_instance.get_credentials.return_value = Mock()

        # First call - should hit STS
        auth_boto_1 = AuthBoto(region, assume_role_arn)

        # Reset mock to verify second call doesn't hit STS
        mock_sts_client.reset_mock()

        # Second call - should use cache
        auth_boto_2 = AuthBoto(region, assume_role_arn)

        # Assert - STS should not be called on second initialization
        mock_sts_client.assume_role.assert_not_called()

    @patch.dict(os.environ, {"SAGEMAKER_MLFLOW_ASSUME_ROLE_TTL_SECONDS": "1800"})
    @patch("boto3.Session")
    def test_custom_ttl_from_environment(self, mock_session):
        # Test that custom TTL is read from environment variable
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}
        mock_session_instance.get_credentials.return_value = Mock()

        # Act
        with patch.object(AuthBoto._credential_cache, 'set_credentials') as mock_set_credentials:
            auth_boto = AuthBoto(region, assume_role_arn)

            # Assert - credentials should be cached with custom TTL
            mock_set_credentials.assert_called_once_with(assume_role_arn, mock_credentials, 1800)

    @patch("boto3.Session")
    def test_default_ttl_used_when_no_environment_variable(self, mock_session):
        # Test that default TTL is used when no environment variable is set
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}
        mock_session_instance.get_credentials.return_value = Mock()

        # Act
        with patch.object(AuthBoto._credential_cache, 'set_credentials') as mock_set_credentials:
            auth_boto = AuthBoto(region, assume_role_arn)

            # Assert - credentials should be cached with default TTL
            mock_set_credentials.assert_called_once_with(assume_role_arn, mock_credentials, DEFAULT_CREDENTIAL_TTL_SECONDS)

    @patch.dict(os.environ, {"SAGEMAKER_MLFLOW_ASSUME_ROLE_TTL_SECONDS": "7200"})  # > 3600
    @patch("boto3.Session")
    def test_ttl_validation_upper_bound(self, mock_session):
        # Test that TTL is capped at 3600 seconds (1 hour)
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}
        mock_session_instance.get_credentials.return_value = Mock()

        # Act
        with patch.object(AuthBoto._credential_cache, 'set_credentials') as mock_set_credentials:
            auth_boto = AuthBoto(region, assume_role_arn)

            # Assert - TTL should be capped at 3600
            mock_set_credentials.assert_called_once_with(assume_role_arn, mock_credentials, 3600)

    @patch.dict(os.environ, {"SAGEMAKER_MLFLOW_ASSUME_ROLE_TTL_SECONDS": "60"})  # < 300
    @patch("boto3.Session")
    def test_ttl_validation_lower_bound(self, mock_session):
        # Test that TTL is set to minimum of 300 seconds (5 minutes)
        region = "us-west-2"
        assume_role_arn = "arn:aws:iam::0123456789:role/test-role"

        # Mock STS response
        mock_session_instance = mock_session.return_value
        mock_sts_client = Mock()
        mock_session_instance.client.return_value = mock_sts_client

        mock_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_sts_client.assume_role.return_value = {"Credentials": mock_credentials}
        mock_session_instance.get_credentials.return_value = Mock()

        # Act
        with patch.object(AuthBoto._credential_cache, 'set_credentials') as mock_set_credentials:
            auth_boto = AuthBoto(region, assume_role_arn)

            # Assert - TTL should be set to minimum of 300
            mock_set_credentials.assert_called_once_with(assume_role_arn, mock_credentials, 300)


if __name__ == "__main__":
    unittest.main()
