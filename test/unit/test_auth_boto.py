import unittest
from unittest.mock import patch, Mock
import boto3
from botocore.awsrequest import AWSRequest
from requests import PreparedRequest

from sagemaker_mlflow.auth import AuthBoto, EMPTY_SHA256_HASH


class TestAuthBoto(unittest.TestCase):

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

    def test_call(self):
        # Arrange
        region = "us-west-2"
        auth_boto = AuthBoto(region)

        mock_sigv4 = Mock()
        auth_boto.sigv4 = mock_sigv4
        auth_boto.creds = Mock()

        url = "https://example.com/path"
        method = "GET"
        headers = {"Connection": "keep-alive", "x-sagemaker": "woof"}
        body = None
        prepared_request = PreparedRequest()
        prepared_request.prepare(url=url, method=method, headers=headers, data=body)

        expected_headers = {
            "X-Amz-Content-SHA256": EMPTY_SHA256_HASH,
            "Connection": "keep-alive",
            "x-sagemaker": "woof",
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
        expected_hash = (
            "4443c6a8412e6c11f324c870a8366d6ede75e7f9ed12f00c36b88d479df371d6"
        )

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


if __name__ == "__main__":
    unittest.main()
