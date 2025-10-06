import unittest
from unittest import mock, TestCase

from sagemaker_mlflow.auth_provider import AuthProvider
import os
from moto import mock_sts

TEST_VALID_ARN_WITH_ROLE = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw#arn:aws:iam::0123456789:role/role-name-with-path"


class AuthProviderTest(TestCase):
    def test_auth_provider_returns_correct_name(self):
        auth_provider = AuthProvider()
        auth_provider_name = auth_provider.get_name()
        self.assertEqual(auth_provider_name, "arn")

    @mock.patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "default_ak",
            "AWS_SECRET_ACCESS_KEY": "default_sk",
            "AWS_DEFAULT_REGION": "us-east-2",
            "AWS_SESSION_TOKEN": "",
            "MLFLOW_TRACKING_URI": "arn:aws:sagemaker:us-east-2:000000000000:mlflow-tracking-server/mw",
        },
    )
    def test_auth_provider_returns_correct_sigv4(self):
        auth_provider = AuthProvider()
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-east-2")
        self.assertEqual(result._assume_role_arn, None)

    @mock.patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "default_ak",
            "AWS_SECRET_ACCESS_KEY": "default_sk",
            "AWS_DEFAULT_REGION": "us-east-2",
            "AWS_SESSION_TOKEN": "",
            "MLFLOW_TRACKING_URI": TEST_VALID_ARN_WITH_ROLE,
        },
    )
    @mock_sts
    def test_auth_provider_returns_correct_sigv4(self):
        auth_provider = AuthProvider()
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-west-2")
        self.assertEqual(result._assume_role_arn, "arn:aws:iam::0123456789:role/role-name-with-path")

    @mock.patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "default_ak",
            "AWS_SECRET_ACCESS_KEY": "default_sk",
            "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_SESSION_TOKEN": "dcs",
            "MLFLOW_TRACKING_URI": "arn:aws:sagemaker:us-east-2:000000000001:mlflow-tracking-server/mw",
        },
    )
    def test_auth_provider_returns_correct_sigv4_session_different_region(self):
        auth_provider = AuthProvider()
        result = auth_provider.get_auth()

        self.assertEqual(result.region, "us-east-2")


if __name__ == "__main__":
    unittest.main()
