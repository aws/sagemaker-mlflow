import unittest
from unittest import mock, TestCase
from sagemaker_mlflow.mlflow_sagemaker_request_header_provider import MlflowSageMakerRequestHeaderProvider
import os

class MlflowSageMakerRequestHeaderProviderTest(TestCase):

    def test_in_context(self):
        provider = MlflowSageMakerRequestHeaderProvider()
        in_context = provider.in_context()
        assert in_context

    @mock.patch.dict(
        os.environ,
        {
            "MLFLOW_TRACKING_URI": "mw"
        },
    )
    def test_request_header(self):
        provider = MlflowSageMakerRequestHeaderProvider()
        header = provider.request_headers()
        assert header.get("x-mlflow-sm-tracking-server-arn") == "mw"


if __name__ == "__main__":
    unittest.main()
