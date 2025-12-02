import unittest
from unittest import mock, TestCase
from sagemaker_mlflow.mlflow_sagemaker_request_header_provider import MlflowSageMakerRequestHeaderProvider
from sagemaker_mlflow.exceptions import ResourceTypeUnsupportedException


class MlflowSageMakerRequestHeaderProviderTest(TestCase):

    def test_in_context(self):
        provider = MlflowSageMakerRequestHeaderProvider()
        in_context = provider.in_context()
        assert in_context

    def test_request_header_tracking_server(self):
        arn = "arn"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "mlflow-tracking-server"
        metadata_provider.arn = arn

        provider = MlflowSageMakerRequestHeaderProvider()
        provider.host_metadata_provider = metadata_provider
        header = provider.request_headers()
        assert header.get("x-mlflow-sm-tracking-server-arn") == arn

    def test_request_header_mlflow_app(self):
        arn = "arn"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "mlflow-app"
        metadata_provider.arn = arn

        provider = MlflowSageMakerRequestHeaderProvider()
        provider.host_metadata_provider = metadata_provider
        header = provider.request_headers()
        assert header.get("x-sm-mlflow-app-arn") == arn

    def test_request_header_unknown(self):
        arn = "arn"
        metadata_provider = mock.Mock()
        metadata_provider.resource_type = "wee"
        metadata_provider.arn = arn

        provider = MlflowSageMakerRequestHeaderProvider()
        provider.host_metadata_provider = metadata_provider

        self.assertRaises(ResourceTypeUnsupportedException, provider.request_headers)


if __name__ == "__main__":
    unittest.main()
