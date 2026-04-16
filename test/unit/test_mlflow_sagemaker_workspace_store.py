import unittest
from unittest import mock, TestCase
from sagemaker_mlflow.mlflow_sagemaker_workspace_store import (
    MlflowSageMakerWorkspaceStore,
    get_host_creds,
)

TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_APP_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-app/app-XXXXXXXXXXXX"


class MlflowSageMakerWorkspaceStoreTest(TestCase):

    def test_get_host_creds_happy(self):
        url = "url"

        with mock.patch(
            "sagemaker_mlflow.mlflow_sagemaker_workspace_store.host_metadata_provider.construct_tracking_server_url",
            return_value=url,
        ):
            result = get_host_creds(TEST_VALID_ARN)
            assert result.host == url
            assert result.auth == "arn"

    def test_get_host_creds_mlflow_app(self):
        url = "url"

        with mock.patch(
            "sagemaker_mlflow.mlflow_sagemaker_workspace_store.host_metadata_provider.construct_tracking_server_url",
            return_value=url,
        ):
            result = get_host_creds(TEST_VALID_APP_ARN)
            assert result.host == url
            assert result.auth == "arn"

    def test_store_instantiation(self):
        with mock.patch(
            "sagemaker_mlflow.mlflow_sagemaker_workspace_store.host_metadata_provider.construct_tracking_server_url",
            return_value="https://test-site.com",
        ):
            store = MlflowSageMakerWorkspaceStore(TEST_VALID_ARN)
            assert store is not None
            assert callable(store.get_host_creds)

    def test_store_host_creds_callable(self):
        url = "https://test-site.com"

        with mock.patch(
            "sagemaker_mlflow.mlflow_sagemaker_workspace_store.host_metadata_provider.construct_tracking_server_url",
            return_value=url,
        ):
            store = MlflowSageMakerWorkspaceStore(TEST_VALID_ARN)
            creds = store.get_host_creds()
            assert creds.host == url
            assert creds.auth == "arn"


if __name__ == "__main__":
    unittest.main()
