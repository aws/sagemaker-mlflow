import unittest
from unittest import mock, TestCase

from sagemaker_mlflow.mlflow_sagemaker_store import MlflowSageMakerStore, get_host_creds


TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_URL = "https://test-site.com"


class MlflowSageMakerStoreTest(TestCase):

    def test_get_host_creds_happy(self):
        arn = TEST_VALID_ARN
        url = "url"

        with mock.patch(
            "sagemaker_mlflow.mlflow_sagemaker_store.host_metadata_provider.construct_tracking_server_url",
            return_value=url,
        ):
            result = get_host_creds(arn)
            assert result.host == url
            assert result.auth == "arn"

    def test_MlflowSageMakerStore_Store(self):
        test_instance = MlflowSageMakerStore(TEST_VALID_ARN, "")
        assert test_instance is not None


if __name__ == "__main__":
    unittest.main()
