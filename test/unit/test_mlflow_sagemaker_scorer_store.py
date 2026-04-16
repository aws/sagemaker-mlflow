import unittest
from unittest import TestCase
from sagemaker_mlflow.mlflow_sagemaker_scorer_store import MlflowSageMakerScorerStore

TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_APP_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-app/app-XXXXXXXXXXXX"


class MlflowSageMakerScorerStoreTest(TestCase):

    def test_store_instantiation(self):
        store = MlflowSageMakerScorerStore(TEST_VALID_ARN)
        assert store is not None

    def test_store_instantiation_mlflow_app(self):
        store = MlflowSageMakerScorerStore(TEST_VALID_APP_ARN)
        assert store is not None

    def test_store_instantiation_none(self):
        store = MlflowSageMakerScorerStore()
        assert store is not None


if __name__ == "__main__":
    unittest.main()
