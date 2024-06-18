import mlflow
import pytest

from mlflow.models import infer_signature

from utils.random_utils import generate_uuid
from utils.sklearn_utils import train_iris_logistic_regression_model

""" This test makes sure that registering models works.
"""
class TestModelRegistry:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        # TODO: Verify that tracking server is created
        mlflow.set_tracking_uri(tracking_server)

    def test_model_registry(self, setup, mlflow_client):
        # Start an MLflow run
        registered_model_name = generate_uuid(20)
        X_train, lr = train_iris_logistic_regression_model()
        with mlflow.start_run():
            signature = infer_signature(X_train, lr.predict(X_train))

            # Log/Register the model
            mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="iris_model",
                signature=signature,
                input_example=X_train,
                registered_model_name=registered_model_name,
            )
            registered_model = mlflow_client.get_registered_model(registered_model_name)
            assert registered_model.name == registered_model_name
            mlflow_client.delete_registered_model(registered_model_name)
