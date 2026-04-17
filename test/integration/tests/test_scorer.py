import time

import mlflow
import pytest

from utils.mlflow_utils import mlflow_version, is_mlflow_app

pytestmark = [
    pytest.mark.skipif(mlflow_version < (3, 4), reason="Scorer APIs require MLflow >= 3.4"),
    pytest.mark.skipif(not is_mlflow_app, reason="Scorer APIs only supported on mlflow-app"),
]

from mlflow.genai.scorers import Correctness, delete_scorer


class TestScorer:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        mlflow.set_tracking_uri(tracking_server)

    def test_scorer(self, setup):
        experiment_id = mlflow.create_experiment(f"test-scorer-{int(time.time())}")

        # Register scorer
        v1 = Correctness().register(experiment_id=experiment_id)
        assert v1.name == "correctness"

        # Delete scorer
        delete_scorer(name="correctness", experiment_id=experiment_id, version="all")

        # Cleanup
        mlflow.delete_experiment(experiment_id)
