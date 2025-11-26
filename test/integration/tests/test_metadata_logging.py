import mlflow
import pytest

from utils.random_utils import generate_uuid, generate_random_float

TEST_METRIC_NAME = "test_metadata_metric"

""" Test Metadata modification, ensure that requests get properly routed
    and that SigV4 header calculation works with SageMaker Mlflow.
"""


class TestMetadataLogging:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        # TODO: Verify that tracking server is created
        mlflow.set_tracking_uri(tracking_server)

    def test_log_metric(self, setup, mlflow_client):
        random_tag = generate_uuid(32)
        tags = {"purpose": random_tag}
        metric_value = generate_random_float()

        run = mlflow_client.create_run("0", tags=tags)
        mlflow_client.log_metric(run.info.run_id, TEST_METRIC_NAME, metric_value, step=0)
        metric = list(mlflow_client.get_metric_history(run.info.run_id, TEST_METRIC_NAME))[0]

        assert "s3" in run.info.artifact_uri
        assert run.data.tags["purpose"] == tags["purpose"]
        assert metric.key == TEST_METRIC_NAME
        assert metric.value == metric_value
        assert run.info.experiment_id == "0"
