import os
import pytest
from mlflow import MlflowClient

from utils.boto_utils import get_default_region, get_account_id


""" Default tracking server that a user can create
"""


@pytest.fixture(scope="module")
def tracking_server():
    server_arn = os.environ.get("MLFLOW_TRACKING_SERVER_URI", "")
    if not server_arn:
        server_name = os.environ.get("MLFLOW_TRACKING_SERVER_NAME", "")
        if server_name:
            region = get_default_region()
            account_id = get_account_id()
            # Reconstruct server arn from env variables
            server_arn = f"arn:aws:sagemaker:{region}:{account_id}:mlflow-tracking-server/{server_name}"
        else:
            server_arn = create_tracking_server()
        os.environ["MLFLOW_TRACKING_SERVER_URI"] = server_arn
    return server_arn


""" Mlflow Client Fixture
"""


@pytest.fixture
def mlflow_client() -> MlflowClient:
    return MlflowClient()


def create_tracking_server() -> str:
    # TODO: Implement
    return "not implemented"
