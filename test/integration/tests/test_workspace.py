import time

import mlflow
import pytest

from utils.mlflow_utils import mlflow_version, is_mlflow_app
from utils.random_utils import generate_uuid

pytestmark = [
    pytest.mark.skipif(mlflow_version < (3, 10), reason="Workspace APIs require MLflow >= 3.10"),
    pytest.mark.skipif(not is_mlflow_app, reason="Workspace APIs only supported on mlflow-app"),
]

""" Test workspace CRUD APIs against a SageMaker MLflow app.
    Workspace APIs require MLflow >= 3.10 and only work against mlflow-app.
"""


class TestWorkspaceAPIs:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        mlflow.set_tracking_uri(tracking_server)

    @pytest.fixture
    def workspace_name(self):
        """Create a unique workspace name and clean up after test."""
        name = f"test-ws-{generate_uuid(8)}"
        yield name
        # Cleanup: delete if it still exists
        try:
            mlflow.delete_workspace(name, mode="CASCADE")
        except Exception:
            pass

    def test_list_workspaces(self, setup):
        workspaces = mlflow.list_workspaces()
        assert isinstance(workspaces, list)
        names = {ws.name for ws in workspaces}
        assert "default" in names

    def test_create_workspace(self, setup, workspace_name):
        created = mlflow.create_workspace(name=workspace_name, description="integ test workspace")
        assert created.name == workspace_name
        assert created.description == "integ test workspace"

    def test_get_workspace(self, setup, workspace_name):
        mlflow.create_workspace(name=workspace_name, description="get test")
        fetched = mlflow.get_workspace(workspace_name)
        assert fetched.name == workspace_name
        assert fetched.description == "get test"

    def test_update_workspace(self, setup, workspace_name):
        mlflow.create_workspace(name=workspace_name, description="before")
        updated = mlflow.update_workspace(name=workspace_name, description="after")
        assert updated.description == "after"
        # Verify via get
        fetched = mlflow.get_workspace(workspace_name)
        assert fetched.description == "after"

    def test_delete_workspace(self, setup, workspace_name):
        mlflow.create_workspace(name=workspace_name)
        mlflow.delete_workspace(workspace_name, mode="CASCADE")
        with pytest.raises(Exception):
            mlflow.get_workspace(workspace_name)

    def test_create_duplicate_workspace(self, setup, workspace_name):
        mlflow.create_workspace(name=workspace_name)
        with pytest.raises(Exception, match="(?i)already exists|RESOURCE_ALREADY_EXISTS"):
            mlflow.create_workspace(name=workspace_name)

    def test_get_nonexistent_workspace(self, setup):
        with pytest.raises(Exception):
            mlflow.get_workspace("nonexistent-ws-xyz")

    def test_create_invalid_name(self, setup):
        with pytest.raises(Exception):
            mlflow.create_workspace(name="INVALID_NAME!")

    def test_set_workspace(self, setup, workspace_name):
        mlflow.create_workspace(name=workspace_name)
        mlflow.set_workspace(workspace_name)
        # Reset
        mlflow.set_workspace(None)
