import os

import mlflow

mlflow_version = tuple(int(x) for x in mlflow.__version__.split(".")[:2])
is_mlflow_app = "mlflow-app" in os.environ.get("MLFLOW_TRACKING_SERVER_URI", "")
