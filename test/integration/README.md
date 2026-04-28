# SageMaker MLflow integration tests

## Usage

### Using a pre-created Tracking Server

- Create an mlflow tracking server and note down its arn.
- Set `MLFLOW_TRACKING_SERVER_URI` to the arn.
- In the `test/integration` directory, run `pytest` (You may need to run python`(python version)` -m pytest)

### Using a pre-created MLflow App

- Create an mlflow app and note down its arn.
- Set `MLFLOW_TRACKING_SERVER_URI` to the arn.
- In the `test/integration` directory, run `pytest` (You may need to run python`(python version)` -m pytest)
- Note: `test_scorer` requires mlflow>=3.4 and `test_workspace` requires mlflow>=3.10. These tests are skipped on older versions.