# SageMaker MLflow integration tests

## Usage

### Using a pre-created Tracking Server

- Create an mlflow tracking server and note down its arn. 
- Set `MLFLOW_TRACKING_SERVER_URI` to the arn.
- In the `test/integration` directory, run `pytest` (You may need to run python`(python version)` -m pytest)
