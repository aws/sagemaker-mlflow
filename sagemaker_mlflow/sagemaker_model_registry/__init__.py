import mlflow

_MLFLOW_3 = int(mlflow.__version__.split(".")[0]) >= 3

if _MLFLOW_3:
    from sagemaker_mlflow.sagemaker_model_registry.evaluate import evaluate, log_evaluation_group
    from sagemaker_mlflow.sagemaker_model_registry.inference_specification import log_inference_specification
    from sagemaker_mlflow.sagemaker_model_registry.inference_spec_types import InferenceSpecification
else:

    def evaluate(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError("sagemaker_mlflow.evaluate requires MLflow >= 3.0")

    def log_evaluation_group(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError("sagemaker_mlflow.log_evaluation_group requires MLflow >= 3.0")

    def log_inference_specification(*args, **kwargs):  # type: ignore[misc]
        raise NotImplementedError("sagemaker_mlflow.log_inference_specification requires MLflow >= 3.0")

    InferenceSpecification = dict  # type: ignore[misc,assignment]

__all__ = [
    "InferenceSpecification",
    "evaluate",
    "log_evaluation_group",
    "log_inference_specification",
]
