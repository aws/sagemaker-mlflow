# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Log evaluation results as SageMaker evaluation group artifacts."""

import json
import logging
import os
import tempfile

import mlflow

logger = logging.getLogger(__name__)

EVAL_GROUP_PREFIX = "sagemaker_evaluation_group_"


def evaluate(model_info, **kwargs):
    """Evaluate a model and log results as a SageMaker evaluation group artifact.

    Convenience wrapper: calls mlflow.evaluate(), then logs the results as an
    evaluation group artifact on the logged model.

    Args:
        model_info: ModelInfo returned by log_model (provides model_id and model_uri).
        **kwargs: All arguments forwarded to mlflow.evaluate()
            (e.g. data, targets, model_type, evaluators, etc.).

    Returns:
        EvaluationResult from mlflow.evaluate().
    """
    result = mlflow.evaluate(model=model_info.model_uri, **kwargs)
    log_evaluation_group(model_info.model_id, result, dataset=kwargs.get("data"))
    return result


def log_evaluation_group(
    model_id: str,
    evaluation_result: "mlflow.models.EvaluationResult",
    dataset: "mlflow.data.Dataset | None" = None,
) -> str:
    """Log an EvaluationResult as a SageMaker evaluation group artifact.

    The server-side plugin reads these at create_model_version time to
    populate evaluation_details in the Model Card.

    Args:
        model_id: MLflow logged model ID (e.g. "m-abc123").
        evaluation_result: EvaluationResult from mlflow.evaluate().
        dataset: Optional mlflow Dataset used for evaluation (provides name and source).

    Returns:
        The artifact URI of the uploaded artifact.
    """
    ds_name, ds_digest, ds_source_uri = _extract_dataset_info(dataset)

    group_name = f"{ds_name}-{ds_digest}" if ds_digest else ds_name
    eval_group = {
        "name": group_name,
        "metric_groups": [
            {
                "name": "Metrics",
                "metric_data": [
                    {"name": k, "value": str(v), "type": "number"} for k, v in evaluation_result.metrics.items()
                ],
            }
        ],
    }
    if ds_source_uri:
        eval_group["datasets"] = [ds_source_uri]

    artifact_name = f"{EVAL_GROUP_PREFIX}{group_name}.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, artifact_name)
        with open(path, "w") as f:
            json.dump(eval_group, f, indent=2)
        client = mlflow.MlflowClient()
        client.log_model_artifact(model_id, path)

    logged_model = client.get_logged_model(model_id)
    return f"{logged_model.artifact_location}/{artifact_name}"


def _extract_dataset_info(data):
    """Extract name, digest, and source URI from an mlflow Dataset.

    Returns:
        Tuple of (name, digest, source_uri).
    """
    if data is None:
        return "evaluation", None, None

    if hasattr(data, "to_dict"):
        ds_dict = data.to_dict()
        name = ds_dict.get("name", "dataset")
        digest = ds_dict.get("digest")
        source_type = ds_dict.get("source_type", "code")
        source_uri = None
        if source_type != "code":
            try:
                source_uri = json.loads(ds_dict.get("source", "{}")).get("uri")
            except (json.JSONDecodeError, TypeError):
                pass
        return name, digest, source_uri

    return "dataset", None, None
