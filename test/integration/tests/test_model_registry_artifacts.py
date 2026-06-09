# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.

"""Integration tests for log_inference_specification, log_evaluation_group, and evaluate."""

import json
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import sagemaker_mlflow
from utils.mlflow_utils import mlflow_version

pytestmark = [
    pytest.mark.skipif(mlflow_version < (3, 0), reason="Requires MLflow >= 3.0"),
]

from sagemaker_mlflow.sagemaker_model_registry import (  # noqa: E402
    log_inference_specification,
    log_evaluation_group,
    InferenceSpecification,
)


@pytest.fixture(scope="module", autouse=True)
def setup_tracking(tracking_server):
    mlflow.set_tracking_uri(tracking_server)
    mlflow.set_experiment("sagemaker-mlflow-model-registry-integ-test")


def _log_sklearn_model():
    """Log a simple sklearn model and return model_info."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LinearRegression().fit(X, y)
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, name="test-model")
    return model_info


class TestLogInferenceSpecification:
    def test_logs_artifact_and_returns_uri(self):
        model_info = _log_sklearn_model()
        spec: InferenceSpecification = {
            "Containers": [
                {
                    "Image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/sklearn:latest",
                    "ModelDataSource": {
                        "S3DataSource": {
                            "S3Uri": "s3://my-bucket/my-model/",
                            "S3DataType": "S3Prefix",
                            "CompressionType": "None",
                        }
                    },
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        }

        uri = log_inference_specification(model_info.model_id, spec)

        assert "sagemaker_inference_specification.json" in uri
        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_logged_model_artifacts(model_info.model_id)]
        assert "sagemaker_inference_specification.json" in artifacts

    def test_auto_populates_s3_uri(self):
        model_info = _log_sklearn_model()
        spec: InferenceSpecification = {
            "Containers": [
                {
                    "Image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/sklearn:latest",
                }
            ],
        }

        log_inference_specification(model_info.model_id, spec)

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/sagemaker_inference_specification.json"
        )
        with open(artifact_path) as f:
            saved_spec = json.load(f)

        s3_uri = saved_spec["Containers"][0]["ModelDataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri == logged_model.artifact_location + "/"
        assert saved_spec["Containers"][0]["ModelDataSource"]["S3DataSource"]["S3DataType"] == "S3Prefix"
        assert saved_spec["Containers"][0]["ModelDataSource"]["S3DataSource"]["CompressionType"] == "None"

    def test_preserves_explicit_s3_uri(self):
        model_info = _log_sklearn_model()
        explicit_uri = "s3://custom-bucket/custom-path/"
        spec: InferenceSpecification = {
            "Containers": [
                {
                    "Image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/sklearn:latest",
                    "ModelDataSource": {
                        "S3DataSource": {
                            "S3Uri": explicit_uri,
                            "S3DataType": "S3Prefix",
                            "CompressionType": "None",
                        }
                    },
                }
            ],
        }

        log_inference_specification(model_info.model_id, spec)

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/sagemaker_inference_specification.json"
        )
        with open(artifact_path) as f:
            saved_spec = json.load(f)

        assert saved_spec["Containers"][0]["ModelDataSource"]["S3DataSource"]["S3Uri"] == explicit_uri


class TestLogEvaluationGroup:
    def test_with_dataset(self):
        model_info = _log_sklearn_model()

        eval_result = MagicMock()
        eval_result.metrics = {"rmse": 0.05, "r2": 0.99}

        dataset = MagicMock()
        dataset.to_dict.return_value = {
            "name": "test_set",
            "digest": "abc123",
            "source_type": "s3",
            "source": '{"uri": "s3://bucket/test.csv"}',
        }

        uri = log_evaluation_group(model_info.model_id, eval_result, dataset=dataset)

        assert "sagemaker_evaluation_group_test_set-abc123.json" in uri
        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/sagemaker_evaluation_group_test_set-abc123.json"
        )
        with open(artifact_path) as f:
            data = json.load(f)
        assert data["datasets"] == ["s3://bucket/test.csv"]
        assert data["name"] == "test_set-abc123"

    def test_without_dataset(self):
        model_info = _log_sklearn_model()

        eval_result = MagicMock()
        eval_result.metrics = {"f1": 0.88}

        uri = log_evaluation_group(model_info.model_id, eval_result)

        assert "sagemaker_evaluation_group_evaluation.json" in uri
        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{logged_model.artifact_location}/sagemaker_evaluation_group_evaluation.json"
        )
        with open(artifact_path) as f:
            data = json.load(f)
        assert "datasets" not in data
        assert data["name"] == "evaluation"


class TestEvaluate:
    def test_evaluate_with_dataset(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        model = LinearRegression().fit(X, y)

        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(model, name="eval-test-model")

        df = pd.DataFrame(X, columns=["a", "b"])
        df["target"] = y
        dataset = mlflow.data.from_pandas(df, targets="target", name="eval_set")

        result = sagemaker_mlflow.evaluate(model_info, data=dataset, model_type="regressor")

        assert result is not None
        assert "mean_absolute_error" in result.metrics or len(result.metrics) > 0

        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_logged_model_artifacts(model_info.model_id)]
        eval_artifacts = [a for a in artifacts if a.startswith("sagemaker_evaluation_group_")]
        assert len(eval_artifacts) == 1

    def test_evaluate_without_dataset(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 2, 3, 4])
        model = LinearRegression().fit(X, y)

        with mlflow.start_run():
            model_info = mlflow.sklearn.log_model(model, name="eval-no-ds-model")

        df = pd.DataFrame(X, columns=["a", "b"])
        df["target"] = y

        result = sagemaker_mlflow.evaluate(model_info, data=df, targets="target", model_type="regressor")

        assert result is not None
        assert len(result.metrics) > 0

        # Without an mlflow Dataset, evaluation group uses default name
        client = mlflow.MlflowClient()
        artifacts = [a.path for a in client.list_logged_model_artifacts(model_info.model_id)]
        eval_artifacts = [a for a in artifacts if a.startswith("sagemaker_evaluation_group_")]
        assert len(eval_artifacts) == 1
        assert any("evaluation" in a for a in eval_artifacts)
