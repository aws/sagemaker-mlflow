# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

from importlib import import_module
from unittest.mock import MagicMock, patch

from sagemaker_mlflow.sagemaker_model_registry.evaluate import log_evaluation_group

_evaluate_mod = import_module("sagemaker_mlflow.sagemaker_model_registry.evaluate")


class TestLogEvaluationResults:
    @patch.object(_evaluate_mod, "mlflow")
    def test_logs_artifact_with_metrics(self, mock_mlflow):
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client
        mock_logged_model = MagicMock()
        mock_logged_model.artifact_location = "s3://bucket/models/m-123/artifacts"
        mock_client.get_logged_model.return_value = mock_logged_model

        eval_result = MagicMock()
        eval_result.metrics = {"rmse": 0.05, "r2": 0.99}

        dataset = MagicMock()
        dataset.to_dict.return_value = {
            "name": "test_set",
            "digest": "abc123",
            "source_type": "s3",
            "source": '{"uri": "s3://bucket/test.csv"}',
        }

        uri = log_evaluation_group("m-123", eval_result, dataset=dataset)

        mock_client.log_model_artifact.assert_called_once()
        assert "sagemaker_evaluation_group_test_set-abc123.json" in uri

    @patch.object(_evaluate_mod, "mlflow")
    def test_without_dataset(self, mock_mlflow):
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client
        mock_logged_model = MagicMock()
        mock_logged_model.artifact_location = "s3://bucket/models/m-456/artifacts"
        mock_client.get_logged_model.return_value = mock_logged_model

        eval_result = MagicMock()
        eval_result.metrics = {"accuracy": 0.95}

        uri = log_evaluation_group("m-456", eval_result)

        mock_client.log_model_artifact.assert_called_once()
        assert "sagemaker_evaluation_group_evaluation.json" in uri

    @patch.object(_evaluate_mod, "mlflow")
    def test_metric_values_stored_as_strings(self, mock_mlflow):
        mock_client = MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client
        mock_logged_model = MagicMock()
        mock_logged_model.artifact_location = "s3://bucket/models/m-789/artifacts"
        mock_client.get_logged_model.return_value = mock_logged_model

        eval_result = MagicMock()
        eval_result.metrics = {"loss": 0.001}

        import json

        # Capture the file content before cleanup by intercepting log_model_artifact
        written_content = {}

        def capture_artifact(model_id, path):
            with open(path) as f:
                written_content.update(json.load(f))

        mock_client.log_model_artifact.side_effect = capture_artifact

        log_evaluation_group("m-789", eval_result)

        assert written_content["metric_groups"][0]["metric_data"][0]["value"] == "0.001"
        assert written_content["metric_groups"][0]["metric_data"][0]["type"] == "number"
