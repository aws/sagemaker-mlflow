# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Tests for log_lineage_inputs and its input-list validation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker_mlflow.sagemaker_model_registry.lineage_inputs import (
    ARTIFACT_NAME,
    log_lineage_inputs,
)

ARTIFACT_LOCATION = "s3://bucket/models/m-123/artifacts"

FEATURE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:000000000000:feature-group/features-v0-0-2"


def _mock_client(mock_client_cls):
    """Wire a MlflowClient mock that captures the uploaded artifact's contents."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_logged_model = MagicMock()
    mock_logged_model.artifact_location = ARTIFACT_LOCATION
    mock_client.get_logged_model.return_value = mock_logged_model

    captured = {}

    def capture_artifact(model_id, path):
        captured["path"] = path
        with open(path) as f:
            captured["content"] = json.load(f)

    mock_client.log_model_artifact.side_effect = capture_artifact
    return mock_client, captured


class TestLogLineageInputs:
    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_writes_inputs_verbatim_under_inputs_key(self, mock_client_cls):
        """The caller passes a list; the artifact wraps it in the wire envelope."""
        mock_client, captured = _mock_client(mock_client_cls)

        inputs = [
            {"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN},
            {
                "Type": "S3TableSnapshot",
                "Uri": "s3://bucket/flight-delay-737-dep15/v0.0.2/",
                "SnapshotId": "7281539204561098345",
            },
        ]
        log_lineage_inputs("m-123", lineage_inputs=inputs)

        mock_client.log_model_artifact.assert_called_once()
        assert captured["content"] == {"Inputs": inputs}

    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_caller_list_is_not_mutated(self, mock_client_cls):
        _mock_client(mock_client_cls)

        inputs = [{"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN}]
        log_lineage_inputs("m-123", lineage_inputs=inputs)

        assert inputs == [{"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN}]

    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_artifact_filename_and_returned_uri(self, mock_client_cls):
        _, captured = _mock_client(mock_client_cls)

        uri = log_lineage_inputs(
            "m-123", lineage_inputs=[{"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN}]
        )

        assert captured["path"].endswith(f"/{ARTIFACT_NAME}")
        assert uri == f"{ARTIFACT_LOCATION}/{ARTIFACT_NAME}"

    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_forwards_unknown_type_unchanged(self, mock_client_cls):
        """A type newer than the client must survive the round trip untouched."""
        _, captured = _mock_client(mock_client_cls)

        entry = {"Type": "SomeFutureType", "Whatever": {"nested": ["value"]}}
        log_lineage_inputs("m-123", lineage_inputs=[entry])

        assert captured["content"]["Inputs"] == [entry]

    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_does_not_resolve_anything(self, mock_client_cls):
        """The client only records the declaration -- no describe/list calls."""
        mock_client, _ = _mock_client(mock_client_cls)

        log_lineage_inputs(
            "m-123", lineage_inputs=[{"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN}]
        )

        assert mock_client.method_calls
        called = {call[0] for call in mock_client.method_calls}
        assert called == {"get_logged_model", "log_model_artifact"}


class TestInputsValidation:
    @pytest.mark.parametrize(
        "bad,expected",
        [
            ([], ValueError),
            ("Inputs", TypeError),
            (None, TypeError),
            (["not-a-dict"], TypeError),
            ([{"Arn": FEATURE_GROUP_ARN}], ValueError),
            ([{"Type": "", "Arn": FEATURE_GROUP_ARN}], ValueError),
        ],
    )
    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_rejects_malformed_inputs(self, mock_client_cls, bad, expected):
        mock_client, _ = _mock_client(mock_client_cls)

        with pytest.raises(expected):
            log_lineage_inputs("m-123", lineage_inputs=bad)

        mock_client.log_model_artifact.assert_not_called()

    @pytest.mark.parametrize(
        "envelope",
        [
            {"Inputs": [{"Type": "FeatureGroup", "Arn": FEATURE_GROUP_ARN}]},
            {"Inputs": []},
            {},
        ],
    )
    @patch("sagemaker_mlflow.sagemaker_model_registry.lineage_inputs.mlflow.MlflowClient")
    def test_old_envelope_dict_is_rejected_with_guidance(self, mock_client_cls, envelope):
        """The previous {"Inputs": [...]} parameter shape must fail loudly, not silently."""
        mock_client, _ = _mock_client(mock_client_cls)

        with pytest.raises(TypeError, match="must be a list of input dicts"):
            log_lineage_inputs("m-123", lineage_inputs=envelope)

        mock_client.log_model_artifact.assert_not_called()
