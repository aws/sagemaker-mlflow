# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Tests for sagemaker_config and type compatibility."""

from unittest.mock import MagicMock, patch

import boto3
from botocore.validate import ParamValidator

from sagemaker_mlflow.sagemaker_model_registry.inference_spec_types import InferenceSpecification
from sagemaker_mlflow.sagemaker_model_registry.inference_specification import log_inference_specification


class TestInferenceSpecTypes:
    def test_typed_dict_keys_match_botocore_schema(self):
        """Verify our TypedDict fields are recognized by botocore's InferenceSpecification shape."""
        service_model = boto3.client("sagemaker", region_name="us-east-1")._service_model
        shape = service_model.shape_for("InferenceSpecification")
        botocore_members = set(shape.members.keys())
        typed_dict_keys = set(InferenceSpecification.__annotations__.keys())

        unknown = typed_dict_keys - botocore_members
        assert not unknown, f"TypedDict has keys not in botocore schema: {unknown}"

    def test_full_spec_validates_against_botocore(self):
        """A fully-populated InferenceSpecification passes botocore validation."""
        spec: InferenceSpecification = {
            "Containers": [
                {
                    "Image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
                    "ModelDataSource": {
                        "S3DataSource": {
                            "S3Uri": "s3://bucket/model/",
                            "S3DataType": "S3Prefix",
                            "CompressionType": "None",
                        }
                    },
                    "Environment": {"KEY": "value"},
                    "ImageDigest": "sha256:abc123",
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge"],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
        }

        service_model = boto3.client("sagemaker", region_name="us-east-1")._service_model
        shape = service_model.shape_for("InferenceSpecification")
        validator = ParamValidator()
        report = validator.validate(spec, shape)
        assert not report.has_errors(), report.generate_report()


class TestLogSageMakerConfig:
    @patch("sagemaker_mlflow.sagemaker_model_registry.inference_specification.mlflow.MlflowClient")
    def test_writes_inference_spec_and_populates_s3_uri(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_logged_model = MagicMock()
        mock_logged_model.artifact_location = "s3://bucket/models/m-123/artifacts"
        mock_client.get_logged_model.return_value = mock_logged_model

        import json

        written_content = {}

        def capture_artifact(model_id, path):
            with open(path) as f:
                written_content.update(json.load(f))

        mock_client.log_model_artifact.side_effect = capture_artifact

        spec = {"Containers": [{"Image": "img:latest"}]}
        log_inference_specification("m-123", inference_specification=spec)

        mock_client.log_model_artifact.assert_called_once()
        assert (
            written_content["Containers"][0]["ModelDataSource"]["S3DataSource"]["S3Uri"]
            == "s3://bucket/models/m-123/artifacts/"
        )
