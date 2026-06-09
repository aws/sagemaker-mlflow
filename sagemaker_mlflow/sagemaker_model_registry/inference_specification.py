# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Manage the sagemaker_inference_specification.json artifact on logged models."""

import copy
import json
import logging
import os
import tempfile

import mlflow

from sagemaker_mlflow.sagemaker_model_registry.inference_spec_types import InferenceSpecification

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "sagemaker_inference_specification.json"


def log_inference_specification(
    model_id: str,
    inference_specification: InferenceSpecification,
) -> str:
    """Log an InferenceSpecification as a JSON artifact on a logged model.

    If a container's ModelDataSource.S3DataSource.S3Uri is not set, it is
    auto-populated with the logged model's artifact location.

    Args:
        model_id: MLflow logged model ID (e.g. "m-abc123").
        inference_specification: InferenceSpecification dict matching the
            SageMaker CreateModelPackage schema.

    Returns:
        The artifact URI of the uploaded artifact.
    """
    client = mlflow.MlflowClient()
    logged_model = client.get_logged_model(model_id)
    spec = _populate_s3_uri(inference_specification, logged_model.artifact_location)

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, ARTIFACT_NAME)
        with open(config_path, "w") as f:
            json.dump(spec, f, indent=2)
        client.log_model_artifact(model_id, config_path)

    return f"{logged_model.artifact_location}/{ARTIFACT_NAME}"


def _populate_s3_uri(spec: InferenceSpecification, artifact_location: str) -> InferenceSpecification:
    """Auto-fill S3Uri on containers that don't have it set."""
    spec = copy.deepcopy(spec)
    for container in spec.get("Containers", []):
        s3_source = container.get("ModelDataSource", {}).get("S3DataSource", {})
        if not s3_source.get("S3Uri") and not container.get("ModelDataUrl"):
            container.setdefault("ModelDataSource", {}).setdefault("S3DataSource", {})
            container["ModelDataSource"]["S3DataSource"]["S3Uri"] = artifact_location + "/"
            container["ModelDataSource"]["S3DataSource"].setdefault("S3DataType", "S3Prefix")
            container["ModelDataSource"]["S3DataSource"].setdefault("CompressionType", "None")
    return spec
