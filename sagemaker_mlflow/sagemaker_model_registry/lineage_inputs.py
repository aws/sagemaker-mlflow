# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Manage the sagemaker_lineage_inputs.json artifact on logged models."""

import json
import logging
import os
import tempfile

import mlflow

from sagemaker_mlflow.sagemaker_model_registry.lineage_input_types import LineageInputs

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "sagemaker_lineage_inputs.json"


def log_lineage_inputs(model_id: str, lineage_inputs: LineageInputs) -> str:
    """Declare the upstream data inputs that produced a logged model.

    Writes the declaration as a JSON artifact on the logged model. No AWS calls
    are made here and nothing is resolved: when the model is registered, the
    SageMaker model registry server plugin reads this artifact and associates
    each declared input upstream of the model's lineage. Declaring an input for
    a model that is never registered has no effect.

    Only the list's shape is validated. Each entry's ``Type`` and its identifier
    fields are interpreted server-side, so an input type newer than the
    installed client is forwarded rather than rejected.

    Args:
        model_id: MLflow logged model ID (e.g. "m-abc123").
        lineage_inputs: List of declared inputs. Each entry needs a ``Type``
            plus the identifier fields for that type -- ``Arn`` for
            ``FeatureGroup``, ``Uri`` (and optionally ``SnapshotId``) for
            ``S3TableSnapshot``.

    Returns:
        The artifact URI of the uploaded artifact.

    Raises:
        TypeError: If the declaration is not a list, or an entry is not a dict.
        ValueError: If the list is empty, or an entry has no ``Type``.

    Example:
        >>> log_lineage_inputs(
        ...     model_info.model_id,
        ...     lineage_inputs=[
        ...         {"Type": "FeatureGroup", "Arn": "arn:aws:sagemaker:..."},
        ...         {"Type": "S3TableSnapshot", "Uri": "s3://bucket/table/"},
        ...     ],
        ... )
    """
    _validate_inputs(lineage_inputs)

    client = mlflow.MlflowClient()
    logged_model = client.get_logged_model(model_id)

    # The artifact keeps a top-level object with an "Inputs" key rather than a bare
    # JSON array: it is the client/server contract, and an object leaves room for
    # forward-compatible top-level keys without changing the artifact's shape.
    declaration = {"Inputs": list(lineage_inputs)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, ARTIFACT_NAME)
        with open(config_path, "w") as f:
            json.dump(declaration, f, indent=2)
        client.log_model_artifact(model_id, config_path)

    return f"{logged_model.artifact_location}/{ARTIFACT_NAME}"


def _validate_inputs(lineage_inputs: LineageInputs) -> None:
    """Check the list's shape without interpreting any input type."""
    if isinstance(lineage_inputs, dict):
        raise TypeError(
            "lineage_inputs must be a list of input dicts, not a dict. Pass the inputs "
            'directly, e.g. [{"Type": "FeatureGroup", "Arn": ...}]'
        )
    if not isinstance(lineage_inputs, list):
        raise TypeError(
            f"lineage_inputs must be a list, got {type(lineage_inputs).__name__}"
        )
    if not lineage_inputs:
        raise ValueError("lineage_inputs must not be empty")

    for index, entry in enumerate(lineage_inputs):
        if not isinstance(entry, dict):
            raise TypeError(
                f"lineage_inputs[{index}] must be a dict, got {type(entry).__name__}"
            )
        if not entry.get("Type"):
            raise ValueError(f"lineage_inputs[{index}] is missing a non-empty 'Type'")
