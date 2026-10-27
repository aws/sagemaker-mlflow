# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Types for declaring the upstream data inputs of a logged model.

The set of input types is closed. The server plugin owns resolution, and it
skips a declaration it cannot resolve with a warning the customer never sees,
so an unrecognized ``Type`` -- or one missing its identifier -- is rejected at
the call site instead. Adding a type means releasing the client alongside the
server resolver.
"""

from typing import List, Literal, TypedDict, Union


class FeatureGroupInput(TypedDict, total=False):
    """A SageMaker Feature Store feature group that fed the model.

    Resolves server-side to the feature group's auto-created lineage Context.

    ``SnapshotId`` is optional: the Iceberg snapshot of the feature group's
    offline store that the model trained on, which pins the data version. Only
    an offline store in the Iceberg table format has snapshots, so omit it for
    a Glue-format or online-only feature group.
    """

    Type: Literal["FeatureGroup"]
    Arn: str
    SnapshotId: str


class HubDataSetInput(TypedDict, total=False):
    """A SageMaker hub dataset that fed the model.

    ``Arn`` is the versioned hub content ARN,
    ``arn:...:hub-content/<hub>/DataSet/<name>/<version>``. A hub dataset
    version never changes, so the version in the ARN already pins the data.
    Resolves server-side to the lineage node the hub dataset already has.

    ``QueryString`` is optional: the Athena query that built the dataset from its
    feature groups, so the dataset can be reproduced.
    """

    Type: Literal["HubDataSet"]
    Arn: str
    QueryString: str


LineageInput = Union[FeatureGroupInput, HubDataSetInput]

# What ``log_lineage_inputs`` accepts: the inputs directly, as a list.
LineageInputs = List[LineageInput]

# The declaration is only meaningful for a type the server plugin can resolve.
# Anything else is rejected at the call site rather than written to the artifact
# and silently dropped server-side, where the customer never sees the warning.
# Adding a type here requires a client release alongside the server resolver.
SUPPORTED_INPUT_TYPES = frozenset({"FeatureGroup", "HubDataSet"})

# The identifier each type is resolved by. A declaration missing it would be
# skipped server-side for the same invisible reason.
REQUIRED_INPUT_FIELDS = {
    "FeatureGroup": "Arn",
    "HubDataSet": "Arn",
}
