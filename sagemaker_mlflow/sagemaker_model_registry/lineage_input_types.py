# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""Types for declaring the upstream data inputs of a logged model.

These types are for editor ergonomics only. The server plugin owns resolution
of every input type, so an unrecognized ``Type`` is forwarded as-is rather than
rejected -- that is what lets a new input type ship without a client release.
"""

from typing import Any, Dict, List, Literal, TypedDict, Union


class FeatureGroupInput(TypedDict, total=False):
    """A SageMaker Feature Store feature group that fed the model.

    Resolves server-side to the feature group's auto-created lineage Context.
    """

    Type: Literal["FeatureGroup"]
    Arn: str


class S3TableSnapshotInput(TypedDict, total=False):
    """A point-in-time S3 table snapshot that fed the model.

    Resolves server-side to a lineage Artifact, created if absent.
    """

    Type: Literal["S3TableSnapshot"]
    Uri: str
    SnapshotId: str
    Name: str


# ``Dict[str, Any]`` is a deliberate escape hatch: an input type the installed
# client predates is still a valid declaration.
LineageInput = Union[FeatureGroupInput, S3TableSnapshotInput, Dict[str, Any]]

# What ``log_lineage_inputs`` accepts: the inputs directly, as a list.
LineageInputs = List[LineageInput]
