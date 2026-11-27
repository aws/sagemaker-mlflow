# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from typing import TYPE_CHECKING

import importlib_metadata

from sagemaker_mlflow.session import set_session, use_session
from sagemaker_mlflow.sagemaker_model_registry import (
    InferenceSpecification,
    LineageInputs,
    evaluate,
    log_evaluation_group,
    log_inference_specification,
    log_lineage_inputs,
)

if TYPE_CHECKING:
    from sagemaker_mlflow.auth_client import SageMakerMlflowAuthClient

__version__ = importlib_metadata.version("sagemaker-mlflow")

__all__ = [
    "__version__",
    "set_session",
    "use_session",
    "SageMakerMlflowAuthClient",
    "InferenceSpecification",
    "LineageInputs",
    "evaluate",
    "log_evaluation_group",
    "log_inference_specification",
    "log_lineage_inputs",
]


def __getattr__(name):
    """Load ``SageMakerMlflowAuthClient`` on first use instead of at import time.

    MLflow imports this package while it is still initializing (it loads plugins).
    ``auth_client`` imports ``mlflow.server.auth.client`` at module load, and doing
    that mid-initialization hits a circular import inside MLflow, which leaves the
    client on a placeholder base class. Importing it here, on first access, runs
    after MLflow has finished initializing.
    """
    if name == "SageMakerMlflowAuthClient":
        from sagemaker_mlflow.auth_client import SageMakerMlflowAuthClient

        return SageMakerMlflowAuthClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
