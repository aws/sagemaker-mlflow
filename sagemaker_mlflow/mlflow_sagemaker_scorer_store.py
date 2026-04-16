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

from mlflow.genai.scorers.registry import MlflowTrackingStore


class MlflowSageMakerScorerStore(MlflowTrackingStore):
    """Scorer store that enables the ``arn`` URI scheme for SageMaker MLflow.

    MlflowTrackingStore delegates all scorer operations to the tracking store
    resolved by ``_get_store(tracking_uri)``, which already handles the ``arn``
    scheme via the ``mlflow.tracking_store`` entrypoint (MlflowSageMakerStore).
    This wrapper only exists so the ``mlflow.scorer_store`` entrypoint registry
    recognises the ``arn`` scheme.
    """

    def __init__(self, tracking_uri=None):
        super().__init__(tracking_uri)
