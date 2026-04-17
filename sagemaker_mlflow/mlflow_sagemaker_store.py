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

from functools import partial

from mlflow.store.tracking.rest_store import RestStore

from sagemaker_mlflow.host_creds import get_host_creds


class MlflowSageMakerStore(RestStore):
    store_uri = ""

    def __init__(self, store_uri, artifact_uri):
        self.store_uri = store_uri
        super().__init__(partial(get_host_creds, store_uri))
