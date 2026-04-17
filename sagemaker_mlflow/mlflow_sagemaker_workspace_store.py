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

from mlflow.store.workspace.rest_store import RestWorkspaceStore

from sagemaker_mlflow.host_creds import get_host_creds


class MlflowSageMakerWorkspaceStore(RestWorkspaceStore):

    def __init__(self, workspace_uri):
        super().__init__(partial(get_host_creds, workspace_uri))
