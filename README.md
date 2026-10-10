# SageMaker MLflow Plugin

## What does this Plugin do?

This plugin generates Signature V4 headers in each outgoing request to the Amazon SageMaker with MLflow capability,
determines the URL of capability to connect to tracking servers, and registers models to the SageMaker Model Registry.
It generates a token with the SigV4 Algorithm that the service will use to conduct Authentication and Authorization
using AWS IAM.

## Installation

To install this plugin (lightweight, depends on `mlflow-skinny`):
```
pip install sagemaker-mlflow
```

To install with the full `mlflow` dependency set:
```
pip install sagemaker-mlflow[full]
```

> **Note:** The default install depends on `mlflow-skinny`. Most of the plugin
> (tracking store, auth provider, artifact repository, presigned uploads) works
> under skinny. The access-control admin client,
> [`SageMakerMlflowAuthClient`](#access-control-admin-client), is the exception:
> it extends MLflow's auth-server package, which is only in full `mlflow`, so it
> requires `sagemaker-mlflow[full]`.

To install from source:
```
pip install .
```

## Custom AWS session

By default, the plugin signs requests using credentials from the boto3 default
credential chain (environment variables, shared config, instance role, etc.).
Callers that need to sign with a specific `boto3.Session` — for example a
non-default profile or per-tenant credentials in a shared process — can inject
one without mutating `os.environ`:

```python
import boto3
import mlflow
import sagemaker_mlflow

custom = boto3.Session(profile_name="my-profile")

with sagemaker_mlflow.use_session(custom):
    mlflow.MlflowClient().search_experiments(max_results=1)
```

`use_session` is a context manager scoped to the current thread / asyncio task;
the previous session is restored on exit (including on exception).
`sagemaker_mlflow.set_session(session)` is also available for setting a default
that lasts for the rest of the context. Resolution order inside `AuthBoto`:
explicit `boto3_session=` kwarg → `use_session`/`set_session` → `boto3.Session()`.

## Presigned S3 artifact uploads

Set `SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED=true` to route recognized S3 artifact uploads through URLs issued by the MLflow tracking server. The tracking-server request uses SageMaker authentication, while the file is streamed directly to the returned S3 URL so those uploads do not require direct S3 write credentials.

```bash
export SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED=true
```

Once a presigned upload is attempted, request or PUT failures propagate to the caller; there is no silent fallback to direct S3. When the setting is disabled, the repository retains the standard MLflow direct-S3 behavior.

When the setting is enabled and the repository has a tracking URI, artifact roots that cannot be identified as run or logged-model targets fail before any server or S3 request; they do not fall back to direct S3. Repositories without a tracking URI retain the standard MLflow direct-S3 behavior.

### Warning: trace logging is not supported with presigned uploads

> **Warning:** Trace payload and attachment uploads are not currently supported when `SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED=true` and a tracking URI is configured. Trace logging fails closed before any server or S3 request and does not fall back to direct S3. To use trace logging until the server supports a `trace_id` upload scope, disable the setting and ensure that the client has direct S3 write permissions.

Presigned uploads for MLflow 3 logged-model artifacts (`log_model`) require both a client containing logged-model scope support and a tracking server containing [mlflow/mlflow#24765](https://github.com/mlflow/mlflow/pull/24765). Upgrading only one side does not enable the flow: an older client still sends the model ID as `run_id`, while a newer client sends `model_id`, which an older server does not support.

## Access control admin client

> **Requires `sagemaker-mlflow[full]`.** `SageMakerMlflowAuthClient` extends
> `mlflow.server.auth.client.AuthServiceClient`, which ships only with full
> `mlflow` (not `mlflow-skinny`). Importing `sagemaker_mlflow` stays safe under
> skinny; constructing the client without the auth-server package raises a clear
> error pointing you to `pip install 'sagemaker-mlflow[full]'`.

For SageMaker MLflow apps with fine-grained access control (FGAC) enabled,
`SageMakerMlflowAuthClient` is the admin client for managing MLflow RBAC. It is
a thin specialization of MLflow's
[`AuthServiceClient`](https://mlflow.org/docs/latest/api_reference/auth/python-api.html#mlflow.server.auth.client.AuthServiceClient)
with three SageMaker-specific differences:

* **IAM identity, not passwords.** A SageMaker MLflow app bridges AWS IAM to
  MLflow's user model: the IAM *role* ARN is the MLflow username and callers
  authenticate with SigV4. `create_user` takes only a role ARN (no password),
  and `update_user_password` is unsupported.
* **Role ARNs only.** Every method that names a user requires a well-formed IAM
  role ARN (pathless); user ARNs and other inputs are rejected client-side.
* **Tracking URI resolution.** The target app is taken from the active MLflow
  tracking URI (`mlflow.get_tracking_uri()`), the same as the other plugin
  helpers; pass `tracking_uri=` to override. The caller must be authenticated as
  the app's Platform Admin (the `AdminPrincipalArn` configured via
  `AccessControlConfig`).

The full RBAC surface of `AuthServiceClient` (roles, role permissions, role
assignments, per-user permission grants) is inherited unchanged.

```python
import mlflow
from sagemaker_mlflow import SageMakerMlflowAuthClient

mlflow.set_tracking_uri(
    "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/my-app"
)
client = SageMakerMlflowAuthClient()

# Register an IAM role as an MLflow user and promote it to Platform Admin.
client.create_user("arn:aws:iam::123456789012:role/AliceDSRole")
client.update_user_admin("arn:aws:iam::123456789012:role/AliceDSRole", is_admin=True)

# Inherited RBAC surface: create a role and assign it.
role = client.create_role(workspace="default", name="viewers")
client.assign_role("arn:aws:iam::123456789012:role/AliceDSRole", role.id)
```

## Development details

### setup.py

`setup.py` Contains the primary entry points for the sdk. 
`install_requires` Installs `mlflow-skinny` (lightweight) by default. The `[full]` extra installs the full `mlflow` package, which is required by `SageMakerMlflowAuthClient` (it extends MLflow's auth-server package).
`entry_points` Contains the entry points for the sdk. See https://mlflow.org/docs/latest/plugins.html#defining-a-plugin
for more details.

### Running tests

#### Setup
To run tests using tox, run:
```
pip install tox
```
Installing tox will enable users to run multi-environment tests. On the other hand, if
running individual tests in a single environment, feel free to continue to use pytest instead.

#### Running format checks
```
tox -e flake8,black-check,typing,twine
```

#### Formatting code to comply with format checks
```
tox -e black-format
```

#### Running unit tests
```
tox --skip-env "black.*|flake8|typing|twine" -- test/unit
```

#### Running integration tests
```
tox --skip-env "black.*|flake8|typing|twine" -- test/integration
```

#### Available test environments by default
tox.ini contains support for:
- Python 3.9: mlflow 2.8.*, 2.9.*, 2.10.*, 2.11.*, 2.12.*, 2.13.*, 2.16.*, 3.0.0
- Python 3.10/3.11: mlflow 2.8.*, 2.9.*, 2.10.*, 2.11.*, 2.12.*, 2.13.*, 2.16.*, 3.0.0, 3.4.0, 3.10.0

To add test environments on tox for additional versions of python or mlflow, modify the
environment configs in `envlist`, as well as `deps` and `depends` in `[testenv]`.
