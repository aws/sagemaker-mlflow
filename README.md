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

To install from source:
```
pip install .
```

## Configuring the tracking server URI

Set the ARN of the tracking server to as the MLflow tracking server URI,
using either the `MLFLOW_TRACKING_URI="arn:..."` environment variable or
`mlflow.set_tracking_uri("arn:...")`. This plugin registers itself as the
handler for the `arn:` scheme, and MLflow knows to invoke this plugin
for accessing the tracking server.

## Assuming another role

The plugin can be configured to assume a specific IAM role for invoking any AWS
APIs. Set the environment variable `SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN` to the
ARN of the desired role to use that role.

This is useful for, e.g., accessing MLflow tracking server in another AWS account:
create a role in the same account as the tracking server, assign sufficient
identity-based permissions to the role to access MLflow, and create a trust policy
to allow the identity used by this plugin to assume the role.

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

## Development details

### setup.py

`setup.py` Contains the primary entry points for the sdk. 
`install_requires` Installs `mlflow-skinny` (lightweight) by default. The `[full]` extra installs the full `mlflow` package.
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
