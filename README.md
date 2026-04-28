# SageMaker MLflow Plugin

## Why use this Plugin?

Mlflow expects to be provided with a url to a tracking server, AWS services use ARNs which cannot be consumed directly.

## What does this Plugin do?

This plugin generates Signature V4 headers in each outgoing request to AWS hosted MLFlow instances (Sagemaker and App),
determines the URL to connect to tracking servers, then generates a token with the SigV4 Algorithm that the service will use 
to conduct Authentication and Authorization using AWS IAM.

## Installation

To install this plugin, run the following command inside the directory:
```
pip install .
```

Eventually when the plugin gets distributed, it will be installed with:
```
pip install sagemaker-mlflow
```

Running this will install the Auth Plugin and mlflow.

To install a specific mlflow version

```
pip install .
pip install mlflow==2.13
```
## Usage

This plugin leverages the boto3 package, and should align with it's patterns.

### Assuming roles

In order to run this with an assumed role, first use [boto3.setup_default_session()](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/boto3.html)

#### Example with SSO
```
base_session = boto3.Session(
  profile_name=profile_name_str, region_name=region_str
)
response = sts.assume_role(RoleArn=assumed_role_arn, RoleSessionName="AssumedSession")
creds = response["Credentials"]
boto3.setup_default_session(
  aws_access_key_id=creds["AccessKeyId"],
  aws_secret_access_key=creds["SecretAccessKey"],
  aws_session_token=creds["SessionToken"],
  region_name=base_session.region_name,
)
```

## Development details

### setup.py

`setup.py` Contains the primary entry points for the sdk. 
`install_requires` Installs mlflow.
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
