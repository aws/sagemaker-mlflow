# SageMaker MLflow Plugin

## What does this Plugin do?

This plugin generates Signature V4 headers in each outgoing request to the Amazon SageMaker with MLflow capability,
determines the URL of capability to connect to tracking servers, and registers models to the SageMaker Model Registry.
It generates a token with the SigV4 Algorithm that the service will use to conduct Authentication and Authorization
using AWS IAM.

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
- Python 3.10/3.11: mlflow 2.8.*, 2.9.*, 2.10.*, 2.11.*, 2.12.*, 2.13.*, 2.16.*, 3.0.0, 3.4.0

To add test environments on tox for additional versions of python or mlflow, modify the
environment configs in `envlist`, as well as `deps` and `depends` in `[testenv]`.
