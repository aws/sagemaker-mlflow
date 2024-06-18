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

"""
sagemaker-mlflow plugin installation.
"""
import os

from setuptools import setup, find_packages


def read(fname):
    """
    Args:
        fname:
    """
    with open(os.path.join(os.path.dirname(__file__), fname), "r") as f:
        contents = f.read()
    return contents


def read_version():
    return read("VERSION").strip()


def read_requirements(filename):
    """Reads requirements file which lists package dependencies.

    Args:
        filename: type(str) Relative file path of requirements.txt file

    Returns:
        list of dependencies extracted from file
    """
    with open(os.path.abspath(filename)) as fp:
        deps = [line.strip() for line in fp.readlines()]
    return deps


test_requirements = read_requirements("requirements/integration_test_requirements.txt")
test_prerelease_requirements = read_requirements("requirements/prerelease_test_requirements.txt")

setup(
    name="sagemaker-mlflow",
    packages=find_packages(),
    author="Amazon Web Services",
    license="Apache License 2.0",
    url = 'https://github.com/aws/sagemaker-mlflow',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # Require MLflow as a dependency of the plugin, so that plugin users can
    # simply install the plugin and then immediately use it with MLflow
    install_requires=["boto3>=1.34", "mlflow>=2.8"],
    extras_require={"test": test_requirements, "test_prerelease": test_prerelease_requirements},
    python_requires=">= 3.8",
    entry_points={
        "mlflow.tracking_store": "arn=sagemaker_mlflow.mlflow_sagemaker_store:MlflowSageMakerStore",
        "mlflow.request_auth_provider": "arn=sagemaker_mlflow.auth_provider:AuthProvider",
        "mlflow.request_header_provider": "arn=sagemaker_mlflow.mlflow_sagemaker_request_header_provider:MlflowSageMakerRequestHeaderProvider",
        "mlflow.model_registry_store": "arn=sagemaker_mlflow.mlflow_sagemaker_registry_store:MlflowSageMakerRegistryStore"
    },
    version=read_version(),
    description="AWS Plugin for MLflow with SageMaker",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
