# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License.

"""Integration tests for sagemaker_mlflow.log_model against a live MLflow tracking server."""

import os
import yaml
import tarfile
import pytest
import mlflow
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

import sagemaker_mlflow
from sagemaker_mlflow.log_model import _default_image_registry

DUMMY_IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-inference:v1"
DUMMY_IMAGE_SKLEARN = "123456789012.dkr.ecr.us-west-2.amazonaws.com/sklearn-serve:v1"
DUMMY_IMAGE_PYTORCH = "123456789012.dkr.ecr.us-west-2.amazonaws.com/pytorch-serve:v1"


@pytest.fixture(scope="module", autouse=True)
def setup_tracking(tracking_server):
    mlflow.set_tracking_uri(tracking_server)
    mlflow.set_experiment("sagemaker-mlflow-log-model-integ-test")
    yield


@pytest.fixture(autouse=True)
def clear_defaults():
    _default_image_registry.clear()
    yield
    _default_image_registry.clear()


def _train_sklearn_model():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = LinearRegression()
    model.fit(X, y)
    return model


class TestLogModelSklearn:
    def test_with_explicit_image_uri(self):
        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-sklearn-with-image-uri",
                inference_image_uri=DUMMY_IMAGE_URI,
            )

        assert model_info.model_id is not None

        # Verify logged model tags
        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_URI
        assert logged_model.tags["sagemaker.inference_flavor"] == "sklearn"
        assert logged_model.tags["sagemaker.compressed"] == "False"
        assert "sagemaker.model_data_url" in logged_model.tags

        # Verify MLmodel metadata from S3
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_info.model_uri)
        with open(os.path.join(local_path, "MLmodel")) as f:
            mlmodel = yaml.safe_load(f)

        assert mlmodel["metadata"]["sagemaker.inference_image_uri"] == DUMMY_IMAGE_URI
        assert mlmodel["metadata"]["sagemaker.inference_flavor"] == "sklearn"
        assert mlmodel["metadata"]["sagemaker.compressed"] == "False"

    def test_without_image_uri(self):
        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-sklearn-no-image-uri",
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert "sagemaker.inference_image_uri" not in logged_model.tags
        assert logged_model.tags["sagemaker.inference_flavor"] == "sklearn"
        assert logged_model.tags["sagemaker.compressed"] == "False"

    def test_compress_true(self):
        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-sklearn-compressed",
                inference_image_uri=DUMMY_IMAGE_SKLEARN,
                log_compressed_model=True,
            )

        # Verify tags
        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.model_data_url"].endswith("/model.tar.gz")
        assert logged_model.tags["sagemaker.model_data_url"].startswith("s3://")
        assert logged_model.tags["sagemaker.compressed"] == "True"

        # Download, extract, and verify contents
        import tempfile

        tar_s3_uri = logged_model.tags["sagemaker.model_data_url"]
        local_tar = mlflow.artifacts.download_artifacts(artifact_uri=tar_s3_uri)
        assert tarfile.is_tarfile(local_tar)

        with tempfile.TemporaryDirectory() as extract_dir:
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(extract_dir)

            extracted = os.listdir(extract_dir)
            assert "MLmodel" in extracted
            assert "model.pkl" in extracted
            assert "conda.yaml" in extracted
            assert "requirements.txt" in extracted

            # Verify MLmodel metadata inside tar
            with open(os.path.join(extract_dir, "MLmodel")) as f:
                mlmodel = yaml.safe_load(f)
            assert mlmodel["metadata"]["sagemaker.inference_image_uri"] == DUMMY_IMAGE_SKLEARN
            assert mlmodel["metadata"]["sagemaker.compressed"] == "True"


class TestLogModelPytorch:
    def test_with_explicit_image_uri(self):
        model = torch.nn.Linear(2, 1)

        with mlflow.start_run():
            model_info = sagemaker_mlflow.log_model(
                "pytorch",
                pytorch_model=model,
                artifact_path="test-pytorch-with-image-uri",
                inference_image_uri=DUMMY_IMAGE_PYTORCH,
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_PYTORCH
        assert logged_model.tags["sagemaker.inference_flavor"] == "pytorch"
        assert logged_model.tags["sagemaker.compressed"] == "False"

        # Verify MLmodel metadata
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_info.model_uri)
        with open(os.path.join(local_path, "MLmodel")) as f:
            mlmodel = yaml.safe_load(f)

        assert mlmodel["metadata"]["sagemaker.inference_image_uri"] == DUMMY_IMAGE_PYTORCH
        assert mlmodel["metadata"]["sagemaker.inference_flavor"] == "pytorch"

    def test_compress_true(self):
        model = torch.nn.Linear(2, 1)

        with mlflow.start_run():
            model_info = sagemaker_mlflow.log_model(
                "pytorch",
                pytorch_model=model,
                artifact_path="test-pytorch-compressed",
                inference_image_uri=DUMMY_IMAGE_PYTORCH,
                log_compressed_model=True,
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.model_data_url"].startswith("s3://")
        assert logged_model.tags["sagemaker.model_data_url"].endswith("/model.tar.gz")
        assert logged_model.tags["sagemaker.compressed"] == "True"

        # Download, extract, and verify contents
        import tempfile

        tar_s3_uri = logged_model.tags["sagemaker.model_data_url"]
        local_tar = mlflow.artifacts.download_artifacts(artifact_uri=tar_s3_uri)
        assert tarfile.is_tarfile(local_tar)

        with tempfile.TemporaryDirectory() as extract_dir:
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(extract_dir)

            extracted = os.listdir(extract_dir)
            assert "MLmodel" in extracted
            assert "data" in extracted  # pytorch stores model in data/
            assert "conda.yaml" in extracted

            # Verify MLmodel metadata inside tar
            with open(os.path.join(extract_dir, "MLmodel")) as f:
                mlmodel = yaml.safe_load(f)
            assert mlmodel["metadata"]["sagemaker.inference_image_uri"] == DUMMY_IMAGE_PYTORCH
            assert mlmodel["metadata"]["sagemaker.compressed"] == "True"


class TestDefaultImageUri:
    def test_flavor_default(self):
        sagemaker_mlflow.set_default_inference_image_uri(DUMMY_IMAGE_SKLEARN, flavor="sklearn")

        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-default-by-flavor",
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_SKLEARN

    def test_global_default(self):
        sagemaker_mlflow.set_default_inference_image_uri(DUMMY_IMAGE_URI)

        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-default-global",
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_URI

    def test_experiment_default(self):
        sagemaker_mlflow.set_default_inference_image_uri(
            DUMMY_IMAGE_URI, experiment="sagemaker-mlflow-log-model-integ-test"
        )

        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-default-by-experiment",
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_URI

    def test_explicit_overrides_default(self):
        sagemaker_mlflow.set_default_inference_image_uri("should-not-use")

        with mlflow.start_run():
            model = _train_sklearn_model()
            model_info = sagemaker_mlflow.log_model(
                "sklearn",
                sk_model=model,
                artifact_path="test-default-override",
                inference_image_uri=DUMMY_IMAGE_SKLEARN,
            )

        client = mlflow.MlflowClient()
        logged_model = client.get_logged_model(model_info.model_id)
        assert logged_model.tags["sagemaker.inference_image_uri"] == DUMMY_IMAGE_SKLEARN
