# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

from unittest.mock import MagicMock, patch
import pytest

from sagemaker_mlflow.log_model import (
    log_model,
    set_default_inference_image_uri,
    _resolve_image_uri,
    _default_image_registry,
)


@pytest.fixture(autouse=True)
def clear_registry():
    _default_image_registry.clear()
    yield
    _default_image_registry.clear()


class TestResolveImageUri:
    def test_direct_uri_takes_priority(self):
        set_default_inference_image_uri("default-uri", flavor="sklearn")
        assert _resolve_image_uri("direct-uri", "sklearn") == "direct-uri"

    def test_flavor_and_experiment_match(self):
        set_default_inference_image_uri("fe-uri", flavor="sklearn", experiment="exp1")
        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = MagicMock(info=MagicMock(experiment_id="123"))
            with patch("sagemaker_mlflow.log_model.mlflow.get_experiment") as mock_exp:
                exp_mock = MagicMock()
                exp_mock.name = "exp1"
                mock_exp.return_value = exp_mock
                assert _resolve_image_uri(None, "sklearn") == "fe-uri"

    def test_experiment_only_match(self):
        set_default_inference_image_uri("exp-uri", experiment="exp1")
        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = MagicMock(info=MagicMock(experiment_id="123"))
            with patch("sagemaker_mlflow.log_model.mlflow.get_experiment") as mock_exp:
                exp_mock = MagicMock()
                exp_mock.name = "exp1"
                mock_exp.return_value = exp_mock
                assert _resolve_image_uri(None, "pytorch") == "exp-uri"

    def test_flavor_only_match(self):
        set_default_inference_image_uri("flavor-uri", flavor="sklearn")
        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = None
            assert _resolve_image_uri(None, "sklearn") == "flavor-uri"

    def test_global_default(self):
        set_default_inference_image_uri("global-uri")
        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = None
            assert _resolve_image_uri(None, "transformers") == "global-uri"

    def test_returns_none_when_nothing_set(self):
        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = None
            assert _resolve_image_uri(None, "sklearn") is None

    def test_resolution_priority_order(self):
        set_default_inference_image_uri("global-uri")
        set_default_inference_image_uri("flavor-uri", flavor="sklearn")
        set_default_inference_image_uri("exp-uri", experiment="exp1")
        set_default_inference_image_uri("fe-uri", flavor="sklearn", experiment="exp1")

        with patch("sagemaker_mlflow.log_model.mlflow.active_run") as mock_run:
            mock_run.return_value = MagicMock(info=MagicMock(experiment_id="123"))
            with patch("sagemaker_mlflow.log_model.mlflow.get_experiment") as mock_exp:
                exp_mock = MagicMock()
                exp_mock.name = "exp1"
                mock_exp.return_value = exp_mock
                # Most specific wins
                assert _resolve_image_uri(None, "sklearn") == "fe-uri"
                # No flavor+experiment match, falls to experiment-only
                assert _resolve_image_uri(None, "pytorch") == "exp-uri"


class TestLogModel:
    @patch("mlflow.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_with_image_uri_uncompressed(self, mock_log_model, mock_client_cls):
        mock_model_info = MagicMock()
        mock_model_info.model_id = "model-123"
        mock_model_info.model_uri = "runs:/abc/model"
        mock_log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        result = log_model(
            "sklearn",
            sk_model=MagicMock(),
            artifact_path="model",
            inference_image_uri="123.dkr.ecr.us-west-2.amazonaws.com/my-img:v1",
        )

        assert result == mock_model_info
        mock_log_model.assert_called_once()
        call_kwargs = mock_log_model.call_args[1]
        assert call_kwargs["metadata"]["sagemaker.inference_image_uri"] == (
            "123.dkr.ecr.us-west-2.amazonaws.com/my-img:v1"
        )
        assert call_kwargs["metadata"]["sagemaker.inference_flavor"] == "sklearn"

        mock_client.set_logged_model_tags.assert_called_once_with(
            "model-123",
            {
                "sagemaker.inference_flavor": "sklearn",
                "sagemaker.inference_image_uri": "123.dkr.ecr.us-west-2.amazonaws.com/my-img:v1",
                "sagemaker.model_data_url": "runs:/abc/model",
                "sagemaker.compressed": "False",
            },
        )

    @patch("mlflow.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_without_image_uri(self, mock_log_model, mock_client_cls):
        mock_model_info = MagicMock()
        mock_model_info.model_id = "model-456"
        mock_model_info.model_uri = "runs:/def/model"
        mock_log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        with patch("sagemaker_mlflow.log_model.mlflow.active_run", return_value=None):
            log_model("sklearn", sk_model=MagicMock(), artifact_path="model")

        mock_client.set_logged_model_tags.assert_called_once_with(
            "model-456",
            {
                "sagemaker.inference_flavor": "sklearn",
                "sagemaker.model_data_url": "runs:/def/model",
                "sagemaker.compressed": "False",
            },
        )

    @patch("mlflow.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_preserves_user_metadata(self, mock_log_model, mock_client_cls):
        mock_model_info = MagicMock()
        mock_model_info.model_id = "model-789"
        mock_model_info.model_uri = "runs:/ghi/model"
        mock_log_model.return_value = mock_model_info
        mock_client_cls.return_value = MagicMock()

        log_model(
            "sklearn",
            sk_model=MagicMock(),
            artifact_path="model",
            inference_image_uri="img:v1",
            metadata={"my_key": "my_value"},
        )

        call_kwargs = mock_log_model.call_args[1]
        assert call_kwargs["metadata"]["my_key"] == "my_value"
        assert call_kwargs["metadata"]["sagemaker.inference_image_uri"] == "img:v1"

    @patch("mlflow.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_compress_true_creates_tar_and_uploads(self, mock_log_model, mock_client_cls):
        import os

        mock_model_info = MagicMock()
        mock_model_info.model_id = "model-000"
        mock_model_info.model_uri = "models:/model-000"
        mock_log_model.return_value = mock_model_info

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_logged_model = MagicMock()
        mock_logged_model.artifact_location = "s3://bucket/path/to/artifacts"
        mock_client.get_logged_model.return_value = mock_logged_model

        # Wrap real save_model to create files but track the call
        import mlflow.sklearn as real_sklearn

        original_save = real_sklearn.save_model

        def fake_save_model(*args, **kwargs):
            path = kwargs.get("path") or args[1]
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "MLmodel"), "w") as f:
                f.write("flavor: sklearn\n")
            with open(os.path.join(path, "model.pkl"), "wb") as f:
                f.write(b"fake")

        with patch.object(real_sklearn, "save_model", side_effect=fake_save_model, wraps=None) as mock_save:
            # Keep the real signature accessible for inspect
            mock_save.__wrapped__ = original_save
            log_model(
                "sklearn",
                sk_model=MagicMock(),
                artifact_path="model",
                log_compressed_model=True,
            )

        # Verify tar.gz was uploaded
        mock_client.log_model_artifact.assert_called_once()
        uploaded_path = mock_client.log_model_artifact.call_args[0][1]
        assert uploaded_path.endswith("model.tar.gz")

        # Verify tags include the S3 tar.gz URI
        tags = mock_client.set_logged_model_tags.call_args[0][1]
        assert tags["sagemaker.model_data_url"] == "s3://bucket/path/to/artifacts/model.tar.gz"
        assert tags["sagemaker.compressed"] == "True"

    @patch("mlflow.MlflowClient")
    @patch("mlflow.sklearn.log_model")
    def test_forwards_all_kwargs(self, mock_log_model, mock_client_cls):
        mock_model_info = MagicMock()
        mock_model_info.model_id = "m1"
        mock_model_info.model_uri = "runs:/x/model"
        mock_log_model.return_value = mock_model_info
        mock_client_cls.return_value = MagicMock()

        log_model(
            "sklearn",
            sk_model="my_model",
            artifact_path="model",
            pip_requirements=["scikit-learn==1.4"],
            signature="my_sig",
        )

        call_kwargs = mock_log_model.call_args[1]
        assert call_kwargs["sk_model"] == "my_model"
        assert call_kwargs["pip_requirements"] == ["scikit-learn==1.4"]
        assert call_kwargs["signature"] == "my_sig"
