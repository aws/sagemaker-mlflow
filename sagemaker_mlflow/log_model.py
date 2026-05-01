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

import mlflow

_default_image_registry = {}  # (flavor|None, experiment|None) -> uri


def set_default_inference_image_uri(uri, flavor=None, experiment=None):
    """Set a default inference image URI for log_model resolution.

    Resolution priority (highest to lowest):
        1. inference_image_uri passed directly to log_model()
        2. (flavor + experiment) match
        3. (experiment only) match
        4. (flavor only) match
        5. Global default (no flavor, no experiment)

    Args:
        uri: ECR image URI.
        flavor: Optional flavor name (e.g. "sklearn", "transformers").
        experiment: Optional experiment name.
    """
    _default_image_registry[(flavor, experiment)] = uri


def _resolve_image_uri(inference_image_uri, flavor):
    if inference_image_uri:
        return inference_image_uri

    experiment = None
    run = mlflow.active_run()
    if run:
        exp = mlflow.get_experiment(run.info.experiment_id)
        experiment = exp.name

    for key in [(flavor, experiment), (None, experiment), (flavor, None), (None, None)]:
        if key in _default_image_registry:
            return _default_image_registry[key]

    return None


def log_model(flavor, artifact_path=None, inference_image_uri=None, log_compressed_model=False, **kwargs):
    """Log a model with SageMaker deployment metadata.

    Wraps mlflow.<flavor>.log_model(), adding inference_image_uri and compression
    metadata as LoggedModel tags and MLmodel metadata for server-side consumption.

    Args:
        flavor: MLflow flavor name (e.g. "sklearn", "pytorch", "transformers").
        artifact_path: Run-relative artifact path (passed to upstream log_model).
        inference_image_uri: ECR image URI for the inference container. If None,
            resolved from defaults set via set_default_inference_image_uri().
            If still None, no inference metadata is recorded.
        log_compressed_model: If True, model artifacts are packaged as model.tar.gz
            (ModelDataUrl). If False (default), uses uncompressed S3 prefix
            (ModelDataSource with CompressionType: None).
        **kwargs: All other arguments forwarded to mlflow.<flavor>.log_model().

    Returns:
        ModelInfo from the upstream log_model call.
    """
    image_uri = _resolve_image_uri(inference_image_uri, flavor)

    # Inject SageMaker metadata into MLmodel YAML via metadata kwarg
    sagemaker_metadata = {}
    if image_uri:
        sagemaker_metadata["sagemaker.inference_image_uri"] = image_uri
    sagemaker_metadata["sagemaker.inference_flavor"] = flavor
    sagemaker_metadata["sagemaker.compressed"] = str(log_compressed_model)

    if sagemaker_metadata:
        user_metadata = kwargs.pop("metadata", None) or {}
        user_metadata.update(sagemaker_metadata)
        kwargs["metadata"] = user_metadata

    # Delegate to upstream flavor log_model
    flavor_module = getattr(mlflow, flavor)
    model_info = flavor_module.log_model(artifact_path=artifact_path, **kwargs)

    # Set logged model tags for fast server-side access (no S3 round-trip)
    tags = {
        "sagemaker.inference_flavor": flavor,
        "sagemaker.compressed": str(log_compressed_model),
    }
    if image_uri:
        tags["sagemaker.inference_image_uri"] = image_uri

    if log_compressed_model:
        tar_gz_uri = _compress_and_log_model_artifacts(model_info, flavor_module, kwargs)
        tags["sagemaker.model_data_url"] = tar_gz_uri
    else:
        tags["sagemaker.model_data_url"] = model_info.model_uri

    client = mlflow.MlflowClient()
    client.set_logged_model_tags(model_info.model_id, tags)

    return model_info


def _compress_and_log_model_artifacts(model_info, flavor_module, kwargs):
    """Save model locally, create model.tar.gz, upload alongside originals.

    Returns the S3 URI of the uploaded tar.gz.
    """
    import inspect
    import os
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_dir = os.path.join(tmp_dir, "model")

        # Filter kwargs to only save_model-compatible args
        save_params = inspect.signature(flavor_module.save_model).parameters
        save_kwargs = {k: v for k, v in kwargs.items() if k in save_params}

        # Serialize model to local dir (no S3 round-trip)
        flavor_module.save_model(path=local_model_dir, **save_kwargs)

        # Create model.tar.gz
        tar_path = os.path.join(tmp_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for item in os.listdir(local_model_dir):
                tar.add(os.path.join(local_model_dir, item), arcname=item)

        # Upload tar.gz as an additional model artifact
        client = mlflow.MlflowClient()
        client.log_model_artifact(model_info.model_id, tar_path)

    # Return S3 URI of the tar.gz
    logged_model = client.get_logged_model(model_info.model_id)
    return f"{logged_model.artifact_location}/model.tar.gz"
