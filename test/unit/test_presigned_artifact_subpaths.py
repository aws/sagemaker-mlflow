from unittest.mock import Mock

import pytest

from sagemaker_mlflow.s3_presigned_artifact_repo import S3PresignedArtifactRepository


@pytest.mark.parametrize("owner", ["123/run-id", "123/models/m-model-id"])
@pytest.mark.parametrize("artifact_path", [None, "weights"])
def test_presigned_upload_keeps_repository_subpath(monkeypatch, tmp_path, owner, artifact_path):
    monkeypatch.setenv("SAGEMAKER_PRESIGNED_URL_UPLOAD_ENABLED", "true")
    repo = S3PresignedArtifactRepository(
        f"s3://bucket/{owner}/artifacts/subdir",
        tracking_uri="arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/test",
    )
    response = Mock(ok=True)
    response.json.return_value = {"presigned_url": "https://example.com/upload"}
    request_url = Mock(return_value=response)
    monkeypatch.setattr(repo, "_request_presigned_url", request_url)
    monkeypatch.setattr("sagemaker_mlflow.s3_presigned_artifact_repo.cloud_storage_http_request", Mock())
    local_file = tmp_path / "model.bin"
    local_file.write_bytes(b"model")

    repo.log_artifact(str(local_file), artifact_path)

    expected = "subdir/weights/model.bin" if artifact_path else "subdir/model.bin"
    assert request_url.call_args.args[1] == expected
