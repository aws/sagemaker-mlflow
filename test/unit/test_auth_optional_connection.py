import boto3
import pytest
import requests

from sagemaker_mlflow.auth import AuthBoto


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_auth_without_connection_header(method):
    session = boto3.Session(aws_access_key_id="test-key", aws_secret_access_key="test-secret")
    auth = AuthBoto("us-west-2", "sagemaker-mlflow", boto3_session=session)
    request = requests.Request(method, "https://example.com/api/test", auth=auth).prepare()

    assert "Connection" not in request.headers
    assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")
