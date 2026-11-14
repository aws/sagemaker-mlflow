import boto3
import pytest
import requests

from sagemaker_mlflow.auth import AuthBoto


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_auth_preserves_plus_in_path_and_normalizes_query_spaces(method):
    session = boto3.Session(aws_access_key_id="test-key", aws_secret_access_key="test-secret")
    auth = AuthBoto("us-west-2", "sagemaker-mlflow", boto3_session=session)
    request = requests.Request(
        method,
        "https://example.com/models/a+b?filter=name+eq+%27a%2Bb%27",
        headers={"Connection": "keep-alive"},
        auth=auth,
    ).prepare()

    assert request.url == "https://example.com/models/a+b?filter=name%20eq%20%27a%2Bb%27"
    assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")
