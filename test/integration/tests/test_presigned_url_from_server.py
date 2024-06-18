import os

import sagemaker_mlflow
import mlflow
import pytest
import requests

import sagemaker_mlflow.presigned_url


""" This test makes sure that getting the presigned url works.
"""
class TestPresignedUrl:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        # TODO: Verify that tracking server is created
        mlflow.set_tracking_uri(tracking_server)

    # Restrict to local environments for now, remove after GA.
    @pytest.mark.skipif(os.environ.get("CODEBUILD_BUILD_ARN", "") != "", reason="Codebuild might not have the right API shape")
    def test_presigned_url(self, setup):
        presigned_url = sagemaker_mlflow.presigned_url.get_presigned_url()
        response = requests.get(url=presigned_url)
        assert response.status_code == 200
