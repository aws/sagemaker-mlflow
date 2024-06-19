import logging
import os

import mlflow
import pytest

from utils.boto_utils import get_file_data_from_s3
from utils.random_utils import generate_uuid, generate_random_float

""" Test that artifacts are being persisted correctly with an 
    Amazon S3 based artifact store. This test is only for artifact
    stores in which the client goes through S3.
"""
class TestArtifactLogging:

    @pytest.fixture(scope="class")
    def setup(self, tracking_server):
        # TODO: Verify that tracking server is created
        mlflow.set_tracking_uri(tracking_server)

    def test_log_artifact(self, setup):
        # Create a random file
        file_name = f"{generate_uuid(20)}.txt"
        file_contents = "".join([generate_uuid(40) for i in range(5)])
        logging.info(f"Writing {file_contents} to {file_name}")
        with open(file_name, "wb") as f:
            f.write(bytes(file_contents.encode("utf-8")))

        current_run = None
        with mlflow.start_run():
            mlflow.log_artifact(file_name)
            current_run = mlflow.active_run()

        assert current_run is not None

        run_artifact_location = current_run.info.artifact_uri
        split_location = run_artifact_location.replace("s3://", "").split("/")
        split_location.append(file_name)
        bucket = split_location[0]
        prefix = "/".join(split_location[1:])
        data = get_file_data_from_s3(bucket, prefix)
        data = data.read().decode("ascii")
        assert data == file_contents

        os.remove(file_name)
