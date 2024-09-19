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

from typing import Optional
import boto3
from requests.auth import AuthBase
from requests.models import PreparedRequest

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from hashlib import sha256
import functools

SERVICE_NAME = "sagemaker-mlflow"
PAYLOAD_BUFFER = 1024 * 1024
# Hardcode SHA256 hash for empty string to reduce latency for requests without a body
EMPTY_SHA256_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class AuthBoto(AuthBase):

    def __init__(self, region: str, assume_role_arn: Optional[str] = None):
        """
        Constructor for Authorization Mechanism
        :param region: AWS region (e.g., us-west-2)
        :param assume_role_arn: ARN of the role to assume (optional)
        """
        session = boto3.Session()

        self._assume_role_arn = None

        if assume_role_arn is not None:
            # Use STS to assume the provided role
            self._assume_role_arn = assume_role_arn
            sts_client = session.client("sts")
            assumed_role_object = sts_client.assume_role(
                RoleArn=assume_role_arn,
                RoleSessionName="AuthBotoSagemakerMlFlow"
            )
            credentials = assumed_role_object['Credentials']
            self.creds = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            ).get_credentials()
        else:
            # Use current session credentials
            self.creds = session.get_credentials()

        self.region = region
        self.sigv4 = SigV4Auth(self.creds, SERVICE_NAME, self.region)

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        """Method to return the prepared request
        :param r: PreparedRequest Base mlflow request
        :return: PreparedRequest Request with SigV4 signed headers
        """

        url = r.url
        method = r.method
        headers = r.headers
        request_body = r.body
        connection_header = headers["Connection"]

        headers["X-Amz-Content-SHA256"] = self.get_request_body_header(request_body)

        # SageMaker Mlflow strips out this header before auth.
        # But boto signs every header even its its uppercase or lower cased.
        if "Connection" in headers:
            connection_header = headers["Connection"]
            del headers["Connection"]

        # Mlflow encodes spaces as +, Auth prefers %20
        if method == "GET" or method == "DELETE":
            url = url.replace("+", "%20")

        # Creating a new request with the SigV4 signed headers.
        aws_request = AWSRequest(method=method, url=url, data=r.body, headers=headers)
        self.sigv4.add_auth(aws_request)

        # Adding back in the connection header.
        final_headers = aws_request.headers
        final_headers["Connection"] = connection_header
        final_request = AWSRequest(
            method=method, url=url, data=r.body, headers=final_headers
        )

        return final_request.prepare()

    def get_request_body_header(self, request_body: bytes):
        """Stripped down version of the botocore method.
        :param request_body: request body
        :return: hex_checksum
        """
        if request_body and hasattr(request_body, "seek"):
            position = request_body.tell()
            read_chunksize = functools.partial(request_body.read, PAYLOAD_BUFFER)
            checksum = sha256()
            for chunk in iter(read_chunksize, b""):
                checksum.update(chunk)
            hex_checksum = checksum.hexdigest()
            request_body.seek(position)
            return hex_checksum
        elif request_body:
            # The request serialization has ensured that
            # request.body is a bytes() type.
            return sha256(request_body).hexdigest()
        else:
            return EMPTY_SHA256_HASH
