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
import os

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from hashlib import sha256
import functools
from sagemaker_mlflow.credential_cache import CredentialCache

PAYLOAD_BUFFER = 1024 * 1024
# Hardcode SHA256 hash for empty string to reduce latency for requests without a body
EMPTY_SHA256_HASH = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Default TTL for cached credentials (55 minutes - safe margin before AWS 1-hour expiration)
DEFAULT_CREDENTIAL_TTL_SECONDS = 3300


class AuthBoto(AuthBase):
    # Class-level credential cache shared across instances
    _credential_cache = CredentialCache()

    def __init__(self, region: str, service_name: str, assume_role_arn: Optional[str] = None):
        """
        Constructor for Authorization Mechanism
        :param region: AWS region (e.g., us-west-2)
        :param service_name: AWS service name for signing
        :param assume_role_arn: ARN of the role to assume (optional)
        """
        self._assume_role_arn = assume_role_arn
        self.region = region

        if assume_role_arn is not None:
            # Use cached or fresh assumed role credentials
            credentials = self._get_cached_credentials(assume_role_arn)
            self.creds = boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            ).get_credentials()
        else:
            # Use current session credentials
            session = boto3.Session()
            self.creds = session.get_credentials()

        self.sigv4 = SigV4Auth(self.creds, service_name, self.region)

    def _get_cached_credentials(self, assume_role_arn: str) -> dict:
        """
        Get cached credentials or fetch new ones via STS assume role.

        :param assume_role_arn: ARN of the role to assume
        :return: AWS credentials dictionary
        """
        # Try to get credentials from cache first
        cached_credentials = self._credential_cache.get_credentials(assume_role_arn)
        if cached_credentials is not None:
            return cached_credentials

        # Cache miss - fetch new credentials via STS
        session = boto3.Session()
        sts_client = session.client("sts")
        assumed_role_object = sts_client.assume_role(RoleArn=assume_role_arn, RoleSessionName="AuthBotoSagemakerMlFlow")
        credentials = assumed_role_object["Credentials"]

        # Get TTL from environment variable with default fallback
        ttl_seconds = int(os.environ.get("SAGEMAKER_MLFLOW_ASSUME_ROLE_TTL_SECONDS", DEFAULT_CREDENTIAL_TTL_SECONDS))

        # Validate TTL is within reasonable bounds (5 minutes to 1 hour)
        ttl_seconds = max(300, min(ttl_seconds, 3600))

        # Cache the credentials
        self._credential_cache.set_credentials(assume_role_arn, credentials, ttl_seconds)

        return credentials

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

        body_bytes = request_body or b""
        if isinstance(body_bytes, str):
            body_bytes = body_bytes.encode("utf-8")
        headers["X-Amz-Content-SHA256"] = self.get_request_body_header(body_bytes)

        # SageMaker Mlflow strips out this header before auth.
        # But boto signs every header even its its uppercase or lower cased.
        if "Connection" in headers:
            connection_header = headers["Connection"]
            del headers["Connection"]

        # Mlflow encodes spaces as +, Auth prefers %20
        if method == "GET" or method == "DELETE":
            url = (url or "").replace("+", "%20")

        # Creating a new request with the SigV4 signed headers.
        aws_request = AWSRequest(method=method, url=url, data=r.body, headers=headers)
        self.sigv4.add_auth(aws_request)

        # Adding back in the connection header.
        final_headers = aws_request.headers
        final_headers["Connection"] = connection_header
        final_request = AWSRequest(method=method, url=url, data=r.body, headers=final_headers)

        return final_request.prepare()

    def get_request_body_header(self, request_body: bytes):
        """Stripped down version of the botocore method.
        :param request_body: request body
        :return: hex_checksum
        """
        has_seek = hasattr(request_body, "seek")
        has_tell = hasattr(request_body, "tell")
        has_read = hasattr(request_body, "read")
        if request_body and has_seek and has_tell and has_read:
            # Type narrowing: we know request_body has file-like methods
            file_obj = request_body  # type: ignore[attr-defined]
            position = file_obj.tell()  # type: ignore[attr-defined]
            read_chunksize = functools.partial(file_obj.read, PAYLOAD_BUFFER)  # type: ignore[attr-defined]
            checksum = sha256()
            for chunk in iter(read_chunksize, b""):
                checksum.update(chunk)
            hex_checksum = checksum.hexdigest()
            file_obj.seek(position)  # type: ignore[attr-defined]
            return hex_checksum
        elif request_body:
            # The request serialization has ensured that
            # request.body is a bytes() type.
            return sha256(request_body).hexdigest()
        else:
            return EMPTY_SHA256_HASH
