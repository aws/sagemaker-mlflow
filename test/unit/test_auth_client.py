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

import unittest
from unittest import TestCase, mock

from sagemaker_mlflow.exceptions import MlflowSageMakerException
from sagemaker_mlflow.auth_client import (
    SageMakerMlflowAuthClient,
    _is_role_arn,
    _validate_role_arn,
)

TEST_APP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-XXXXXXXXXXXX"
TEST_ROLE_ARN = "arn:aws:iam::123456789012:role/AliceDSRole"
TEST_USER_ARN = "arn:aws:iam::123456789012:user/alice"

_MODULE = "sagemaker_mlflow.auth_client"


def _mock_response(payload, status_code=200):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.content = b"x" if payload else b""
    resp.json.return_value = payload
    return resp


def _client(tracking_uri=TEST_APP_ARN):
    """Construct the client without touching mlflow.get_tracking_uri."""
    return SageMakerMlflowAuthClient(tracking_uri=tracking_uri)


class IsRoleArnTest(TestCase):
    def test_accepts_role_arns(self):
        self.assertTrue(_is_role_arn(TEST_ROLE_ARN))

    def test_accepts_non_commercial_partitions(self):
        self.assertTrue(_is_role_arn("arn:aws-us-gov:iam::123456789012:role/GovRole"))

    def test_rejects_user_arns(self):
        # The client is role-only even though the server accepts user ARNs too.
        self.assertFalse(_is_role_arn(TEST_USER_ARN))

    def test_rejects_non_arn_and_non_role(self):
        for bad in [
            "alice",
            "",
            None,
            123,
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-x",
            "arn:aws:iam::123:role/ShortAccount",  # account id not 12 digits
            "arn:aws:iam::123456789012:group/devs",  # not a role
            "arn:aws:iam::123456789012:user/alice",  # user, not role
        ]:
            self.assertFalse(_is_role_arn(bad), bad)


class ValidateRoleArnTest(TestCase):
    def test_rejects_non_role_arn(self):
        with self.assertRaises(MlflowSageMakerException) as ctx:
            _validate_role_arn("not-an-arn")
        self.assertIn("IAM role ARN", str(ctx.exception))

    def test_rejects_role_arn_with_path(self):
        with self.assertRaises(MlflowSageMakerException) as ctx:
            _validate_role_arn("arn:aws:iam::123456789012:role/team/AliceDSRole")
        # Error surfaces the pathless form the caller should use instead.
        self.assertIn("arn:aws:iam::123456789012:role/AliceDSRole", str(ctx.exception))

    def test_accepts_pathless_role_arn(self):
        self.assertIsNone(_validate_role_arn(TEST_ROLE_ARN))


class ConstructionTest(TestCase):
    def test_uses_explicit_tracking_uri(self):
        self.assertEqual(_client().tracking_uri, TEST_APP_ARN)

    def test_falls_back_to_active_tracking_uri(self):
        with mock.patch(f"{_MODULE}.mlflow.get_tracking_uri", return_value=TEST_APP_ARN):
            client = SageMakerMlflowAuthClient()
        self.assertEqual(client.tracking_uri, TEST_APP_ARN)

    def test_no_tracking_uri_raises(self):
        with mock.patch(f"{_MODULE}.mlflow.get_tracking_uri", return_value=""):
            with self.assertRaises(MlflowSageMakerException) as ctx:
                SageMakerMlflowAuthClient()
        self.assertIn("No tracking URI", str(ctx.exception))


class CreateUserTest(TestCase):
    def test_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException) as ctx:
            _client().create_user(TEST_USER_ARN)
        self.assertIn("IAM role ARN", str(ctx.exception))

    def test_posts_empty_password_with_signed_creds(self):
        created = {"id": 3, "username": TEST_ROLE_ARN, "is_admin": False}
        sentinel_creds = object()
        with mock.patch(f"{_MODULE}.get_host_creds", return_value=sentinel_creds) as mock_creds, mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_mock_response({"user": created})
        ) as mock_http, mock.patch(
            "mlflow.utils.rest_utils.verify_rest_response",
            side_effect=lambda resp, endpoint, expected_status=200: resp,
        ):
            user = _client().create_user(TEST_ROLE_ARN)

        self.assertEqual(user.username, TEST_ROLE_ARN)
        mock_creds.assert_called_once_with(TEST_APP_ARN)
        args, kwargs = mock_http.call_args
        host_creds, endpoint, method = args
        self.assertIs(host_creds, sentinel_creds)
        self.assertEqual(method, "POST")
        self.assertEqual(kwargs["json"], {"username": TEST_ROLE_ARN, "password": ""})


class UpdateUserAdminTest(TestCase):
    def test_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException) as ctx:
            _client().update_user_admin(TEST_USER_ARN, is_admin=True)
        self.assertIn("IAM role ARN", str(ctx.exception))

    def test_patches_admin_flag_without_password(self):
        with mock.patch(f"{_MODULE}.get_host_creds", return_value=object()), mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_mock_response({}, status_code=200)
        ) as mock_http, mock.patch(
            "mlflow.utils.rest_utils.verify_rest_response",
            side_effect=lambda resp, endpoint, expected_status=200: resp,
        ):
            _client().update_user_admin(TEST_ROLE_ARN, is_admin=True)

        _, kwargs = mock_http.call_args
        self.assertEqual(kwargs["json"], {"username": TEST_ROLE_ARN, "is_admin": True})
        self.assertNotIn("password", kwargs["json"])


class UpdateUserPasswordBlockedTest(TestCase):
    def test_always_raises(self):
        with self.assertRaises(MlflowSageMakerException) as ctx:
            _client().update_user_password(TEST_ROLE_ARN, "pw")
        self.assertIn("password cannot be set", str(ctx.exception))


class GetDeleteUserValidationTest(TestCase):
    def test_get_user_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException):
            _client().get_user(TEST_USER_ARN)

    def test_delete_user_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException):
            _client().delete_user(TEST_USER_ARN)


class RoleAssignmentValidationTest(TestCase):
    def test_assign_role_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException):
            _client().assign_role(TEST_USER_ARN, 1)

    def test_grant_user_permission_rejects_user_arn(self):
        with self.assertRaises(MlflowSageMakerException):
            _client().grant_user_permission(TEST_USER_ARN, "experiment", "1", "READ")


class InheritedRoleSurfaceTest(TestCase):
    def test_create_role_is_inherited_and_signed(self):
        # create_role carries no username, so it is used unchanged from the base
        # class; confirm it routes through the overridden signed _request.
        role = {"id": 7, "name": "viewers", "workspace": "default"}
        with mock.patch(f"{_MODULE}.get_host_creds", return_value=object()) as mock_creds, mock.patch(
            "mlflow.utils.rest_utils.http_request", return_value=_mock_response({"role": role})
        ) as mock_http, mock.patch(
            "mlflow.utils.rest_utils.verify_rest_response",
            side_effect=lambda resp, endpoint, expected_status=200: resp,
        ):
            result = _client().create_role(workspace="default", name="viewers")

        self.assertEqual(result.name, "viewers")
        mock_creds.assert_called_once_with(TEST_APP_ARN)
        _, kwargs = mock_http.call_args
        self.assertEqual(kwargs["json"], {"workspace": "default", "name": "viewers"})


class SkinnyDegradationTest(TestCase):
    def test_helpful_error_when_auth_server_missing(self):
        import sagemaker_mlflow.auth_client as ac

        with mock.patch.object(ac, "_AUTH_SERVER_IMPORT_ERROR", ImportError("no module")):
            with self.assertRaises(MlflowSageMakerException) as ctx:
                SageMakerMlflowAuthClient(tracking_uri=TEST_APP_ARN)
        self.assertIn("sagemaker-mlflow[full]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
