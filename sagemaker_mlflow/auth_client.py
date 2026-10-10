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

"""Admin auth client for SageMaker MLflow apps with access control enabled.

This module exposes :class:`SageMakerMlflowAuthClient`, a thin specialization of
MLflow's :class:`mlflow.server.auth.client.AuthServiceClient` for SageMaker
MLflow fine-grained access control (FGAC). It inherits the full RBAC surface
(roles, role permissions, role assignments, per-user permission grants) and
differs from the upstream client in three ways:

* **IAM identity, not passwords.** A SageMaker MLflow app bridges AWS IAM to
  MLflow's RBAC user model: the IAM *role* ARN is the MLflow username and
  callers authenticate with SigV4, never a password. ``create_user`` takes only
  a ``role_arn`` and sends an empty password; ``update_user_password`` is
  unsupported (the server refuses it); and every method that names a user
  validates that the name is an IAM role ARN before the request leaves the
  client.
* **Role ARNs only.** The server accepts role *or* user ARNs, but SageMaker
  MLflow FGAC registers assumed-role identities, so this client restricts to
  roles and fails fast on anything else.
* **Tracking URI resolution.** Like the other plugin helpers (for example
  ``get_presigned_url``), the target app is taken from the active MLflow
  tracking URI (``mlflow.get_tracking_uri()``); requests are signed via the
  plugin's ``arn`` auth provider through :func:`get_host_creds`.

Dependency note: the client extends ``mlflow.server.auth.client``, which ships
with the full ``mlflow`` auth-server package and is **not** present in
``mlflow-skinny`` (this package's default dependency). Importing
``sagemaker_mlflow`` stays safe under skinny; constructing
:class:`SageMakerMlflowAuthClient` without the auth server installed raises a
clear :class:`MlflowSageMakerException` pointing at the ``[full]`` extra.
"""

import logging
import re
from typing import Optional

import mlflow

from sagemaker_mlflow.exceptions import MlflowSageMakerException
from sagemaker_mlflow.host_creds import get_host_creds

logger = logging.getLogger(__name__)

# The MLflow auth server package (client + entities + routes) is only in full
# mlflow, not mlflow-skinny. Import it lazily so `import sagemaker_mlflow` keeps
# working under skinny; the helpful error is raised at construction time.
#
# TODO: this import runs at MODULE LOAD, which is not actually lazy. In some full
# mlflow installs (observed with mlflow 3.13.0) importing it here triggers a
# circular import inside mlflow itself
# (mlflow.server.__init__ -> handlers -> gateway.budget -> store ->
# mlflow.genai.datasets -> mlflow.tracking.get_tracking_uri) when mlflow.server is
# pulled in before mlflow.tracking has finished initializing. The ImportError is
# then miscaptured below as "auth-server missing", surfacing the misleading
# "install sagemaker-mlflow[full]" error even though [full] IS installed.
# Fix: defer this import into _require_auth_server() (first-use), binding
# _AuthServiceClient at construction time instead of module-import time, so a
# partially-initialized mlflow at import time cannot wedge the subclass. Keep the
# skinny-vs-full ImportError handling, just move it to first use. Until then,
# consumers hitting the cycle can warm up mlflow first:
#     import mlflow, mlflow.tracking, mlflow.store.tracking.dbmodels.models
#     import mlflow.server.auth.client
# (The Brazil build / unit tests import in a clean order and are unaffected.)
try:
    from mlflow.server.auth.client import AuthServiceClient as _AuthServiceClient

    _AUTH_SERVER_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only under skinny
    _AuthServiceClient = object  # placeholder base so the subclass can be defined
    _AUTH_SERVER_IMPORT_ERROR = exc

_AUTH_SERVER_MISSING_MESSAGE = (
    "SageMakerMlflowAuthClient requires MLflow's auth-server package "
    "(mlflow.server.auth), which is not included in mlflow-skinny. Install the "
    "full dependency set with: pip install 'sagemaker-mlflow[full]'"
)

# Mirror of the server-side principal-ARN validation
# (awsmlflow.auth.identity_decryptor._PRINCIPAL_ARN_PATTERN), narrowed to IAM
# *role* ARNs only. The lookahead caps the ARN at IAM's max of a 512-char path
# plus a 64-char name; re.ASCII blocks homoglyphs.
_ROLE_ARN_PATTERN = re.compile(
    r"\A(?=.{1,640}\Z)" r"arn:aws(?:-[a-z]+)*:iam::[0-9]{12}:role" r"(?:/[\w+=,.@-]{1,64})+\Z",
    re.ASCII,
)

# The server ignores any password on users/create and keys the user on the IAM
# principal ARN. We send an empty string to make the "no password" contract
# explicit on the wire.
_EMPTY_PASSWORD = ""


def _is_role_arn(value: str) -> bool:
    """True when value is a well-formed IAM role ARN, path included."""
    return isinstance(value, str) and _ROLE_ARN_PATTERN.match(value) is not None


def _validate_role_arn(role_arn: str) -> None:
    """Reject anything that is not a usable IAM role ARN before the request.

    Raises:
        MlflowSageMakerException: if ``role_arn`` is not a well-formed IAM role
            ARN, or is a role ARN that carries a path. APS forwards role ARNs
            without their path, so a pathed registration could never be matched
            at request time — reject it here with the pathless form.
    """
    if not _is_role_arn(role_arn):
        raise MlflowSageMakerException(f"role_arn must be an IAM role ARN, got: {role_arn!r}")
    account_prefix, sep, role_path = role_arn.partition(":role/")
    if sep and "/" in role_path:
        pathless = f"{account_prefix}:role/{role_path.rsplit('/', 1)[1]}"
        raise MlflowSageMakerException(f"Role ARNs must not include a path: use {pathless}")


def _require_auth_server() -> None:
    """Raise a helpful error when the MLflow auth-server package is unavailable."""
    if _AUTH_SERVER_IMPORT_ERROR is not None:
        raise MlflowSageMakerException(_AUTH_SERVER_MISSING_MESSAGE) from _AUTH_SERVER_IMPORT_ERROR


class SageMakerMlflowAuthClient(_AuthServiceClient):
    """RBAC admin client for a SageMaker MLflow app, keyed by IAM role ARN.

    Subclasses MLflow's :class:`~mlflow.server.auth.client.AuthServiceClient`,
    inheriting its full role and permission surface, and specializes it for
    SageMaker MLflow FGAC: identity is an IAM role ARN, authentication is SigV4
    (no passwords), and the target app is the active MLflow tracking URI.

    The caller must be authenticated as the app's Platform Admin (the
    ``AdminPrincipalArn`` configured via ``AccessControlConfig``); the server
    authorizes each request by SigV4 identity.

    .. code-block:: python
        :caption: Example

        import mlflow
        from sagemaker_mlflow import SageMakerMlflowAuthClient

        mlflow.set_tracking_uri(
            "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/my-app"
        )
        client = SageMakerMlflowAuthClient()

        client.create_user("arn:aws:iam::123456789012:role/AliceDSRole")
        client.update_user_admin(
            "arn:aws:iam::123456789012:role/AliceDSRole", is_admin=True
        )

        role = client.create_role(workspace="default", name="viewers")
        client.assign_role(
            "arn:aws:iam::123456789012:role/AliceDSRole", role.id
        )
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """Resolve the target SageMaker MLflow app and bind the client to it.

        Args:
            tracking_uri: SageMaker MLflow app ARN to target. Defaults to the
                active MLflow tracking URI (``mlflow.get_tracking_uri()``), the
                same resolution the other plugin helpers use.

        Raises:
            MlflowSageMakerException: if the MLflow auth-server package is not
                installed (install ``sagemaker-mlflow[full]``), or if no
                tracking URI can be resolved.
        """
        _require_auth_server()
        resolved_uri = tracking_uri or mlflow.get_tracking_uri()
        if not resolved_uri:
            raise MlflowSageMakerException(
                "No tracking URI is set. Pass tracking_uri or call "
                "mlflow.set_tracking_uri() with the SageMaker MLflow app ARN."
            )
        super().__init__(resolved_uri)

    def _request(self, endpoint, method, *, expected_status: int = 200, **kwargs):
        """Issue a signed request against the SageMaker MLflow endpoint.

        Overrides the upstream ``_request`` to build host creds via
        :func:`get_host_creds` (``auth="arn"`` → SigV4) instead of basic auth.
        """
        from mlflow.utils.rest_utils import http_request, verify_rest_response

        host_creds = get_host_creds(self.tracking_uri)
        resp = http_request(host_creds, endpoint, method, **kwargs)
        resp = verify_rest_response(resp, endpoint, expected_status=expected_status)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # ---- User management: role-ARN only, password-free ----

    def create_user(self, role_arn: str):
        """Register an IAM role ARN as an MLflow user (no password).

        Args:
            role_arn: The IAM role ARN to register, e.g.
                ``"arn:aws:iam::123456789012:role/AliceDSRole"``. Must not
                include a path. This value becomes the MLflow username.

        Returns:
            The created :py:class:`mlflow.server.auth.entities.User`.

        Raises:
            MlflowSageMakerException: if ``role_arn`` is not a well-formed IAM
                role ARN or includes a path.
            mlflow.exceptions.RestException: if the server rejects the request
                (for example, the caller is not the admin, access control is not
                enabled, or the user already exists).
        """
        _validate_role_arn(role_arn)
        logger.info("Registering MLflow user %s on %s", role_arn, self.tracking_uri)
        return super().create_user(role_arn, _EMPTY_PASSWORD)

    def update_user_admin(self, role_arn: str, is_admin: bool):
        """Set the admin flag on an IAM-registered MLflow user (no password).

        The admin flag is the only mutable field for an IAM-keyed user: identity
        is the role ARN and there is no password to change.

        Args:
            role_arn: The IAM role ARN of an already-registered user. Must not
                include a path.
            is_admin: ``True`` to grant admin privileges, ``False`` to revoke.

        Raises:
            MlflowSageMakerException: if ``role_arn`` is not a well-formed IAM
                role ARN or includes a path.
            mlflow.exceptions.RestException: if the server rejects the request.
        """
        _validate_role_arn(role_arn)
        logger.info("Setting is_admin=%s for MLflow user %s on %s", is_admin, role_arn, self.tracking_uri)
        return super().update_user_admin(role_arn, is_admin)

    def update_user_password(self, *args, **kwargs):
        """Unsupported: a SageMaker MLflow app authenticates by IAM principal.

        Raises:
            MlflowSageMakerException: always. There is no password to set; the
                server refuses ``users/update-password`` for FGAC apps.
        """
        raise MlflowSageMakerException("This app authenticates by IAM principal, so a password cannot be set")

    def get_user(self, role_arn: str):
        """Get a registered user by IAM role ARN."""
        _validate_role_arn(role_arn)
        return super().get_user(role_arn)

    def delete_user(self, role_arn: str):
        """Delete a registered user by IAM role ARN."""
        _validate_role_arn(role_arn)
        return super().delete_user(role_arn)

    # ---- Role assignment / per-user permissions: validate the user ARN ----

    def assign_role(self, role_arn: str, role_id: int):
        """Assign an RBAC role to an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().assign_role(role_arn, role_id)

    def unassign_role(self, role_arn: str, role_id: int):
        """Remove an RBAC role from an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().unassign_role(role_arn, role_id)

    def list_user_roles(self, role_arn: str):
        """List the RBAC roles assigned to an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().list_user_roles(role_arn)

    def grant_user_permission(self, role_arn: str, resource_type: str, resource_id: str, permission: str):
        """Grant a single resource permission to an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().grant_user_permission(role_arn, resource_type, resource_id, permission)

    def revoke_user_permission(self, role_arn: str, resource_type: str, resource_id: str):
        """Revoke a single resource permission from an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().revoke_user_permission(role_arn, resource_type, resource_id)

    def get_user_permission(self, role_arn: str, resource_type: str, resource_id: str):
        """Check a single resource permission for an IAM-registered user."""
        _validate_role_arn(role_arn)
        return super().get_user_permission(role_arn, resource_type, resource_id)
