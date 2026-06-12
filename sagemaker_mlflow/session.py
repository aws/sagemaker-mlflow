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

"""Caller-provided boto3 Session injection for the SageMaker MLflow plugin.

The MLflow plugin entry-point API does not let callers forward arguments to
plugin instances, so callers that want the SigV4 signer to use a non-default
boto3 Session (e.g. a specific profile, refreshable credentials, or per-tenant
credentials in a shared process) need an out-of-band transport. This module
provides one via a ``contextvars.ContextVar``.

Resolution order applied by ``AuthBoto``:

1. Explicit ``boto3_session=`` kwarg on ``AuthBoto.__init__``.
2. Session set via ``set_session()`` or ``use_session()`` here.
3. ``boto3.Session()`` (the default credential chain).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

import boto3

_session_var: ContextVar[Optional[boto3.Session]] = ContextVar("sagemaker_mlflow_session", default=None)


def set_session(session: Optional[boto3.Session]) -> None:
    """Set the boto3 Session used by AuthBoto for the current context.

    Pass ``None`` to clear. Scoped to the current ``contextvars.Context`` —
    in practice this is per-thread / per-asyncio-task. For block-scoped
    overrides prefer :func:`use_session`.
    """
    _session_var.set(session)


@contextmanager
def use_session(session: Optional[boto3.Session]) -> Iterator[None]:
    """Context manager that scopes a boto3 Session to a ``with`` block.

    The previous value is restored on exit, including when the body raises.
    """
    token = _session_var.set(session)
    try:
        yield
    finally:
        _session_var.reset(token)


def _get_current_session() -> Optional[boto3.Session]:
    """Return the session currently bound by ``set_session``/``use_session``."""
    return _session_var.get()
