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

import threading
import time
from typing import Optional, Dict, Any


class CredentialCache:
    """Thread-safe TTL cache for AWS STS assumed role credentials."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_credentials(self, role_arn: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached credentials for the given role ARN if they haven't expired.

        Args:
            role_arn (str): The ARN of the role for which to retrieve credentials

        Returns:
            Optional[Dict[str, Any]]: Cached credentials if valid, None if expired or not found
        """
        with self._lock:
            if role_arn not in self._cache:
                return None

            cached_entry = self._cache[role_arn]
            current_time = time.time()

            if current_time >= cached_entry["expires_at"]:
                # Credentials have expired, remove from cache
                del self._cache[role_arn]
                return None

            return cached_entry["credentials"]

    def set_credentials(self, role_arn: str, credentials: Dict[str, Any], ttl_seconds: int):
        """
        Store credentials in the cache with a TTL.

        Args:
            role_arn (str): The ARN of the role for which to cache credentials
            credentials (Dict[str, Any]): The credentials to cache
            ttl_seconds (int): Time-to-live in seconds
        """
        expires_at = time.time() + ttl_seconds

        with self._lock:
            self._cache[role_arn] = {
                "credentials": credentials,
                "expires_at": expires_at
            }
            # Clean up expired entries periodically
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired entries from the cache to prevent memory leaks."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time >= entry["expires_at"]
        ]

        for key in expired_keys:
            del self._cache[key]

    def clear(self):
        """Clear all cached credentials. Useful for testing."""
        with self._lock:
            self._cache.clear()