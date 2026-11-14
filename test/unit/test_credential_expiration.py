from datetime import datetime, timezone

import pytest

from sagemaker_mlflow.credential_cache import CredentialCache


@pytest.mark.parametrize("expiry, ttl, cutoff", [(1100, 3600, 1100), (999, 3600, 1000), (5000, 100, 1100)])
def test_cache_expires_at_earlier_of_ttl_and_sts_expiration(monkeypatch, expiry, ttl, cutoff):
    clock = [1000.0]
    monkeypatch.setattr("sagemaker_mlflow.credential_cache.time.time", lambda: clock[0])
    credentials = {"AccessKeyId": "test", "Expiration": datetime.fromtimestamp(expiry, timezone.utc)}
    cache = CredentialCache()
    cache.set_credentials("role", credentials, ttl)

    if cutoff > clock[0]:
        assert cache.get_credentials("role") == credentials
    clock[0] = cutoff
    assert cache.get_credentials("role") is None
