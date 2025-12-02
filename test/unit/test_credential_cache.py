import unittest
import threading
from unittest.mock import patch

from sagemaker_mlflow.credential_cache import CredentialCache


class TestCredentialCache(unittest.TestCase):

    def setUp(self):
        self.cache = CredentialCache()
        self.test_role_arn = "arn:aws:iam::123456789012:role/test-role"
        self.test_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
        }

    def test_cache_miss_returns_none(self):
        # Test that cache miss returns None
        result = self.cache.get_credentials(self.test_role_arn)
        self.assertIsNone(result)

    def test_cache_hit_returns_credentials(self):
        # Test that cache hit returns stored credentials
        ttl_seconds = 300
        self.cache.set_credentials(self.test_role_arn, self.test_credentials, ttl_seconds)

        result = self.cache.get_credentials(self.test_role_arn)
        self.assertEqual(result, self.test_credentials)

    @patch("time.time")
    def test_expired_credentials_return_none(self, mock_time):
        # Set up credentials that will expire
        mock_time.return_value = 1000.0
        ttl_seconds = 300
        self.cache.set_credentials(self.test_role_arn, self.test_credentials, ttl_seconds)

        # Fast forward time to after expiration
        mock_time.return_value = 1301.0  # 1000 + 300 + 1 seconds

        result = self.cache.get_credentials(self.test_role_arn)
        self.assertIsNone(result)

    @patch("time.time")
    def test_cleanup_expired_removes_old_entries(self, mock_time):
        # Add credentials that will expire
        mock_time.return_value = 1000.0
        ttl_seconds = 300
        self.cache.set_credentials(self.test_role_arn, self.test_credentials, ttl_seconds)

        # Verify credentials are cached
        result = self.cache.get_credentials(self.test_role_arn)
        self.assertEqual(result, self.test_credentials)

        # Fast forward time and add new credentials (triggers cleanup)
        mock_time.return_value = 1301.0
        other_role_arn = "arn:aws:iam::123456789012:role/other-role"
        self.cache.set_credentials(other_role_arn, self.test_credentials, ttl_seconds)

        # Verify expired credentials are cleaned up
        result = self.cache.get_credentials(self.test_role_arn)
        self.assertIsNone(result)

        # Verify new credentials are still there
        result = self.cache.get_credentials(other_role_arn)
        self.assertEqual(result, self.test_credentials)

    def test_multiple_role_arns_isolated(self):
        # Test that different role ARNs have isolated cache entries
        role_arn_1 = "arn:aws:iam::123456789012:role/role-1"
        role_arn_2 = "arn:aws:iam::123456789012:role/role-2"
        credentials_1 = {"AccessKeyId": "key1", "SecretAccessKey": "secret1", "SessionToken": "token1"}
        credentials_2 = {"AccessKeyId": "key2", "SecretAccessKey": "secret2", "SessionToken": "token2"}

        ttl_seconds = 300
        self.cache.set_credentials(role_arn_1, credentials_1, ttl_seconds)
        self.cache.set_credentials(role_arn_2, credentials_2, ttl_seconds)

        result_1 = self.cache.get_credentials(role_arn_1)
        result_2 = self.cache.get_credentials(role_arn_2)

        self.assertEqual(result_1, credentials_1)
        self.assertEqual(result_2, credentials_2)

    def test_clear_removes_all_entries(self):
        # Test that clear() removes all cached entries
        ttl_seconds = 300
        self.cache.set_credentials(self.test_role_arn, self.test_credentials, ttl_seconds)

        # Verify credentials are cached
        result = self.cache.get_credentials(self.test_role_arn)
        self.assertEqual(result, self.test_credentials)

        # Clear cache
        self.cache.clear()

        # Verify credentials are gone
        result = self.cache.get_credentials(self.test_role_arn)
        self.assertIsNone(result)

    def test_thread_safety(self):
        # Test that the cache is thread-safe
        results = []
        ttl_seconds = 300

        def cache_operation(thread_id):
            role_arn = f"arn:aws:iam::123456789012:role/role-{thread_id}"
            credentials = {
                "AccessKeyId": f"key-{thread_id}",
                "SecretAccessKey": f"secret-{thread_id}",
                "SessionToken": f"token-{thread_id}",
            }

            # Set credentials
            self.cache.set_credentials(role_arn, credentials, ttl_seconds)

            # Get credentials
            result = self.cache.get_credentials(role_arn)
            results.append((thread_id, result))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations completed successfully
        self.assertEqual(len(results), 10)
        for thread_id, result in results:
            expected_credentials = {
                "AccessKeyId": f"key-{thread_id}",
                "SecretAccessKey": f"secret-{thread_id}",
                "SessionToken": f"token-{thread_id}",
            }
            self.assertEqual(result, expected_credentials)


if __name__ == "__main__":
    unittest.main()
