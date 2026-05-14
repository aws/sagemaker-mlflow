import unittest
from unittest.mock import Mock

from sagemaker_mlflow.session import _get_current_session, set_session, use_session


class TestSession(unittest.TestCase):

    def tearDown(self):
        # Ensure no leaked state across tests.
        set_session(None)

    def test_use_session_sets_and_clears(self):
        s = Mock()
        with use_session(s):
            self.assertIs(_get_current_session(), s)
        self.assertIsNone(_get_current_session())

    def test_use_session_nested(self):
        s1 = Mock()
        s2 = Mock()
        with use_session(s1):
            self.assertIs(_get_current_session(), s1)
            with use_session(s2):
                self.assertIs(_get_current_session(), s2)
            self.assertIs(_get_current_session(), s1)
        self.assertIsNone(_get_current_session())

    def test_set_session_persists(self):
        s = Mock()
        set_session(s)
        self.assertIs(_get_current_session(), s)
        set_session(None)
        self.assertIsNone(_get_current_session())

    def test_use_session_restores_after_exception(self):
        s = Mock()

        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            with use_session(s):
                self.assertIs(_get_current_session(), s)
                raise Boom()

        self.assertIsNone(_get_current_session())


if __name__ == "__main__":
    unittest.main()
