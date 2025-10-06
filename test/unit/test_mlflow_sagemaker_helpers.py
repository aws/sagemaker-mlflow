import unittest

from sagemaker_mlflow.mlflow_sagemaker_helpers import (
    validate_and_parse_arn,
    get_tracking_server_url,
    get_dns_suffix,
    Arn,
)
from sagemaker_mlflow.exceptions import MlflowSageMakerException
from unittest import TestCase
from unittest import mock
import os


TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_ARN_WITH_ROLE = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw#arn:aws:iam::0123456789:role/role-name-with-path"
TEST_ASSUME_ROLE = "arn:aws:iam::0123456789:role/role-name-with-path"


class MlflowSageMakerHelpersTest(TestCase):

    def test_validate_and_parse_arn_happy(self):
        arn = TEST_VALID_ARN
        result = validate_and_parse_arn(arn)
        assert type(result) is Arn
        assert result.partition == "aws"
        assert result.service == "sagemaker"
        assert result.region == "us-west-2"
        assert result.account == "000000000000"
        assert result.resource_type == "mlflow-tracking-server"
        assert result.resource_id == "xw"
        assert result.maybe_assume_role_arn is None

    def test_validate_and_parse_arn_with_assume_role_arn(self):
        arn = TEST_VALID_ARN
        assume_role = TEST_ASSUME_ROLE
        result = validate_and_parse_arn(arn)

        assert type(result) is Arn
        assert result.partition == "aws"
        assert result.service == "sagemaker"
        assert result.region == "us-west-2"
        assert result.account == "000000000000"
        assert result.resource_type == "mlflow-tracking-server"
        assert result.resource_id == "xw"
        assert result.maybe_assume_role_arn == assume_role

    def test_validate_and_parse_arn_invalid_service(self):
        arn = "arn:aws:wagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
        with self.assertRaises(MlflowSageMakerException):
            validate_and_parse_arn(arn)

    def test_validate_and_parse_arn_invalid_arn(self):
        arn = "arn:aws:sagemaker:us-west-2mlflow-tracking-server/xw"
        with self.assertRaises(Exception):
            validate_and_parse_arn(arn)

    def test_get_tracking_server_url_normal(self):
        url = get_tracking_server_url(TEST_VALID_ARN)
        assert url == "https://us-west-2.experiments.sagemaker.aws"

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT": "https://a.com"},
    )
    def test_get_tracking_server_url_custom(self):
        url = get_tracking_server_url(TEST_VALID_ARN)
        assert url == "https://a.com"

    def test_dns_suffix_happy(self):
        suffix = get_dns_suffix("aws")
        assert suffix == "aws"

    def test_dns_suffix_invalid(self):
        with self.assertRaises(MlflowSageMakerException):
            get_dns_suffix("aws-ocean")


if __name__ == "__main__":
    unittest.main()
