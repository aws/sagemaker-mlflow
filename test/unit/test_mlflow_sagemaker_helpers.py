import unittest
import warnings

from sagemaker_mlflow.mlflow_sagemaker_helpers import (
    SageMakerMLflowHostMetadataProvider,
    validate_and_parse_arn,
    get_tracking_server_url,
    get_dns_suffix,
    Arn,
)
from sagemaker_mlflow.exceptions import MlflowSageMakerException, ResourceTypeUnsupportedException
from unittest import TestCase
from unittest import mock
import os


TEST_VALID_ARN = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
TEST_VALID_ARN_MLFLOW_APP = "arn:aws:sagemaker:us-west-2:000000000000:mlflow-app/xw"
TEST_VALID_ROLE_ARN = "arn:aws:iam::0123456789:role/role-name-with-path"


class MlflowSageMakerHelpersTest(TestCase):

    def test_validate_and_parse_arn_tracking_server(self):
        arn = TEST_VALID_ARN
        result = SageMakerMLflowHostMetadataProvider()
        result.set_arn(arn)
        assert result.partition == "aws"
        assert result.service == "sagemaker"
        assert result.region == "us-west-2"
        assert result.account == "000000000000"
        assert result.resource_type == "mlflow-tracking-server"
        assert result.resource_id == "xw"

    def test_validate_and_parse_arn_mlflow_app(self):
        arn = TEST_VALID_ARN_MLFLOW_APP
        result = SageMakerMLflowHostMetadataProvider()
        result.set_arn(arn)
        assert result.partition == "aws"
        assert result.service == "sagemaker"
        assert result.region == "us-west-2"
        assert result.account == "000000000000"
        assert result.resource_type == "mlflow-app"
        assert result.resource_id == "xw"

    def test_validate_and_parse_arn_invalid_service(self):
        arn = "arn:aws:wagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
        with self.assertRaises(MlflowSageMakerException):
            result = SageMakerMLflowHostMetadataProvider()
            result.set_arn(arn)

    def test_validate_and_parse_arn_invalid_arn(self):
        arn = "arn:aws:sagemaker:us-west-2mlflow-tracking-server/xw"
        with self.assertRaises(Exception):
            result = SageMakerMLflowHostMetadataProvider()
            result.set_arn(arn)

    def test_construct_tracking_server_url_tracking_server(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN)
        url = provider.construct_tracking_server_url()
        assert url == "https://us-west-2.experiments.sagemaker.aws"

    def test_construct_tracking_server_url_mlflow_app(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN_MLFLOW_APP)
        url = provider.construct_tracking_server_url()
        assert url == "https://mlflow.sagemaker.us-west-2.app.aws"

    def test_construct_tracking_server_url_gov(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN_MLFLOW_APP)
        provider.partition = "aws-us-gov"
        url = provider.construct_tracking_server_url()
        assert url == "https://mlflow.sagemaker.us-west-2.app.aws"

    def test_construct_tracking_server_url_partition_unknown(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN_MLFLOW_APP)
        provider.partition = "wee"

        self.assertRaises(MlflowSageMakerException, provider.construct_tracking_server_url)

    def test_construct_tracking_server_url_resource_type_unknown(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN_MLFLOW_APP)
        provider.resource_type = "wee"

        self.assertRaises(ResourceTypeUnsupportedException, provider.construct_tracking_server_url)

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT": "https://a.com"},
    )
    def test_get_tracking_server_url_custom(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN_MLFLOW_APP)
        url = provider.construct_tracking_server_url()
        assert url == "https://a.com"

    def test_maybe_assume_role_arn_none(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN)
        assert provider.maybe_assume_role_arn is None

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": TEST_VALID_ROLE_ARN},
    )
    def test_maybe_assume_role_arn_valid(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN)
        assert provider.maybe_assume_role_arn == TEST_VALID_ROLE_ARN

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"},
    )
    def test_maybe_assume_role_arn_invalid_service(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN)
        with self.assertRaises(MlflowSageMakerException):
            _ = provider.maybe_assume_role_arn

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": "invalid-arn"},
    )
    def test_maybe_assume_role_arn_invalid_format(self):
        provider = SageMakerMLflowHostMetadataProvider()
        provider.set_arn(TEST_VALID_ARN)
        with self.assertRaises(MlflowSageMakerException):
            _ = provider.maybe_assume_role_arn


class DeprecatedArnTest(TestCase):
    """Tests for deprecated Arn class and related functions"""

    def test_validate_and_parse_arn_happy(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
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
            assert len(w) == 2  # One for validate_and_parse_arn, one for Arn.__init__
            assert issubclass(w[0].category, DeprecationWarning)

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": TEST_VALID_ROLE_ARN},
    )
    def test_validate_and_parse_arn_with_assume_role_arn(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arn = TEST_VALID_ARN
            result = validate_and_parse_arn(arn)

            assert type(result) is Arn
            assert result.partition == "aws"
            assert result.service == "sagemaker"
            assert result.region == "us-west-2"
            assert result.account == "000000000000"
            assert result.resource_type == "mlflow-tracking-server"
            assert result.resource_id == "xw"
            assert result.maybe_assume_role_arn == TEST_VALID_ROLE_ARN

    def test_validate_and_parse_arn_invalid_service(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arn = "arn:aws:wagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"
            with self.assertRaises(MlflowSageMakerException):
                validate_and_parse_arn(arn)

    def test_validate_and_parse_arn_invalid_arn(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arn = "arn:aws:sagemaker:us-west-2mlflow-tracking-server/xw"
            with self.assertRaises(Exception):
                validate_and_parse_arn(arn)

    def test_get_tracking_server_url_normal(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            url = get_tracking_server_url(TEST_VALID_ARN)
            assert url == "https://us-west-2.experiments.sagemaker.aws"

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT": "https://a.com"},
    )
    def test_get_tracking_server_url_custom(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            url = get_tracking_server_url(TEST_VALID_ARN)
            assert url == "https://a.com"

    def test_dns_suffix_happy(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            suffix = get_dns_suffix("aws")
            assert suffix == "aws"

    def test_dns_suffix_invalid(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with self.assertRaises(MlflowSageMakerException):
                get_dns_suffix("aws-ocean")

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": "arn:aws:sagemaker:us-west-2:000000000000:mlflow-tracking-server/xw"},
    )
    def test_validate_and_parse_arn_with_invalid_assume_role_arn_service(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arn = TEST_VALID_ARN
            result = validate_and_parse_arn(arn)
            with self.assertRaises(MlflowSageMakerException):
                _ = result.maybe_assume_role_arn

    @mock.patch.dict(
        os.environ,
        {"SAGEMAKER_MLFLOW_ASSUME_ROLE_ARN": "invalid-arn"},
    )
    def test_validate_and_parse_arn_with_invalid_assume_role_arn_format(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            arn = TEST_VALID_ARN
            result = validate_and_parse_arn(arn)
            with self.assertRaises(MlflowSageMakerException):
                _ = result.maybe_assume_role_arn


if __name__ == "__main__":
    unittest.main()
