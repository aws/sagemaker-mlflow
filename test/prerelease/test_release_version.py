import sagemaker_mlflow

from packaging.version import Version


def test_release_version():
    plugin_version = Version(sagemaker_mlflow.__version__)
    assert not plugin_version.is_devrelease, f"sagemaker_mlflow version is dev - {plugin_version}"
    assert not plugin_version.is_prerelease, f"sagemaker_mlflow version is prerelease - {plugin_version}"
    assert not plugin_version.is_postrelease, f"sagemaker_mlflow version is postrelease - {plugin_version}"
