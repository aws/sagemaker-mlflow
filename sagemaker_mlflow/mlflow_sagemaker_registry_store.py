from mlflow.store.model_registry.rest_store import RestStore
from sagemaker_mlflow.mlflow_sagemaker_helpers import get_tracking_server_url
from mlflow.utils import rest_utils
from functools import partial
import os

class MlflowSageMakerRegistryStore(RestStore):

    store_uri = ""

    def __init__(self, store_uri):
        self.store_uri = store_uri
        super().__init__(partial(get_host_creds, store_uri))


# Extra environment variables which take precedence for setting the basic/bearer
# auth on http requests.
_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"
_TRACKING_TOKEN_ENV_VAR = "MLFLOW_TRACKING_TOKEN"

# sets verify param of 'requests.request' function
# see https://requests.readthedocs.io/en/master/api/
_TRACKING_INSECURE_TLS_ENV_VAR = "MLFLOW_TRACKING_INSECURE_TLS"
_TRACKING_SERVER_CERT_PATH_ENV_VAR = "MLFLOW_TRACKING_SERVER_CERT_PATH"

# sets cert param of 'requests.request' function
# see https://requests.readthedocs.io/en/master/api/
_TRACKING_CLIENT_CERT_PATH_ENV_VAR = "MLFLOW_TRACKING_CLIENT_CERT_PATH"


def get_host_creds(store_uri):
    """Configuring mlflow's client"""
    tracking_server_url = get_tracking_server_url(store_uri)
    return rest_utils.MlflowHostCreds(
        host=tracking_server_url,
        username=os.environ.get(_TRACKING_USERNAME_ENV_VAR),
        password=os.environ.get(_TRACKING_PASSWORD_ENV_VAR),
        token=os.environ.get(_TRACKING_TOKEN_ENV_VAR),
        auth="arn",
        # aws_sigv4="False",  # Auth provider is used instead for now
        ignore_tls_verification=os.environ.get(_TRACKING_INSECURE_TLS_ENV_VAR)
        == "true",
        client_cert_path=os.environ.get(_TRACKING_CLIENT_CERT_PATH_ENV_VAR),
        server_cert_path=os.environ.get(_TRACKING_SERVER_CERT_PATH_ENV_VAR),
    )
