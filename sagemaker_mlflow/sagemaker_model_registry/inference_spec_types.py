# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/

"""SageMaker InferenceSpecification types and helpers."""

from typing import Dict, List, Literal, TypedDict


class S3DataSource(TypedDict, total=False):
    S3Uri: str
    S3DataType: Literal["S3Prefix", "S3Object"]
    CompressionType: Literal["None", "Gzip"]
    ETag: str
    ManifestS3Uri: str
    ManifestEtag: str


class ModelDataSource(TypedDict, total=False):
    S3DataSource: S3DataSource


class AdditionalModelDataSource(TypedDict, total=False):
    ChannelName: str
    S3DataSource: S3DataSource


class ModelInput(TypedDict, total=False):
    DataInputConfig: str


class ContainerDefinition(TypedDict, total=False):
    Image: str
    ImageDigest: str
    ModelDataUrl: str
    ModelDataETag: str
    ModelDataSource: ModelDataSource
    AdditionalModelDataSources: List[AdditionalModelDataSource]
    ContainerHostname: str
    Environment: Dict[str, str]
    Framework: str
    FrameworkVersion: str
    NearestModelName: str
    ModelInput: ModelInput


class InferenceSpecification(TypedDict, total=False):
    Containers: List[ContainerDefinition]
    SupportedContentTypes: List[str]
    SupportedResponseMIMETypes: List[str]
    SupportedRealtimeInferenceInstanceTypes: List[str]
    SupportedTransformInstanceTypes: List[str]
