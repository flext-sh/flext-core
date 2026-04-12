"""Pydantic v2 base model types exported via FlextModels.

This module provides public aliases for pydantic v2 base model classes
that are used across the flext ecosystem. All projects consuming these
must extend from flext_core* instead of directly from pydantic.

Architecture: Abstraction boundary - models layer
Boundary: flext-core is sole owner of pydantic v2 integration. All other
projects receive pydantic model bases ONLY through public facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    FieldSerializationInfo,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    GetPydanticSchema,
    PlainSerializer,
    PlainValidator,
    RootModel,
    TypeAdapter,
    WrapSerializer,
    WrapValidator,
)
from pydantic_core import (
    SchemaValidator,
)
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    # Public Pydantic v2 model bases available via m.*
    BaseModel = BaseModel
    BaseSettings = BaseSettings
    RootModel = RootModel

    # Annotation validators
    AfterValidator = AfterValidator
    BeforeValidator = BeforeValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator

    # Serializers
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    # Validation and serialization context helpers
    FieldSerializationInfo = FieldSerializationInfo
    TypeAdapter = TypeAdapter

    # Schema and validator handlers
    GetCoreSchemaHandler = GetCoreSchemaHandler
    GetJsonSchemaHandler = GetJsonSchemaHandler
    GetPydanticSchema = GetPydanticSchema

    # Schema and JSON utilities (from pydantic_core)
    SchemaValidator = SchemaValidator

    # Settings sources (from pydantic_settings)
    EnvSettingsSource = EnvSettingsSource
    PydanticBaseSettingsSource = PydanticBaseSettingsSource
