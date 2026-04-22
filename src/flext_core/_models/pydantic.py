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
    BaseModel as PydanticBaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    GetPydanticSchema,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    RootModel as PydanticRootModel,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import (
    SchemaValidator,
)
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    class BaseModel(PydanticBaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    class BaseSettings(PydanticBaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    class RootModel[RootValueT](PydanticRootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    # Pydantic field utilities
    ConfigDict = ConfigDict
    SettingsConfigDict = SettingsConfigDict

    Field = Field
    PrivateAttr = PrivateAttr
    SkipValidation = SkipValidation
    computed_field = computed_field
    field_validator = field_validator

    # Annotation validators
    AfterValidator = AfterValidator
    BeforeValidator = BeforeValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator

    # Serializers
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    # Validation and serialization context helpers
    FieldInfo = FieldInfo
    FieldSerializationInfo = FieldSerializationInfo
    TypeAdapter = TypeAdapter

    # Schema and validator handlers
    GetCoreSchemaHandler = GetCoreSchemaHandler
    GetJsonSchemaHandler = GetJsonSchemaHandler
    GetPydanticSchema = GetPydanticSchema

    # Validation exception (re-exported so consumers avoid `import pydantic`)
    ValidationError = ValidationError

    # Schema and JSON utilities (from pydantic_core)
    SchemaValidator = SchemaValidator

    # Settings sources (from pydantic_settings)
    EnvSettingsSource = EnvSettingsSource
    PydanticBaseSettingsSource = PydanticBaseSettingsSource
