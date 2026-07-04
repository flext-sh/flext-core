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

from collections.abc import Callable
from pathlib import Path
from re import Pattern

from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel as PydanticBaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    FieldSerializationInfo,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    GetPydanticSchema,
    JsonValue,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    RootModel as PydanticRootModel,
    SkipValidation,
    TypeAdapter as PydanticTypeAdapter,
    ValidationError,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import (
    PydanticUndefined,
    PydanticUndefinedType,
    SchemaValidator,
)
from pydantic_settings import (
    BaseSettings as PydanticBaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

type _FieldValue = JsonValue | Path
type _FieldKeywordValue = (
    _FieldValue
    | PydanticUndefinedType
    | FieldInfo
    | AliasChoices
    | AliasPath
    | Discriminator
    | Pattern[str]
    | Callable[..., _FieldValue | None]
)
_FIELD_FACTORY: Callable[..., FieldInfo] = Field


def _field[DefaultT](
    default: DefaultT | PydanticUndefinedType = PydanticUndefined,
    **kwargs: _FieldKeywordValue | None,
) -> FieldInfo:
    """Typed FLEXT facade for ``pydantic.Field``."""
    field = _FIELD_FACTORY(default, **kwargs)
    if not isinstance(field, FieldInfo):
        msg = "pydantic.Field returned a non-FieldInfo value"
        raise TypeError(msg)
    return field


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

    Field = staticmethod(_field)
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

    TypeAdapter = PydanticTypeAdapter

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
