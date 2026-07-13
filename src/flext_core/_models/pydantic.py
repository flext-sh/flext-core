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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, TypeAlias, dataclass_transform

from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel as PydanticBaseModel,
    BeforeValidator,
    ConfigDict as _PydanticConfigDict,
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
    SettingsConfigDict as _PydanticSettingsConfigDict,
)

if TYPE_CHECKING:
    from types import EllipsisType

type _FieldValue = JsonValue | Path
type _FieldSchemaExtra = Mapping[str, _FieldValue | Sequence[_FieldValue]]
type _FieldKeywordValue[DefaultT] = (
    _FieldValue
    | _FieldSchemaExtra
    | PydanticUndefinedType
    | FieldInfo
    | AliasChoices
    | AliasPath
    | Discriminator
    | Pattern[str]
    | Callable[..., DefaultT]
    | Callable[..., _FieldValue | None]
    | type[DefaultT]
)


def _field[DefaultT](
    default: DefaultT | PydanticUndefinedType | EllipsisType = PydanticUndefined,
    **kwargs: _FieldKeywordValue[DefaultT] | None,
) -> DefaultT:
    """Typed FLEXT facade for ``pydantic.Field``."""
    field_factory: Callable[..., DefaultT] = Field
    return field_factory(default, **kwargs)


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(_field, Field, PrivateAttr),
    )
    class BaseModel(PydanticBaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(_field, Field, PrivateAttr),
    )
    class BaseSettings(PydanticBaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(_field, Field, PrivateAttr),
    )
    class RootModel[RootValueT](PydanticRootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    # Pydantic field utilities
    ConfigDict: TypeAlias = _PydanticConfigDict
    SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict

    Field = staticmethod(_field)
    # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
    # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
    # function class attribute called through the facade (mixins.py:59 error).
    PrivateAttr = staticmethod(PrivateAttr)
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

    type TypeAdapterType[T] = PydanticTypeAdapter[T]
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
