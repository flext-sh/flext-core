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
from types import EllipsisType
from typing import Literal, dataclass_transform

from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel as PydanticBaseModel,
    BeforeValidator,
    ConfigDict as _PydanticConfigDict,
    Discriminator,
    FailFast,
    Field,
    FieldSerializationInfo as PydanticFieldSerializationInfo,
    GetPydanticSchema,
    InstanceOf as PydanticInstanceOf,
    JsonValue,
    PlainSerializer,
    PlainValidator,
    PrivateAttr as PydanticPrivateAttr,
    RootModel as PydanticRootModel,
    SerializeAsAny,
    SkipValidation,
    StringConstraints,
    TypeAdapter as PydanticTypeAdapter,
    ValidateAs,
    ValidationError,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from pydantic_settings import (
    BaseSettings as _PydanticBaseSettings,
    PydanticBaseSettingsSource as _PydanticBaseSettingsSource,
    SettingsConfigDict as _PydanticSettingsConfigDict,
)

type _FieldValue = JsonValue | Path
type _FieldSchemaExtra = Mapping[str, _FieldValue | Sequence[_FieldValue]]
type _FieldKeywordValue[DefaultT] = (
    _FieldValue
    | _FieldSchemaExtra
    | PydanticUndefinedType
    | PydanticFieldInfo
    | AliasChoices
    | AliasPath
    | Discriminator
    | Pattern[str]
    | Callable[..., DefaultT]
    | Callable[..., _FieldValue | None]
    | type[DefaultT]
)


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    @staticmethod
    def _field[DefaultT](
        default: DefaultT | PydanticUndefinedType | EllipsisType = PydanticUndefined,
        **kwargs: _FieldKeywordValue[DefaultT] | None,
    ) -> DefaultT:
        """Typed FLEXT facade for ``pydantic.Field``.

        Returns:
            The resulting ``DefaultT``.

        """
        field_factory: Callable[..., DefaultT] = Field
        return field_factory(default, **kwargs)

    @staticmethod
    def _private_attr[PrivateT](
        default: PrivateT | PydanticUndefinedType = PydanticUndefined,
        *,
        default_factory: Callable[..., PrivateT] | None = None,
        init: Literal[False] = False,
    ) -> PrivateT:
        """Typed FLEXT facade for ``pydantic.PrivateAttr``.

        Returns:
            The resulting ``PrivateT``.

        """
        private_attr_factory: Callable[..., PrivateT] = PydanticPrivateAttr
        return private_attr_factory(default, default_factory=default_factory, init=init)

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(
            _field,
            Field,
            PydanticPrivateAttr,
            _private_attr,
        ),
    )
    class BaseModel(PydanticBaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    class BaseSettings(_PydanticBaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(
            _field,
            Field,
            PydanticPrivateAttr,
            _private_attr,
        ),
    )
    class RootModel[RootValueT](PydanticRootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    class ConfigDict(_PydanticConfigDict, total=False):
        """Canonical model configuration exported through the models facade."""

    class SettingsConfigDict(_PydanticSettingsConfigDict, total=False):
        """Canonical settings configuration exported through the models facade."""

    Field = staticmethod(_field)
    PrivateAttr = staticmethod(_private_attr)
    SkipValidation = SkipValidation
    # Same unwrapped-class-attribute problem as PrivateAttr above: pyright
    # binds the bare decorator through the facade and infers the facade type
    # for every decorated property (reportIndexIssue on real consumers).
    computed_field = staticmethod(computed_field)
    field_validator = field_validator
    # Why (abstraction boundary): ENFORCE-070 makes flext-core the sole owner of
    # pydantic, and the tier-whitelist gate rejects a bare ``pydantic`` import in
    # every downstream project. A name that this facade does not re-export
    # therefore has no compliant spelling at all: the consumer must either import
    # pydantic directly (gate violation) or reach forward into a later layer
    # (namespace chain violation). ``model_validator`` is the canonical
    # model-level counterpart of ``field_validator`` above, and its absence alone
    # forced bare pydantic imports across 32 downstream modules.
    model_validator = model_validator

    # Annotation constraints and tagged-union discrimination
    Discriminator = Discriminator
    StringConstraints = StringConstraints
    # Field alias declarations (runtime markers placed in ``Field`` arguments).
    AliasChoices = AliasChoices
    AliasPath = AliasPath

    # Annotation validators
    AfterValidator = AfterValidator
    BeforeValidator = BeforeValidator
    FailFast = FailFast
    type InstanceOf[T] = PydanticInstanceOf[T]
    PlainValidator = PlainValidator
    ValidateAs = ValidateAs
    WrapValidator = WrapValidator

    # Serializers
    PlainSerializer = PlainSerializer
    SerializeAsAny = SerializeAsAny
    WrapSerializer = WrapSerializer

    # Types of the pydantic final classes and protocols: annotation-only names.
    # ``u.type_adapter`` constructs the adapter this type describes.
    type FieldInfo = PydanticFieldInfo
    type FieldSerializationInfo = PydanticFieldSerializationInfo
    type TypeAdapter[T] = PydanticTypeAdapter[T]

    # Annotation marker that wraps a callable schema hook.
    GetPydanticSchema = GetPydanticSchema

    # Validation exception (re-exported so consumers avoid `import pydantic`)
    ValidationError = ValidationError

    # Settings-source hook contract: ``settings_customise_sources`` overrides
    # annotate the wide upstream base (Liskov-correct parameter widening).
    type PydanticBaseSettings = _PydanticBaseSettings
    type PydanticBaseSettingsSource = _PydanticBaseSettingsSource


__all__: list[str] = ["FlextModelsPydantic"]
