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

from typing import TYPE_CHECKING, ClassVar, dataclass_transform

import pydantic
import pydantic.fields
import pydantic_settings
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BeforeValidator,
    Discriminator,
    FailFast,
    GetPydanticSchema,
    PlainSerializer,
    PlainValidator,
    SerializeAsAny,
    SkipValidation,
    StringConstraints,
    ValidateAs,
    WrapSerializer,
    WrapValidator,
)

from flext_core._protocols import FlextProtocolsBase, FlextProtocolsPydantic
from flext_core._typings.native_export import FlextTypingNativeExport


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    # Preserve the native overloads and field-specifier identity without a wrapper.
    Field = FlextTypingNativeExport(pydantic.Field)
    FieldInfo = pydantic.fields.FieldInfo

    # Compatibility: older flext-infra code calls m.empty(contract_factory=...).
    # Delegate to the canonical utility owner.
    @staticmethod
    def empty(*args, **kwargs):
        from flext_core._utilities.pydantic import FlextUtilitiesPydantic

        return FlextUtilitiesPydantic.empty(*args, **kwargs)

    if TYPE_CHECKING:
        PrivateAttr: ClassVar[FlextProtocolsPydantic.PrivateAttrFactory]
    else:
        PrivateAttr = staticmethod(pydantic.PrivateAttr)

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(pydantic.Field, pydantic.PrivateAttr),
    )
    class BaseModel(pydantic.BaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    class BaseSettings(pydantic_settings.BaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    class EnvSettingsSource(pydantic_settings.EnvSettingsSource):
        """Canonical environment source exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(pydantic.Field, pydantic.PrivateAttr),
    )
    class RootModel[RootValueT](pydantic.RootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    class ConfigDict(pydantic.ConfigDict, total=False):
        """Canonical model configuration exported through the models facade."""

    class SettingsConfigDict(pydantic_settings.SettingsConfigDict, total=False):
        """Canonical settings configuration exported through the models facade."""

    SkipValidation = SkipValidation
    field_validator = FlextTypingNativeExport(pydantic.field_validator)
    # Why (abstraction boundary): ENFORCE-070 makes flext-core the sole owner of
    # pydantic, and the tier-whitelist gate rejects a bare ``pydantic`` import in
    # every downstream project. A name that this facade does not re-export
    # therefore has no compliant spelling at all: the consumer must either import
    # pydantic directly (gate violation) or reach forward into a later layer
    # (namespace chain violation). ``model_validator`` is the canonical
    # model-level counterpart of ``field_validator`` above, and its absence alone
    # forced bare pydantic imports across 32 downstream modules.
    # Re-export native decorators with their complete upstream overload sets.
    model_validator = FlextTypingNativeExport(pydantic.model_validator)
    field_serializer: ClassVar[FlextProtocolsBase.FieldSerializerFactory] = (
        pydantic.field_serializer
    )
    model_serializer: ClassVar[FlextProtocolsBase.ModelSerializerFactory] = (
        pydantic.model_serializer
    )
    computed_field: ClassVar[FlextProtocolsBase.ComputedFieldFactory] = (
        pydantic.computed_field
    )

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
    type InstanceOf[T] = pydantic.InstanceOf[T]
    PlainValidator = PlainValidator
    ValidateAs = ValidateAs
    WrapValidator = WrapValidator

    # Serializers
    PlainSerializer = PlainSerializer
    SerializeAsAny = SerializeAsAny
    WrapSerializer = WrapSerializer

    # Types of the pydantic final classes and protocols: annotation-only names.
    # ``u.type_adapter`` constructs the adapter this type describes.
    type FieldSerializationInfo = pydantic.FieldSerializationInfo

    type TypeAdapter[T] = pydantic.TypeAdapter[T]

    # Annotation marker that wraps a callable schema hook.
    GetPydanticSchema = GetPydanticSchema

    # Validation exception (re-exported so consumers avoid `import pydantic`)
    ValidationError = pydantic.ValidationError

    # Settings-source hook contract: ``settings_customise_sources`` overrides
    # annotate the wide upstream base (Liskov-correct parameter widening).
    type PydanticBaseSettings = pydantic_settings.BaseSettings
    type PydanticBaseSettingsSource = pydantic_settings.PydanticBaseSettingsSource


__all__: list[str] = ["FlextModelsPydantic"]
