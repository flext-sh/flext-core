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
from types import EllipsisType
from typing import TYPE_CHECKING, ClassVar, dataclass_transform, overload

import pydantic
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
from pydantic.fields import FieldInfo
from typing_extensions import TypeForm

if TYPE_CHECKING:
    from flext_core import p


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    # Preserve the native overloads and field-specifier identity without a wrapper.
    Field = staticmethod(pydantic.Field)
    FieldInfo = FieldInfo

    if TYPE_CHECKING:
        PrivateAttr: ClassVar[p.PrivateAttrFactory]
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

    field_validator = staticmethod(pydantic.field_validator)
    # Why (abstraction boundary): ENFORCE-070 makes flext-core the sole owner of
    # pydantic, and the tier-whitelist gate rejects a bare ``pydantic`` import in
    # every downstream project. A name that this facade does not re-export
    # therefore has no compliant spelling at all: the consumer must either import
    # pydantic directly (gate violation) or reach forward into a later layer
    # (namespace chain violation). ``model_validator`` is the canonical
    # model-level counterpart of ``field_validator`` above, and its absence alone
    # forced bare pydantic imports across 32 downstream modules.
    # Re-export native decorators with their complete upstream overload sets.
    model_validator = staticmethod(pydantic.model_validator)
    field_serializer: ClassVar[p.FieldSerializerFactory] = pydantic.field_serializer
    model_serializer: ClassVar[p.ModelSerializerFactory] = pydantic.model_serializer
    computed_field: ClassVar[p.ComputedFieldFactory] = pydantic.computed_field

    @overload
    @staticmethod
    def empty[ContractT](contract: TypeForm[ContractT]) -> Callable[[], ContractT]:
        raise NotImplementedError

    @overload
    @staticmethod
    def empty[ContractT](
        *, contract_factory: Callable[[], TypeForm[ContractT]]
    ) -> Callable[[], ContractT]:
        raise NotImplementedError

    @staticmethod
    def empty[ContractT](
        contract: TypeForm[ContractT] | EllipsisType = ...,
        *,
        contract_factory: Callable[[], TypeForm[ContractT]] | None = None,
    ) -> Callable[[], ContractT]:
        from flext_core._utilities.pydantic import FlextUtilitiesPydantic

        if contract_factory is None:
            if isinstance(contract, EllipsisType):
                msg = "empty requires a collection contract or contract_factory"
                raise TypeError(msg)
            return FlextUtilitiesPydantic.empty(contract)
        if isinstance(contract, EllipsisType):
            return FlextUtilitiesPydantic.empty(contract_factory=contract_factory)
        msg = "empty accepts either contract or contract_factory, not both"
        raise TypeError(msg)

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
    SkipValidation = SkipValidation

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
