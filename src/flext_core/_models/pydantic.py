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
from functools import partialmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Literal,
    TypeAliasType,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)

from flext_core.constants import c

if TYPE_CHECKING:
    from flext_core._typings import FlextTypingBase

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
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from typing_extensions import TypeForm

from flext_core._protocols import FlextProtocolsBase, FlextProtocolsPydantic


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: m.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    type FieldValidatorCallable = (
        Callable[..., FlextProtocolsBase.AttributeProbe]
        | classmethod[type, ..., FlextProtocolsBase.AttributeProbe]
        | staticmethod[..., FlextProtocolsBase.AttributeProbe]
        | partialmethod[Callable[..., FlextProtocolsBase.AttributeProbe]]
    )

    # Static type resolves the polymorphic factory contract (protocols layer);
    # runtime binds the native pydantic functions byte-for-byte.
    if TYPE_CHECKING:
        Field: ClassVar[FlextProtocolsPydantic.FieldFactory]
        PrivateAttr: ClassVar[FlextProtocolsPydantic.PrivateAttrFactory]
    else:
        Field = staticmethod(pydantic.Field)
        PrivateAttr = staticmethod(pydantic.PrivateAttr)

    class EmptyDefault[ContractT]:
        """Default factory of a collection contract's canonical empty value.

        Built by ``u.empty``. The value is what pydantic validates an empty
        JSON document into for the contract's container, so a field default
        is exactly the value the same contract yields from a real empty input:
        it serializes, round-trips through JSON unchanged, and every call
        returns a fresh instance. An empty collection never validates an
        element, so only the container matters: element types, forward
        references and arbitrary types are never resolved. The adapter is
        built on the first call, keeping model class creation free of schema
        work. A contract that is not a collection fails loud with
        ``TypeError``.
        """

        __slots__ = ("_adapter", "_seed", "contract")

        def __init__(self, contract: TypeForm[ContractT]) -> None:
            """Bind the collection contract whose empty value this yields."""
            self.contract: TypeForm[ContractT] = contract
            self._adapter: pydantic.TypeAdapter[ContractT] | None = None
            self._seed: bytes = b""

        @staticmethod
        def collection_origin(contract: FlextTypingBase.TypeFormSpecifier) -> type:
            """Resolve a contract to its container class.

            ``Annotated`` metadata and PEP 695 aliases, plain or subscripted,
            are unwrapped until the outermost generic origin or class remains.

            Returns:
                The container class of ``contract``.

            Raises:
                TypeError: The contract does not resolve to a class.

            """
            current = contract
            while True:
                origin = get_origin(current)
                if origin is Annotated:
                    current = get_args(current)[0]
                elif isinstance(current, TypeAliasType):
                    current = current.__value__
                elif isinstance(origin, TypeAliasType):
                    current = origin.__value__
                else:
                    break
            resolved = origin or current
            if not isinstance(resolved, type):
                msg = f"empty default requires a collection contract; got {contract!r}"
                raise TypeError(msg)
            return resolved

        def __call__(self) -> ContractT:
            """Return a fresh canonical empty value of the contract.

            Returns:
                The contract's empty value as pydantic validates it.

            Raises:
                TypeError: The contract is not a collection.

            """
            adapter = self._adapter
            if adapter is None:
                origin = FlextModelsPydantic.EmptyDefault.collection_origin(
                    self.contract,
                )
                # The origin's empty value is the contract's empty value: no
                # element is ever validated, so the adapter is typed at this
                # single binding to the contract it stands for.
                adapter = cast(
                    "pydantic.TypeAdapter[ContractT]",
                    pydantic.TypeAdapter(origin),
                )
                schema = adapter.json_schema()
                shape = schema.get(c.JSON_SCHEMA_TYPE_KEY)
                seed = (
                    c.EMPTY_COLLECTION_JSON_SEEDS.get(shape)
                    if isinstance(shape, str)
                    and c.JSON_SCHEMA_PROPERTIES_KEY not in schema
                    else None
                )
                if seed is None:
                    msg = (
                        f"empty default requires a collection contract; "
                        f"{self.contract!r} has JSON schema {schema!r}"
                    )
                    raise TypeError(msg)
                self._seed = seed
                self._adapter = adapter
            return adapter.validate_json(self._seed)

    @staticmethod
    def empty[ContractT](contract: TypeForm[ContractT]) -> Callable[[], ContractT]:
        """Return the default factory of ``contract``'s canonical empty value.

        Use as ``u.Field(default_factory=u.empty(TypeForm(<annotation>)))``:
        the contract is the field's own annotation, so the default's static
        type and runtime value both derive from it. The explicit PEP 747
        ``TypeForm(...)`` conversion is what lets every type checker read a
        class-scoped alias such as ``t.JsonMapping`` as a type expression; a
        bare ``dict``/``list``/``set`` factory leaves its type parameters
        Unknown even under a declared annotation.

        Returns:
            A zero-argument factory of the contract's empty value.

        """
        return FlextModelsPydantic.EmptyDefault(contract)

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(
            pydantic.Field,
            pydantic.PrivateAttr,
        ),
    )
    class BaseModel(pydantic.BaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    class BaseSettings(pydantic_settings.BaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    class EnvSettingsSource(pydantic_settings.EnvSettingsSource):
        """Canonical environment source exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(
            pydantic.Field,
            pydantic.PrivateAttr,
        ),
    )
    class RootModel[RootValueT](pydantic.RootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    class ConfigDict(pydantic.ConfigDict, total=False):
        """Canonical model configuration exported through the models facade."""

    class SettingsConfigDict(pydantic_settings.SettingsConfigDict, total=False):
        """Canonical settings configuration exported through the models facade."""

    SkipValidation = SkipValidation
    # ``field_validator`` must re-export pydantic's real overload surface. A
    # plain-attr re-export resolves in mypy but pyright binds it as a method
    # (the first positional ``field: str`` swallows the facade class), and a
    # ``staticmethod(...)`` wrap collapses pydantic's overloads to the first
    # (wrap-only) in both checkers. The stacked ``@overload @staticmethod``
    # declarations below mirror pydantic's exact overload set for both
    # checkers, while the runtime branch keeps pydantic's own function object
    # (byte-for-byte the previous binding). This models-layer declaration is
    # the single validator owner: the utilities facade re-exports the runtime
    # function only, and the ENFORCE map routes every consumer to ``m.*``.
    if TYPE_CHECKING:

        @overload
        @staticmethod
        def field_validator[ValidatorT: FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["wrap"],
            check_fields: bool | None = ...,
            json_schema_input_type: FlextTypingBase.TypeHintSpecifier
            | PydanticUndefinedType
            | None = ...,
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @overload
        @staticmethod
        def field_validator[ValidatorT: FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["before", "plain"],
            check_fields: bool | None = ...,
            json_schema_input_type: FlextTypingBase.TypeHintSpecifier
            | PydanticUndefinedType
            | None = ...,
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @overload
        @staticmethod
        def field_validator[ValidatorT: FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["after"] = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @staticmethod
        def field_validator[ValidatorT: FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["wrap", "before", "plain", "after"] = "after",
            check_fields: bool | None = None,
            json_schema_input_type: FlextTypingBase.TypeHintSpecifier
            | PydanticUndefinedType
            | None = PydanticUndefined,
        ) -> Callable[[ValidatorT], ValidatorT]:
            """Delegate to pydantic's ``field_validator`` (type-checking only).

            Returns:
                The resulting pydantic decorator factory.

            """
            decorator_factory: Callable[
                ...,
                Callable[[ValidatorT], ValidatorT],
            ] = field_validator
            return decorator_factory(
                field,
                *fields,
                mode=mode,
                check_fields=check_fields,
                json_schema_input_type=json_schema_input_type,
            )

    else:
        field_validator = field_validator
    # Why (abstraction boundary): ENFORCE-070 makes flext-core the sole owner of
    # pydantic, and the tier-whitelist gate rejects a bare ``pydantic`` import in
    # every downstream project. A name that this facade does not re-export
    # therefore has no compliant spelling at all: the consumer must either import
    # pydantic directly (gate violation) or reach forward into a later layer
    # (namespace chain violation). ``model_validator`` is the canonical
    # model-level counterpart of ``field_validator`` above, and its absence alone
    # forced bare pydantic imports across 32 downstream modules.
    # Re-export native decorators with their complete upstream overload sets.
    model_validator = model_validator
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
    type FieldInfo = pydantic.fields.FieldInfo
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
