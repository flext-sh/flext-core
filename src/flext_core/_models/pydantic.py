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
from functools import partialmethod
from pathlib import Path
from re import Pattern
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    TypeAliasType,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    overload,
)

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase

import pydantic
import pydantic.fields
import pydantic_settings
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    FailFast,
    Field,
    GetPydanticSchema,
    JsonValue,
    PlainSerializer,
    PlainValidator,
    SerializeAsAny,
    SkipValidation,
    StringConstraints,
    ValidateAs,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticUndefined, PydanticUndefinedType
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import TypeForm

from flext_core._constants import FlextConstantsPydantic
from flext_core._protocols.base import FlextProtocolsBase

# Module-level alias bindings: the pydantic owner classes keep their published
# names for the facade mirrors below while the imports stay unaliased.
_PydanticConfigDict = ConfigDict
_PydanticSettingsConfigDict = SettingsConfigDict

type _FieldValue = JsonValue | Path
type _FieldSchemaExtra = Mapping[str, _FieldValue | Sequence[_FieldValue]]
# Contract for the class-member shapes ``field_validator`` decorates: the
# function shapes and the ``classmethod``/``staticmethod``/``partialmethod``
# descriptors pydantic auto-wraps. Mirrors pydantic's validator-callable union
# (``pydantic.functional_validators`` ``_V2*`` bounds) without private imports.
type _FieldValidatorCallable = (
    Callable[..., FlextProtocolsBase.AttributeProbe]
    | classmethod[type, ..., FlextProtocolsBase.AttributeProbe]
    | staticmethod[..., FlextProtocolsBase.AttributeProbe]
    | partialmethod[Callable[..., FlextProtocolsBase.AttributeProbe]]
)
type _FieldKeywordValue[DefaultT] = (
    _FieldValue
    | _FieldSchemaExtra
    | PydanticUndefinedType
    | pydantic.fields.FieldInfo
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
        *,
        default_factory: Callable[[], DefaultT] | None = None,
        **kwargs: _FieldKeywordValue[DefaultT] | None,
    ) -> DefaultT:
        """Typed FLEXT facade for ``pydantic.Field``.

        ``default_factory`` is declared explicitly (never absorbed by
        ``**kwargs``): ``dataclass_transform`` synthesis reads only the
        declared specifier parameters, so a kwargs-routed ``default_factory``
        would leave every factory-defaulted field required in the synthesized
        ``__init__``. ``None`` means absent and passes through unchanged.

        Args:
            default: Explicit default value or sentinel.
            default_factory: Zero-argument callable producing the default.
            **kwargs: Remaining ``Field`` keyword taxonomy, typed by
                ``_FieldKeywordValue``.

        Returns:
            The resulting ``DefaultT``.

        """
        field_factory: Callable[..., DefaultT] = Field
        return field_factory(default, default_factory=default_factory, **kwargs)

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
        private_attr_factory: Callable[..., PrivateT] = pydantic.PrivateAttr
        return private_attr_factory(default, default_factory=default_factory, init=init)

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
                shape = schema.get(FlextConstantsPydantic.JSON_SCHEMA_TYPE_KEY)
                seed = (
                    FlextConstantsPydantic.EMPTY_COLLECTION_JSON_SEEDS.get(shape)
                    if isinstance(shape, str)
                    and FlextConstantsPydantic.JSON_SCHEMA_PROPERTIES_KEY not in schema
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
    def _empty[ContractT](contract: TypeForm[ContractT]) -> Callable[[], ContractT]:
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
            _field,
            Field,
            pydantic.PrivateAttr,
            _private_attr,
        ),
    )
    class BaseModel(pydantic.BaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

    class BaseSettings(BaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    @dataclass_transform(
        kw_only_default=True,
        field_specifiers=(
            _field,
            Field,
            pydantic.PrivateAttr,
            _private_attr,
        ),
    )
    class RootModel[RootValueT](pydantic.RootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    class ConfigDict(_PydanticConfigDict, total=False):
        """Canonical model configuration exported through the models facade."""

    class SettingsConfigDict(_PydanticSettingsConfigDict, total=False):
        """Canonical settings configuration exported through the models facade."""

    Field = _field
    PrivateAttr = _private_attr
    empty = staticmethod(_empty)
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
        def field_validator[ValidatorT: _FieldValidatorCallable](
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
        def field_validator[ValidatorT: _FieldValidatorCallable](
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
        def field_validator[ValidatorT: _FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["after"] = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @staticmethod
        def field_validator[ValidatorT: _FieldValidatorCallable](
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
    field_serializer = field_serializer
    model_validator = model_validator
    computed_field = staticmethod(computed_field)
    model_serializer = model_serializer

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
    type PydanticBaseSettings = BaseSettings
    type PydanticBaseSettingsSource = pydantic_settings.PydanticBaseSettingsSource


__all__: list[str] = ["FlextModelsPydantic"]
