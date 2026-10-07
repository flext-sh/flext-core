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
from functools import cached_property, partialmethod
from functools import cached_property, partialmethod
from pathlib import Path
from re import Pattern
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, dataclass_transform, overload

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase

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
    field_serializer,
    field_validator,
    model_serializer,
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
# Contract for the class-member shapes ``field_validator`` decorates: the
# function shapes and the ``classmethod``/``staticmethod``/``partialmethod``
# descriptors pydantic auto-wraps. Mirrors pydantic's validator-callable union
# (``pydantic.functional_validators`` ``_V2*`` bounds) without private imports.
type FieldValidatorCallable = (
    Callable[..., Any]
    | classmethod[Any, Any, Any]
    | staticmethod[Any, Any]
    | partialmethod[Any]
)
# Contract for the class-member shapes ``field_serializer`` decorates: the
# same function/descriptor shapes pydantic auto-wraps for serializers.
type FieldSerializerCallable = (
    Callable[..., Any]
    | classmethod[Any, Any, Any]
    | staticmethod[Any, Any]
    | partialmethod[Any]
)
# Contract for the model-level callables ``model_validator`` decorates.
type ModelValidatorCallable = (
    Callable[..., Any] | classmethod[Any, Any, Any] | staticmethod[Any, Any]
)
# Contract for the class-member shapes ``computed_field`` decorates: the
# property/cached_property descriptors pydantic rewraps for serialization.
type ComputedFieldCallable = (
    Callable[..., Any] | property | cached_property[Any, Any]
)
# Contract for the model-level callables ``model_serializer`` decorates.
type ModelSerializerCallable = (
    Callable[..., Any] | classmethod[Any, Any, Any] | staticmethod[Any, Any]
)
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
            json_schema_input_type: object = ...,
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @overload
        @staticmethod
        def field_validator[ValidatorT: FieldValidatorCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["before", "plain"],
            check_fields: bool | None = ...,
            json_schema_input_type: object = ...,
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
        def field_validator(
            field: str,
            /,
            *fields: str,
            mode: Literal["wrap", "before", "plain", "after"] = "after",
            check_fields: bool | None = None,
            json_schema_input_type: object = PydanticUndefined,
        ) -> Callable[[Any], Any]:
            """Delegate to pydantic's ``field_validator`` (type-checking only).

            Returns:
                The resulting pydantic decorator factory.

            """
            decorator_factory: Callable[..., Callable[[Any], Any]] = field_validator
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
    # ``field_serializer``/``model_validator`` mirror the ``field_validator``
    # treatment: a plain-attr or ``staticmethod(...)`` re-export collapses
    # pydantic's overload set (pyright resolves only the first overload —
    # "Argument missing for parameter ``mode``" downstream), so the stacked
    # ``@overload @staticmethod`` declarations carry the exact pydantic
    # overload set for both checkers while runtime keeps pydantic's function.
    if TYPE_CHECKING:

        @overload
        @staticmethod
        def field_serializer[SerializerT: FieldSerializerCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["wrap"],
            return_type: FlextTypingBase.TypeHintSpecifier = ...,
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @overload
        @staticmethod
        def field_serializer[SerializerT: FieldSerializerCallable](
            field: str,
            /,
            *fields: str,
            mode: Literal["plain"] = ...,
            return_type: FlextTypingBase.TypeHintSpecifier = ...,
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @staticmethod
        def field_serializer(
            field: str,
            /,
            *fields: str,
            mode: Literal["plain", "wrap"] = "plain",
            return_type: Any = PydanticUndefined,
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = "always",
            check_fields: bool | None = None,
        ) -> Callable[[Any], Any]:
            """Delegate to pydantic's ``field_serializer`` (type-checking only).

            Returns:
                The resulting pydantic decorator factory.

            """
            decorator_factory: Callable[..., Callable[[Any], Any]] = field_serializer
            return decorator_factory(
                field,
                *fields,
                mode=mode,
                return_type=return_type,
                when_used=when_used,
                check_fields=check_fields,
            )

        @overload
        @staticmethod
        def model_validator[ValidatorT: ModelValidatorCallable](
            *,
            mode: Literal["wrap"],
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @overload
        @staticmethod
        def model_validator[ValidatorT: ModelValidatorCallable](
            *,
            mode: Literal["before", "after"],
        ) -> Callable[[ValidatorT], ValidatorT]: ...

        @staticmethod
        def model_validator(
            *,
            mode: Literal["wrap", "before", "after"],
        ) -> Callable[[Any], Any]:
            """Delegate to pydantic's ``model_validator`` (type-checking only).

            Returns:
                The resulting pydantic decorator factory.

            """
            decorator_factory: Callable[..., Callable[[Any], Any]] = model_validator
            return decorator_factory(mode=mode)

    else:
        field_serializer = field_serializer
        model_validator = model_validator

    # ``computed_field``/``model_serializer`` complete the typed owner surface
    # with the same treatment: a plain-attr re-export lets pyright bind the
    # facade class as the first positional argument (``computed_field`` takes
    # ``func`` positionally), and the ``staticmethod(...)`` wrap erased the
    # overload set entirely — pyrefly/pyright read the decorator as Unknown and
    # every decorated property degraded (flext-1wjg1.16/flext-8ag3t fleet
    # defect; pydantic 2.13 ships kwargs-only + exclusion overloads the wrap
    # never exposed). The stacked ``@overload @staticmethod`` declarations
    # mirror pydantic's exact overload set for both checkers while runtime
    # keeps pydantic's own function object.
    if TYPE_CHECKING:

        @overload
        @staticmethod
        def computed_field[PropertyT: ComputedFieldCallable](
            func: PropertyT,
            /,
        ) -> PropertyT: ...

        @overload
        @staticmethod
        def computed_field[PropertyT: ComputedFieldCallable](
            *,
            alias: str | None = ...,
            alias_priority: int | None = ...,
            exclude_if: Callable[[Any], bool] | None = ...,
            title: str | None = ...,
            field_title_generator: Callable[[Any, Any], str] | None = ...,
            description: str | None = ...,
            deprecated: str | bool | None = ...,
            examples: list[Any] | None = ...,
            json_schema_extra: Mapping[str, Any]
            | Callable[[Mapping[str, Any]], None]
            | None = ...,
            repr: bool = ...,
            return_type: Any = ...,
        ) -> Callable[[PropertyT], PropertyT]: ...

        @staticmethod
        def computed_field(
            func: ComputedFieldCallable | None = None,
            /,
            *,
            alias: str | None = None,
            alias_priority: int | None = None,
            exclude_if: Callable[[Any], bool] | None = None,
            title: str | None = None,
            field_title_generator: Callable[[Any, Any], str] | None = None,
            description: str | None = None,
            deprecated: str | bool | None = None,
            examples: list[Any] | None = None,
            json_schema_extra: Mapping[str, Any]
            | Callable[[Mapping[str, Any]], None]
            | None = None,
            repr: bool = True,
            return_type: Any = PydanticUndefined,
        ) -> Any:
            """Delegate to pydantic's ``computed_field`` (type-checking only).

            Returns:
                The resulting pydantic decorator or decorated descriptor.

            """
            decorator_factory: Callable[..., Any] = computed_field
            if func is None:
                return decorator_factory(
                    alias=alias,
                    alias_priority=alias_priority,
                    exclude_if=exclude_if,
                    title=title,
                    field_title_generator=field_title_generator,
                    description=description,
                    deprecated=deprecated,
                    examples=examples,
                    json_schema_extra=json_schema_extra,
                    repr=repr,
                    return_type=return_type,
                )
            return decorator_factory(func)

        @overload
        @staticmethod
        def model_serializer[SerializerT: ModelSerializerCallable](
            f: SerializerT,
            /,
        ) -> SerializerT: ...

        @overload
        @staticmethod
        def model_serializer[SerializerT: ModelSerializerCallable](
            *,
            mode: Literal["wrap"],
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = ...,
            return_type: Any = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @overload
        @staticmethod
        def model_serializer[SerializerT: ModelSerializerCallable](
            *,
            mode: Literal["plain"] = ...,
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = ...,
            return_type: Any = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @staticmethod
        def model_serializer(
            f: ModelSerializerCallable | None = None,
            /,
            *,
            mode: Literal["plain", "wrap"] = "plain",
            when_used: Literal[
                "always",
                "unless-none",
                "json",
                "json-with-timestamps",
            ] = "always",
            return_type: Any = PydanticUndefined,
        ) -> Any:
            """Delegate to pydantic's ``model_serializer`` (type-checking only).

            Returns:
                The resulting pydantic decorator or decorated callable.

            """
            decorator_factory: Callable[..., Any] = model_serializer
            if f is None:
                return decorator_factory(
                    mode=mode,
                    when_used=when_used,
                    return_type=return_type,
                )
            return decorator_factory(f)

    else:
        computed_field = computed_field
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
