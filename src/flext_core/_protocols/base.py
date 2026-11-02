"""FlextProtocolsBase - foundational protocol primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from flext_core import p


from collections.abc import Callable, Iterable, Mapping, MutableSequence
from typing import Literal, Protocol, Self, overload, runtime_checkable

from typing_extensions import TypeForm

from flext_core import t

if TYPE_CHECKING:
    from types import TracebackType

    from pydantic.config import JsonDict
    from pydantic.fields import Deprecated
    from pydantic.functional_serializers import (
        FieldPlainSerializer,
        FieldWrapSerializer,
        ModelPlainSerializer,
        ModelWrapSerializer,
    )
    from pydantic_core.core_schema import WhenUsed


class FlextProtocolsBase:
    """Hierarchical protocol namespace organized by Interface Segregation Principle."""

    @runtime_checkable
    class FieldSerializerFactory(Protocol):
        """Native Pydantic field serializer overloads, including callback bounds."""

        @overload
        def __call__[SerializerT: FieldWrapSerializer, ReturnT](
            self,
            field: str,
            /,
            *fields: str,
            mode: Literal["wrap"],
            return_type: TypeForm[ReturnT] = ...,
            when_used: WhenUsed = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @overload
        def __call__[SerializerT: FieldPlainSerializer, ReturnT](
            self,
            field: str,
            /,
            *fields: str,
            mode: Literal["plain"] = ...,
            return_type: TypeForm[ReturnT] = ...,
            when_used: WhenUsed = ...,
            check_fields: bool | None = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

    @runtime_checkable
    class ModelSerializerFactory(Protocol):
        """Native Pydantic bare, plain and wrap model serializer overloads."""

        @overload
        def __call__[SerializerT: ModelPlainSerializer](
            self, f: SerializerT, /
        ) -> SerializerT: ...

        @overload
        def __call__[SerializerT: ModelWrapSerializer, ReturnT](
            self,
            *,
            mode: Literal["wrap"],
            when_used: WhenUsed = "always",
            return_type: TypeForm[ReturnT] = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

        @overload
        def __call__[SerializerT: ModelPlainSerializer, ReturnT](
            self,
            *,
            mode: Literal["plain"] = ...,
            when_used: WhenUsed = "always",
            return_type: TypeForm[ReturnT] = ...,
        ) -> Callable[[SerializerT], SerializerT]: ...

    @runtime_checkable
    class ComputedFieldFactory(Protocol):
        """Native Pydantic computed-field overloads and keyword contracts."""

        @overload
        def __call__[PropertyT](self, func: PropertyT, /) -> PropertyT: ...

        @overload
        def __call__[PropertyT, ExcludeT, ExampleT](
            self,
            *,
            alias: str | None = None,
            alias_priority: int | None = None,
            exclude_if: Callable[[ExcludeT], bool] | None = None,
            title: str | None = None,
            field_title_generator: Callable[[str, t.ComputedFieldInfo], str]
            | None = None,
            description: str | None = None,
            deprecated: Deprecated | str | bool | None = None,
            examples: list[ExampleT] | None = None,
            json_schema_extra: JsonDict | Callable[[JsonDict], None] | None = None,
            repr: bool = True,
            # This native parameter accepts an unconstrained schema descriptor.
            return_type: FlextProtocolsBase.AttributeProbe = ...,
        ) -> Callable[[PropertyT], PropertyT]: ...

    @runtime_checkable
    class Base(Protocol):
        """Base protocol for FLEXT structural types."""

    @runtime_checkable
    class AttributeProbe(Protocol):
        """Structural marker for values inspected only via ``hasattr``/``getattr``.

        Empty Protocol body — accepts any runtime value structurally. Used by
        ``FlextDecoratorsRuntime`` and other infrastructure helpers as a typed
        parameter annotation when the actual contract is "any value that
        may carry inspectable attributes" without loose top-type annotations.
        """

    @runtime_checkable
    class ModuleOwned(Protocol):
        """Structural contract for runtime symbols that expose owner module."""

        __module__: str

    @runtime_checkable
    class ModelCopy(Protocol):
        """Copy a validated model without constraining its serialized payload."""

        def model_copy(
            self,
            *,
            update: Mapping[str, object] | None = None,
            deep: bool = False,
        ) -> Self:
            """Copy through the native model's update and depth contract."""
            ...

    @runtime_checkable
    class Model[DumpT: t.JsonPayload = t.JsonDict](Base, ModelCopy, Protocol):
        """Structural typing protocol for Pydantic v2 models.

        Ensures types have Pydantic signatures without importing BaseModel directly
        in typings.py, preventing circular dependencies.
        """

        # NOTE (multi-agent, mro-wkii.17 / agent: codex): declare only the
        # instance capability consumed by interfaces; Pydantic class APIs stay
        # on ``ModelType``.

        def model_dump(
            self,
            *,
            mode: str = "python",
            exclude: set[str] | None = None,
            by_alias: bool | None = None,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
        ) -> DumpT:
            """Dump the validated model at an external serialization boundary."""
            ...

        def model_dump_json(
            self,
            *,
            indent: int | None = None,
            exclude: set[str] | None = None,
            by_alias: bool | None = None,
            exclude_none: bool = False,
            round_trip: bool = False,
        ) -> str:
            """Serialize through the native JSON options structural consumers use."""
            ...

    @runtime_checkable
    class Routable(Protocol):
        """Base protocol for messages that carry explicit route information."""

    @runtime_checkable
    class CommandRoutable(Routable, Protocol):
        """Protocol for command messages."""

        command_type: str
        """Command type identifier."""

    @runtime_checkable
    class EventRoutable(Routable, Protocol):
        """Protocol for event messages."""

        event_type: str
        """Event type identifier."""

    @runtime_checkable
    class QueryRoutable(Routable, Protocol):
        """Protocol for query messages."""

        query_type: str
        """Query type identifier."""

    @runtime_checkable
    class ConfigObject(Protocol):
        """Protocol for mapping-like configuration payloads."""

        def get(
            self,
            key: str,
            default: t.JsonPayload | None = None,
        ) -> t.JsonPayload | None:
            """Fetch a configuration value by key."""
            ...

        def keys(self) -> Iterable[str]:
            """Return known configuration keys."""
            ...

        def items(self) -> Iterable[tuple[str, t.JsonPayload]]:
            """Return key/value entries for configuration payloads."""
            ...

    @runtime_checkable
    class Flushable(Protocol):
        """Protocol for objects with a flush() method."""

        def flush(self) -> None: ...

    @runtime_checkable
    class Lock(Protocol):
        """Protocol for lock-like synchronization primitives."""

        def __enter__(self) -> bool:
            """Context manager entry."""
            ...

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            value: BaseException | None,
            traceback: TracebackType | None,
            /,
        ) -> None:
            """Context manager exit."""
            ...

    @runtime_checkable
    class HasDomainEvents(Protocol):
        """Protocol for DDD aggregate roots that buffer uncommitted domain events.

        Every entity that carries a ``domain_events`` list and an identity
        satisfies this protocol. Use with ``u.add_domain_event`` utility.
        """

        unique_id: str
        domain_events: MutableSequence[p.DomainEvent]

    @runtime_checkable
    class EnforcedModel(Model, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ManagedModel(EnforcedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class EnumManagedModel(ManagedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class NormalizedModel(EnumManagedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class StrictManagedModel(NormalizedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class StrictModel(StrictManagedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class FrozenModel(StrictModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ArbitraryTypesModel(ManagedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class StrictBoundaryModel(FrozenModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class FlexibleInternalModel(NormalizedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ImmutableValueModel(ManagedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class TaggedModel(EnforcedModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class FlexibleModel(ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class DynamicModel(FlexibleModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class FrozenDynamicModel(DynamicModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class Metadata(Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def created_at(self) -> datetime: ...
        @property
        def updated_at(self) -> datetime: ...
        @property
        def version(self) -> str: ...
        @property
        def created_by(self) -> str | None: ...
        @property
        def modified_by(self) -> str | None: ...
        @property
        def tags(self) -> Sequence[str]: ...
        @property
        def attributes(self) -> t.JsonMapping: ...
        @property
        def metadata_value(self) -> t.Scalar | None: ...

    @runtime_checkable
    class ContractModel(StrictModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class ExternalContractModel(ContractModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class FrozenValueModel(ContractModel, Protocol):
        """Structural data contract projected from the owning model."""

    @runtime_checkable
    class RetryConfigurationMixin(Model, Protocol):
        """Structural data contract projected from the owning model."""

        max_retries: t.NonNegativeInt
        initial_delay_seconds: t.PositiveFloat
        max_delay_seconds: t.PositiveFloat

    @runtime_checkable
    class TimestampedModel(ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""


__all__: list[str] = ["FlextProtocolsBase"]
