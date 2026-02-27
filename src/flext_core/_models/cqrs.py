"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_validator,
)

from flext_core._models.base import FlextModelFoundation
from flext_core.constants import c
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(BaseModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal["command"]] = "command"

        message_type: Literal["command"] = Field(
            default="command",
            frozen=True,
            description="Message type discriminator (always 'command')",
        )

        command_type: str = Field(
            default=c.Cqrs.DEFAULT_COMMAND_TYPE,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Command type identifier",
        )
        command_id: str = Field(
            default_factory=lambda: FlextRuntime.generate_prefixed_id("cmd"),
        )
        issuer_id: str | None = None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for commands)."""
            return None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for commands)."""
            return None

    class Pagination(BaseModel):
        """Pagination model for query results."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Pagination",
                "description": (
                    "Pagination model for query results with computed fields"
                ),
            },
        )

        page: Annotated[
            int,
            Field(
                default=c.Pagination.DEFAULT_PAGE_NUMBER,
                ge=c.Reliability.RETRY_COUNT_MIN,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_NUMBER
        size: Annotated[
            int,
            Field(
                default=c.Pagination.DEFAULT_PAGE_SIZE,
                ge=c.Reliability.RETRY_COUNT_MIN,
                le=c.Pagination.MAX_PAGE_SIZE_EXAMPLE,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_SIZE

        @computed_field
        @property
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

        @computed_field
        @property
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

    class Query(BaseModel):
        """Query model for CQRS query operations."""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )

        tag: ClassVar[Literal["query"]] = "query"

        message_type: Literal["query"] = Field(
            default="query",
            frozen=True,
            description="Message type discriminator",
        )

        filters: t.Dict = Field(default_factory=t.Dict)
        pagination: FlextModelsCqrs.Pagination | t.Dict = Field(
            default_factory=t.Dict,
        )
        query_id: str = Field(
            default_factory=lambda: FlextRuntime.generate_prefixed_id("query"),
        )
        query_type: str | None = None

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for queries)."""
            return None

        @property
        def event_type(self) -> str | None:
            """Event type identifier (always None for queries)."""
            return None

        @classmethod
        def _resolve_pagination_class(
            cls: type[FlextModelsCqrs.Query],
        ) -> type[FlextModelsCqrs.Pagination]:
            """Resolve correct Pagination class based on context."""
            if cls.__module__ != "flext_core.models" or "." not in cls.__qualname__:
                return FlextModelsCqrs.Pagination
            parts = cls.__qualname__.split(".")
            models_module = sys.modules.get("flext_core.models")
            min_qualname_parts = 2
            if not models_module or len(parts) < min_qualname_parts:
                return FlextModelsCqrs.Pagination
            obj: t.GuardInputValue | None = getattr(
                models_module,
                parts[0],
                None,
            )
            for part in parts[1:-1]:
                if obj and hasattr(obj, part):
                    obj = getattr(obj, part)
            if obj and hasattr(obj, "Pagination"):
                pagination_cls_attr = getattr(obj, "Pagination", None)
                if (
                    pagination_cls_attr is not None
                    and isinstance(pagination_cls_attr, type)
                    and FlextModelsCqrs.Pagination in pagination_cls_attr.__mro__
                ):
                    # Type-safe narrowing: pagination_cls_attr is confirmed as subclass
                    result_cls: type[FlextModelsCqrs.Pagination] = pagination_cls_attr
                    return result_cls
            return FlextModelsCqrs.Pagination

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: FlextModelsCqrs.Pagination | t.Dict | Mapping[str, t.ScalarValue] | None,
        ) -> FlextModelsCqrs.Pagination:
            """Convert pagination to Pagination instance."""
            pagination_cls = cls._resolve_pagination_class()
            adapter: TypeAdapter[
                FlextModelsCqrs.Pagination | t.Dict | Mapping[str, t.ScalarValue] | None
            ] = TypeAdapter(
                FlextModelsCqrs.Pagination
                | t.Dict
                | Mapping[str, t.ScalarValue]
                | None,
            )
            parsed_input = adapter.validate_python(v)
            if parsed_input is None:
                return pagination_cls()

            payload: BaseModel | Mapping[str, t.MetadataAttributeValue] | str
            if isinstance(parsed_input, t.Dict):
                payload = parsed_input.root
            elif isinstance(parsed_input, FlextModelsCqrs.Pagination):
                payload = parsed_input.model_dump()
            else:
                payload = dict(parsed_input)

            try:
                return pagination_cls.model_validate(payload)
            except ValidationError:
                return pagination_cls()

    class Bus(BaseModel):
        """Dispatcher configuration model for CQRS routing."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Dispatcher",
                "description": "CQRS dispatcher configuration",
            },
        )
        enable_middleware: bool = Field(
            default=True,
            description="Enable middleware pipeline",
        )
        enable_metrics: bool = Field(
            default=True,
            description="Enable metrics collection",
        )
        enable_caching: bool = Field(
            default=True,
            description="Enable query result caching",
        )
        execution_timeout: int = Field(
            default=c.Defaults.TIMEOUT,
            description="Command execution timeout",
        )
        max_cache_size: int = Field(
            default=c.Defaults.DEFAULT_MAX_CACHE_SIZE,
            description="Maximum cache size",
        )
        implementation_path: str = Field(
            default=c.Dispatcher.DEFAULT_DISPATCHER_PATH,
            pattern=c.Platform.PATTERN_MODULE_PATH,
            description="Implementation path",
        )

    class Handler(BaseModel):
        """Handler configuration model with Builder pattern support."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            },
        )
        handler_id: str = Field(description="Unique handler identifier")
        handler_name: str = Field(description="Human-readable handler name")
        handler_type: c.Cqrs.HandlerType = Field(
            default=c.Cqrs.HandlerType.COMMAND,
            description="Handler type",
        )
        handler_mode: c.Cqrs.HandlerType = Field(
            default=c.Cqrs.HandlerType.COMMAND,
            description="Handler mode",
        )
        command_timeout: int = Field(
            default=c.Cqrs.DEFAULT_COMMAND_TIMEOUT,
            description="Command timeout from c (default). Models use Config values in initialization.",
        )
        max_command_retries: int = Field(
            default=c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
            description="Maximum retry attempts from c (default). Models use Config values in initialization.",
        )
        metadata: FlextModelFoundation.Metadata | None = Field(
            default=None,
            description="Handler metadata (Pydantic model)",
        )

        class ConfigParams(BaseModel):
            """Parameter object for handler configuration (reduces parameter count)."""

            model_config = ConfigDict(
                arbitrary_types_allowed=True,
                json_schema_extra={
                    "title": "HandlerConfigParams",
                    "description": "Parameter object for handler configuration",
                },
            )
            default_name: str | None = None
            default_id: str | None = None
            handler_config: t.ConfigMap | None = None
            command_timeout: int = 0
            max_command_retries: int = 0

        class Builder:
            """Builder pattern for Handler (reduces 8 params to fluent API).

            Example:
                config = (Handler.Builder(handler_type=c.Cqrs.HandlerType.COMMAND)
                         .with_name("MyHandler")
                         .with_timeout(30)
                         .build())

            """

            def __init__(self, handler_type: c.Cqrs.HandlerType) -> None:
                """Initialize builder with required handler_type."""
                super().__init__()
                handler_short_id = FlextRuntime.generate_prefixed_id("", length=8)
                self._data: t.Dict = t.Dict(
                    root={
                        "handler_type": handler_type,
                        "handler_mode": (
                            c.Dispatcher.HANDLER_MODE_COMMAND
                            if handler_type == c.Cqrs.HandlerType.COMMAND
                            else c.Dispatcher.HANDLER_MODE_QUERY
                        ),
                        "handler_id": f"{handler_type}_handler_{handler_short_id}",
                        "handler_name": f"{handler_type.title()} Handler",
                        "command_timeout": c.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                        "max_command_retries": c.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                        "metadata": None,
                    },
                )

            def with_id(self, handler_id: str) -> Self:
                """Set handler ID (fluent API)."""
                self._data.root["handler_id"] = handler_id
                return self

            def with_name(self, handler_name: str) -> Self:
                """Set handler name (fluent API)."""
                self._data.root["handler_name"] = handler_name
                return self

            def with_timeout(self, timeout: int) -> Self:
                """Set command timeout (fluent API)."""
                self._data.root["command_timeout"] = timeout
                return self

            def with_retries(self, max_retries: int) -> Self:
                """Set max retries (fluent API)."""
                self._data.root["max_command_retries"] = max_retries
                return self

            def with_metadata(self, metadata: FlextModelFoundation.Metadata) -> Self:
                """Set metadata (fluent API - Pydantic model)."""
                self._data.root["metadata"] = metadata
                return self

            def merge_config(
                self,
                config: t.ConfigMap,
            ) -> Self:
                """Merge additional config (fluent API)."""
                self._data.root.update(config.root)
                return self

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data.root)

    class Event(BaseModel):
        """Event model for CQRS event operations.

        Events represent domain events that occur as a result of command execution.
        They are immutable records of what happened in the system.
        """

        tag: ClassVar[Literal["event"]] = "event"

        message_type: Literal["event"] = Field(
            default="event",
            frozen=True,
            description="Message type discriminator (always 'event')",
        )

        event_type: str = Field(
            description="Event type identifier",
        )

        @property
        def command_type(self) -> str | None:
            """Command type identifier (always None for events)."""
            return None

        @property
        def query_type(self) -> str | None:
            """Query type identifier (always None for events)."""
            return None

        aggregate_id: str = Field(
            description="ID of the aggregate that generated this event",
        )
        event_id: str = Field(
            default_factory=lambda: FlextRuntime.generate_prefixed_id("evt"),
        )
        data: t.Dict = Field(
            default_factory=t.Dict,
            description="Event payload data",
        )
        metadata: t.Dict = Field(
            default_factory=t.Dict,
            description="Event metadata (timestamps, correlation IDs, etc.)",
        )

    type FlextMessage = Annotated[
        Command | Query | Event,
        Discriminator("message_type"),
    ]

    @staticmethod
    def parse_message(payload: t.ConfigMapValue) -> FlextMessage:
        """Parse a message payload into a FlextMessage instance."""
        msg = "parse_message must be implemented by subclasses"
        raise NotImplementedError(msg)

    class HandlerBatchRegistrationResult(BaseModel):
        """Result of batch handler registration."""

        status: str
        count: int
        handlers: list[str]


__all__ = ["FlextModelsCqrs"]
