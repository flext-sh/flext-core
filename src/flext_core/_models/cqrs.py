"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.entity import FlextModelsEntity
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes are accessed via FlextModels.Cqrs.* in the main models.py.
    """

    class Command(
        FlextModelsEntity.ArbitraryTypesModel,
        FlextModelsEntity.IdentifiableMixin,
        FlextModelsEntity.TimestampableMixin,
    ):
        """Base class for CQRS commands with validation."""

        message_type: Literal["command"] = Field(
            default="command",
            frozen=True,
            description="Message type discriminator for union routing - always 'command'",
        )

        command_type: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_COMMAND_TYPE,
            min_length=1,
            description="Command type identifier",
        )
        issuer_id: str | None = None

        @field_validator("command_type", mode="before")
        @classmethod
        def validate_command(cls, v: object) -> str:
            """Auto-set command type from class name if empty."""
            if isinstance(v, str):
                return v if v.strip() else cls.__name__
            if not v:
                return cls.__name__
            return str(v)

    class Pagination(BaseModel):
        """Pagination model for query results."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Pagination",
                "description": "Pagination model for query results with computed fields",
            },
        )

        page: Annotated[
            int,
            Field(
                default=FlextConstants.Pagination.DEFAULT_PAGE_NUMBER,
                ge=1,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = FlextConstants.Pagination.DEFAULT_PAGE_NUMBER
        size: Annotated[
            int,
            Field(
                default=FlextConstants.Pagination.DEFAULT_PAGE_SIZE,
                ge=1,
                le=1000,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = FlextConstants.Pagination.DEFAULT_PAGE_SIZE

        @property
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

        @property
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

    class Query(BaseModel):
        """Query model for CQRS query operations."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )

        message_type: Literal["query"] = Field(
            default="query",
            frozen=True,
            description="Message type discriminator",
        )

        filters: dict[str, object] = Field(default_factory=dict)
        pagination: FlextModelsCqrs.Pagination | dict[str, int] = Field(
            default_factory=dict
        )
        query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        query_type: str | None = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls, v: FlextModelsCqrs.Pagination | dict[str, int | str] | None
        ) -> FlextModelsCqrs.Pagination:
            """Convert pagination to Pagination instance."""
            if isinstance(v, FlextModelsCqrs.Pagination):
                return v
            if FlextRuntime.is_dict_like(v):
                v_dict = v
                page_raw = v_dict.get("page", 1)
                size_raw = v_dict.get("size", 20)
                page: int | str = page_raw if isinstance(page_raw, (int, str)) else 1
                size: int | str = size_raw if isinstance(size_raw, (int, str)) else 20
                if isinstance(page, str):
                    try:
                        page = int(page)
                    except ValueError:
                        page = 1
                if isinstance(size, str):
                    try:
                        size = int(size)
                    except ValueError:
                        size = 20
                return FlextModelsCqrs.Pagination(page=page, size=size)
            return FlextModelsCqrs.Pagination()

        @classmethod
        def validate_query(
            cls, query_payload: dict[str, object]
        ) -> FlextResult[FlextModelsCqrs.Query]:
            """Validate and create Query from payload."""
            try:
                filters: object = query_payload.get("filters", {})
                pagination_data = query_payload.get("pagination", {})
                if FlextRuntime.is_dict_like(pagination_data):
                    pagination_dict = pagination_data
                    page_raw = pagination_dict.get("page", 1)
                    size_raw = pagination_dict.get("page", 20)
                    page: int = int(page_raw) if isinstance(page_raw, (int, str)) else 1
                    size: int = (
                        int(size_raw) if isinstance(size_raw, (int, str)) else 20
                    )
                    pagination: dict[str, int] = {"page": page, "size": size}
                else:
                    pagination = {"page": 1, "size": 20}
                query_id = str(query_payload.get("query_id", str(uuid.uuid4())))
                query_type: object = query_payload.get("query_type")
                if not FlextRuntime.is_dict_like(filters):
                    filters = {}
                filters_dict = cast("dict[str, object]", filters)
                query = cls(
                    filters=filters_dict,
                    pagination=pagination,
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return FlextResult[FlextModelsCqrs.Query].ok(query)
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[FlextModelsCqrs.Query].fail(
                    f"Query validation failed: {e}"
                )

    class Bus(BaseModel):
        """Bus configuration model."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Bus",
                "description": "CQRS command bus configuration",
            }
        )
        enable_middleware: bool = Field(
            default=True, description="Enable middleware pipeline"
        )
        enable_metrics: bool = Field(
            default=True, description="Enable metrics collection"
        )
        enable_caching: bool = Field(
            default=True, description="Enable query result caching"
        )
        execution_timeout: int = Field(
            default=FlextConstants.Defaults.TIMEOUT,
            description="Command execution timeout",
        )
        max_cache_size: int = Field(default=100, description="Maximum cache size")
        implementation_path: str = Field(
            default="flext_core.bus:FlextBus",
            pattern=r"^[^:]+:[^:]+$",
            description="Implementation path",
        )

    class Handler(BaseModel):
        """Handler configuration model with Builder pattern support."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Handler",
                "description": "CQRS handler configuration",
            }
        )
        handler_id: str = Field(description="Unique handler identifier")
        handler_name: str = Field(description="Human-readable handler name")
        handler_type: FlextConstants.Cqrs.HandlerType = Field(
            default=FlextConstants.Cqrs.HandlerType.COMMAND, description="Handler type"
        )
        handler_mode: FlextConstants.Cqrs.HandlerType = Field(
            default=FlextConstants.Cqrs.HandlerType.COMMAND, description="Handler mode"
        )
        command_timeout: int = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT,
            description="Command timeout",
        )
        max_command_retries: int = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
            description="Maximum retry attempts",
        )
        metadata: dict[str, object] = Field(
            default_factory=dict, description="Handler metadata"
        )

        class ConfigParams(BaseModel):
            """Parameter object for handler configuration (reduces parameter count)."""

            model_config = ConfigDict(
                json_schema_extra={
                    "title": "HandlerConfigParams",
                    "description": "Parameter object for handler configuration",
                }
            )
            default_name: str | None = None
            default_id: str | None = None
            handler_config: dict[str, object] | None = None
            command_timeout: int = 0
            max_command_retries: int = 0

        class Builder:
            """Builder pattern for Handler (reduces 8 params to fluent API).

            Example:
                config = (Handler.Builder(handler_type="command")
                         .with_name("MyHandler")
                         .with_timeout(30)
                         .build())

            """

            def __init__(self, handler_type: FlextConstants.Cqrs.HandlerType) -> None:
                """Initialize builder with required handler_type."""
                self._data: dict[str, object] = {
                    "handler_type": handler_type,
                    "handler_mode": (
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND
                        if handler_type == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
                        else FlextConstants.Dispatcher.HANDLER_MODE_QUERY
                    ),
                    "handler_id": f"{handler_type}_handler_{uuid.uuid4().hex[:8]}",
                    "handler_name": f"{handler_type.title()} Handler",
                    "command_timeout": FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                    "max_command_retries": FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                    "metadata": {},
                }

            def with_id(self, handler_id: str) -> Self:
                """Set handler ID (fluent API)."""
                self._data["handler_id"] = handler_id
                return self

            def with_name(self, handler_name: str) -> Self:
                """Set handler name (fluent API)."""
                self._data["handler_name"] = handler_name
                return self

            def with_timeout(self, timeout: int) -> Self:
                """Set command timeout (fluent API)."""
                self._data["command_timeout"] = timeout
                return self

            def with_retries(self, max_retries: int) -> Self:
                """Set max retries (fluent API)."""
                self._data["max_command_retries"] = max_retries
                return self

            def with_metadata(self, metadata: dict[str, object]) -> Self:
                """Set metadata (fluent API)."""
                self._data["metadata"] = metadata
                return self

            def merge_config(self, config: dict[str, object]) -> Self:
                """Merge additional config (fluent API)."""
                self._data.update(config)
                return self

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data)

        @classmethod
        def create_handler_config(
            cls,
            handler_type: FlextConstants.Cqrs.HandlerType,
            *,
            params: FlextModelsCqrs.Handler.ConfigParams | None = None,
            default_name: str | None = None,
            default_id: str | None = None,
            handler_config: dict[str, object] | None = None,
            command_timeout: int = 0,
            max_command_retries: int = 0,
        ) -> FlextModelsCqrs.Handler:
            """Create handler configuration (legacy API - use Builder instead).

            DEPRECATED: Use Handler.Builder() for cleaner API.

            Example (NEW - RECOMMENDED):
                config = Handler.Builder("command").with_name("MyHandler").build()

            Example (PARAMETER OBJECT - also good):
                params = Handler.ConfigParams(default_name="MyHandler", command_timeout=30)
                config = Handler.create_handler_config("command", params=params)

            Example (OLD - backward compatible):
                config = Handler.create_handler_config("command", default_name="MyHandler")
            """
            # If params object provided, extract values (params takes precedence)
            if params is not None:
                default_name = params.default_name or default_name
                default_id = params.default_id or default_id
                handler_config = params.handler_config or handler_config
                command_timeout = params.command_timeout or command_timeout
                max_command_retries = params.max_command_retries or max_command_retries

            # Delegate to Builder for cleaner implementation
            builder = cls.Builder(handler_type)

            if default_id:
                builder.with_id(default_id)
            if default_name:
                builder.with_name(default_name)
            if command_timeout:
                builder.with_timeout(command_timeout)
            if max_command_retries:
                builder.with_retries(max_command_retries)
            if handler_config:
                builder.merge_config(handler_config)

            return builder.build()


__all__ = ["FlextModelsCqrs"]
