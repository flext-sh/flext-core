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
            if isinstance(v, dict):
                v_dict = cast("dict[str, object]", v)
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
                if isinstance(pagination_data, dict):
                    pagination_dict = cast("dict[str, object]", pagination_data)
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
                if not isinstance(filters, dict):
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
        """Handler configuration model."""

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

        @classmethod
        def create_handler_config(
            cls,
            handler_type: FlextConstants.Cqrs.HandlerType,
            *,
            default_name: str | None = None,
            default_id: str | None = None,
            handler_config: dict[str, object] | None = None,
            command_timeout: int = 0,
            max_command_retries: int = 0,
        ) -> Self:
            """Create handler configuration."""
            handler_mode_value = (
                FlextConstants.Dispatcher.HANDLER_MODE_COMMAND
                if handler_type == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
                else FlextConstants.Dispatcher.HANDLER_MODE_QUERY
            )
            config_data: dict[str, object] = {
                "handler_id": default_id
                or f"{handler_type}_handler_{uuid.uuid4().hex[:8]}",
                "handler_name": default_name or f"{handler_type.title()} Handler",
                "handler_type": handler_type,
                "handler_mode": handler_mode_value,
                "command_timeout": command_timeout,
                "max_command_retries": max_command_retries,
                "metadata": {},
            }
            if handler_config:
                config_data.update(handler_config)
            return cls.model_validate(config_data)


__all__ = ["FlextModelsCqrs"]
