"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.entity import FlextModelsEntity
from flext_core._models.metadata import Metadata
from flext_core._utilities.data_mapper import FlextUtilitiesDataMapper
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.utilities import FlextUtilities

# Constants for validator logic
_MIN_QUALNAME_PARTS_FOR_WRAPPER = 2  # FlextModels.Cqrs requires at least 2 parts


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

        message_type: FlextConstants.Cqrs.CommandMessageTypeLiteral = Field(
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

        message_type: FlextConstants.Cqrs.QueryMessageTypeLiteral = Field(
            default="query",
            frozen=True,
            description="Message type discriminator",
        )

        filters: dict[str, object] = Field(default_factory=dict)
        pagination: FlextModelsCqrs.Pagination | dict[str, int] = Field(
            default_factory=dict,
        )
        query_id: str = Field(default_factory=FlextUtilities.Generators.generate_uuid)
        query_type: str | None = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: FlextModelsCqrs.Pagination | dict[str, int | str] | None,
        ) -> FlextModelsCqrs.Pagination:
            """Convert pagination to Pagination instance.

            Dynamically determines correct Pagination class based on context.
            """
            import sys  # noqa: PLC0415

            pagination_cls: type[FlextModelsCqrs.Pagination] = (
                FlextModelsCqrs.Pagination
            )

            # If cls is from models.py, get Pagination from models
            if cls.__module__ == "flext_core.models" and "." in cls.__qualname__:
                parts = cls.__qualname__.split(".")
                models_module = sys.modules.get("flext_core.models")
                if models_module and len(parts) >= _MIN_QUALNAME_PARTS_FOR_WRAPPER:
                    obj: object = getattr(models_module, parts[0], None)
                    for part in parts[1:-1]:  # Navigate to Cqrs (skip Query)
                        if obj and hasattr(obj, part):
                            obj = getattr(obj, part)
                    if obj and hasattr(obj, "Pagination"):
                        # obj is known to have "Pagination" attribute at runtime
                        # Use getattr to access safely for type checker
                        pagination_cls_attr = getattr(obj, "Pagination", None)
                        if pagination_cls_attr is not None:
                            pagination_cls = pagination_cls_attr

            # Return existing Pagination instance
            if isinstance(v, FlextModelsCqrs.Pagination):
                return v

            # Convert dict to Pagination
            if FlextRuntime.is_dict_like(v):
                v_dict = v
                page = FlextUtilitiesDataMapper.convert_to_int_safe(
                    v_dict.get("page"), 1
                )
                size = FlextUtilitiesDataMapper.convert_to_int_safe(
                    v_dict.get("size"), 20
                )
                return pagination_cls(page=page, size=size)

            # Default empty Pagination
            return pagination_cls()

        @classmethod
        def validate_query(
            cls,
            query_payload: dict[str, object],
        ) -> FlextResult[FlextModelsCqrs.Query]:
            """Validate and create Query from payload."""
            try:
                # Fast fail: filters and pagination must be dict or None
                filters_raw = query_payload.get("filters")
                filters: dict[str, object] = (
                    filters_raw if FlextRuntime.is_dict_like(filters_raw) else {}
                )
                pagination_raw = query_payload.get("pagination")
                pagination_data: dict[str, object] = (
                    pagination_raw if FlextRuntime.is_dict_like(pagination_raw) else {}
                )
                if FlextRuntime.is_dict_like(pagination_data):
                    pagination_dict = pagination_data
                    # Fast fail: page and size must be int or None
                    page_raw = pagination_dict.get("page")
                    page: int = int(page_raw) if isinstance(page_raw, (int, str)) else 1
                    size_raw = pagination_dict.get("size")
                    size: int = (
                        int(size_raw) if isinstance(size_raw, (int, str)) else 20
                    )
                    pagination: dict[str, int] = {"page": page, "size": size}
                else:
                    pagination = {"page": 1, "size": 20}
                # Fast fail: query_id must be str or None
                query_id_raw = query_payload.get("query_id")
                query_id: str = (
                    FlextUtilities.Generators.generate_uuid()
                    if query_id_raw is None
                    else str(query_id_raw)
                )
                query_type: object = query_payload.get("query_type")
                # Fast fail: filters must be dict-like
                if not FlextRuntime.is_dict_like(filters):
                    filters_dict: dict[str, object] = {}
                elif isinstance(filters, dict):
                    filters_dict = filters
                else:
                    filters_dict = (
                        dict(filters.items()) if hasattr(filters, "items") else {}
                    )
                query = cls(
                    filters=filters_dict,
                    pagination=pagination,
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return FlextResult[FlextModelsCqrs.Query].ok(query)
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return FlextResult[FlextModelsCqrs.Query].fail(
                    f"Query validation failed: {e}",
                )

    class Bus(BaseModel):
        """Bus configuration model."""

        model_config = ConfigDict(
            json_schema_extra={
                "title": "Bus",
                "description": "CQRS command bus configuration",
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
            },
        )
        handler_id: str = Field(description="Unique handler identifier")
        handler_name: str = Field(description="Human-readable handler name")
        handler_type: FlextConstants.Cqrs.HandlerType = Field(
            default=FlextConstants.Cqrs.HandlerType.COMMAND,
            description="Handler type",
        )
        handler_mode: FlextConstants.Cqrs.HandlerType = Field(
            default=FlextConstants.Cqrs.HandlerType.COMMAND,
            description="Handler mode",
        )
        command_timeout: int = Field(
            default_factory=lambda: (
                int(timeout)
                if (
                    timeout
                    := FlextConfig.get_global_instance().dispatcher_timeout_seconds
                )
                > 0
                else FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT
            ),
            description="Command timeout from FlextConfig (priority) or FlextConstants (default). Models use Config values in initialization.",
        )
        max_command_retries: int = Field(
            default_factory=lambda: (
                retries
                if (retries := FlextConfig.get_global_instance().max_retry_attempts) > 0
                else FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES
            ),
            description="Maximum retry attempts from FlextConfig (priority) or FlextConstants (default). Models use Config values in initialization.",
        )
        metadata: Metadata | None = Field(
            default=None,
            description="Handler metadata (Pydantic model)",
        )

        class ConfigParams(BaseModel):
            """Parameter object for handler configuration (reduces parameter count)."""

            model_config = ConfigDict(
                json_schema_extra={
                    "title": "HandlerConfigParams",
                    "description": "Parameter object for handler configuration",
                },
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
                handler_short_id = FlextUtilities.Generators.generate_short_id(length=8)
                self._data: dict[str, object] = {
                    "handler_type": handler_type,
                    "handler_mode": (
                        FlextConstants.Dispatcher.HANDLER_MODE_COMMAND
                        if handler_type == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
                        else FlextConstants.Dispatcher.HANDLER_MODE_QUERY
                    ),
                    "handler_id": f"{handler_type}_handler_{handler_short_id}",
                    "handler_name": f"{handler_type.title()} Handler",
                    "command_timeout": FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                    "max_command_retries": FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                    "metadata": None,
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

            def with_metadata(self, metadata: Metadata) -> Self:
                """Set metadata (fluent API - Pydantic model)."""
                self._data["metadata"] = metadata
                return self

            def merge_config(self, config: dict[str, object]) -> Self:
                """Merge additional config (fluent API)."""
                self._data.update(config)
                return self

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data)


__all__ = ["FlextModelsCqrs"]
