"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t
from flext_core.utilities import u


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or directly via FlextModelsCqrs.*
    """

    class Command(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.IdentifiableMixin,
        FlextModelsBase.TimestampableMixin,
    ):
        """Base class for CQRS commands with validation."""

        message_type: c.Cqrs.CommandMessageTypeLiteral = Field(
            default=c.Cqrs.HandlerType.COMMAND,
            frozen=True,
            description="Message type discriminator (always 'command')",
        )

        command_type: str = Field(
            default_factory=lambda: c.Cqrs.DEFAULT_COMMAND_TYPE,
            min_length=1,
            description="Command type identifier",
        )
        issuer_id: str | None = None

        @field_validator("command_type", mode="before")
        @classmethod
        def validate_command(cls, v: t.GeneralValueType) -> str:
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
                default=c.Pagination.DEFAULT_PAGE_NUMBER,
                ge=1,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_NUMBER
        size: Annotated[
            int,
            Field(
                default=c.Pagination.DEFAULT_PAGE_SIZE,
                ge=1,
                le=1000,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = c.Pagination.DEFAULT_PAGE_SIZE

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

        # Constants for internal use
        _MIN_QUALNAME_PARTS_FOR_WRAPPER: ClassVar[int] = 2  # Requires at least 2 parts

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )

        message_type: c.Cqrs.QueryMessageTypeLiteral = Field(
            default=c.Cqrs.HandlerType.QUERY,
            frozen=True,
            description="Message type discriminator",
        )

        filters: t.Types.ConfigurationMapping = Field(default_factory=dict)
        pagination: FlextModelsCqrs.Pagination | dict[str, int] = Field(
            default_factory=dict,
        )
        query_id: str = Field(default_factory=lambda: u.generate("query"))
        query_type: str | None = None

        @classmethod
        def _resolve_pagination_class(
            cls: type[FlextModelsCqrs.Query],
        ) -> type[FlextModelsCqrs.Pagination]:
            """Resolve correct Pagination class based on context."""
            if cls.__module__ != "flext_core.models" or "." not in cls.__qualname__:
                return FlextModelsCqrs.Pagination
            parts = cls.__qualname__.split(".")
            models_module = sys.modules.get("flext_core.models")
            # Use constant value directly - attribute is on Pagination class, not on type
            min_parts = 2
            if not models_module or len(parts) < min_parts:
                return FlextModelsCqrs.Pagination
            obj: t.GeneralValueType | None = getattr(
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
                    and issubclass(pagination_cls_attr, FlextModelsCqrs.Pagination)
                ):
                    # Type-safe narrowing: pagination_cls_attr is confirmed as subclass
                    result_cls: type[FlextModelsCqrs.Pagination] = pagination_cls_attr
                    return result_cls
            return FlextModelsCqrs.Pagination

        @staticmethod
        def _convert_dict_to_pagination(
            v: dict[str, int | str],
            pagination_cls: type[FlextModelsCqrs.Pagination],
        ) -> FlextModelsCqrs.Pagination:
            """Convert dict to Pagination instance."""
            page = u.convert(u.get(v, "page", default=1) or 1, int, default=1)
            size = u.convert(u.get(v, "size", default=20) or 20, int, default=20)
            return pagination_cls(page=page, size=size)

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: FlextModelsCqrs.Pagination | dict[str, int | str] | None,
        ) -> FlextModelsCqrs.Pagination:
            """Convert pagination to Pagination instance."""
            pagination_cls = cls._resolve_pagination_class()
            if isinstance(v, FlextModelsCqrs.Pagination):
                return v

            # Convert dict to Pagination
            if FlextRuntime.is_dict_like(v):
                # TypeGuard narrows v to Mapping[str, GeneralValueType]
                v_dict: t.Types.ConfigurationMapping = v
                # .get() returns GeneralValueType | None, pass directly (None is valid GeneralValueType)
                page = u.convert(u.get(v_dict, "page", default=1) or 1, int, default=1)
                size = u.convert(
                    u.get(v_dict, "size", default=20) or 20, int, default=20
                )
                return pagination_cls(page=page, size=size)

            # Default empty Pagination
            return pagination_cls()

        @classmethod
        def validate_query(
            cls,
            query_payload: t.Types.ConfigurationMapping,
        ) -> r[FlextModelsCqrs.Query]:
            """Validate and create Query from payload."""
            try:
                # Fast fail: filters and pagination must be dict or None
                filters_raw = query_payload.get("filters")
                # TypeGuard narrows to Mapping[str, GeneralValueType] when is_dict_like is True
                filters: t.Types.ConfigurationMapping = (
                    filters_raw if FlextRuntime.is_dict_like(filters_raw) else {}
                )
                pagination_raw = query_payload.get("pagination")
                # TypeGuard narrows to Mapping[str, GeneralValueType] when is_dict_like is True
                pagination_data: t.Types.ConfigurationMapping = (
                    pagination_raw if FlextRuntime.is_dict_like(pagination_raw) else {}
                )
                if FlextRuntime.is_dict_like(pagination_data):
                    pagination_dict = pagination_data
                    # Use parse() for concise type conversion
                    page = u.convert(
                        u.get(pagination_dict, "page", default=1) or 1, int, default=1
                    )
                    size = u.convert(
                        u.get(pagination_dict, "size", default=20) or 20,
                        int,
                        default=20,
                    )
                    pagination: dict[str, int] = {"page": page, "size": size}
                else:
                    pagination = {"page": 1, "size": 20}
                # Fast fail: query_id must be str or None
                query_id_raw = query_payload.get("query_id")
                query_id: str = (
                    u.generate("uuid") if query_id_raw is None else str(query_id_raw)
                )
                query_type: t.GeneralValueType | None = query_payload.get(
                    "query_type",
                )
                # filters is already guaranteed to be ConfigurationMapping from earlier validation
                query = cls(
                    filters=filters,
                    pagination=pagination,
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return r[FlextModelsCqrs.Query].ok(query)
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                return r[FlextModelsCqrs.Query].fail(
                    f"Query validation failed: {e}",
                )

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
        max_cache_size: int = Field(default=100, description="Maximum cache size")
        implementation_path: str = Field(
            default="flext_core.dispatcher:FlextDispatcher",
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
        metadata: FlextModelsBase.Metadata | None = Field(
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
            handler_config: t.Types.ConfigurationMapping | None = None
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
                handler_short_id = u.generate("id", length=8)
                self._data: dict[str, t.GeneralValueType] = {
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

            def with_metadata(self, metadata: FlextModelsBase.Metadata) -> Self:
                """Set metadata (fluent API - Pydantic model)."""
                # Convert Metadata model to dict for GeneralValueType compatibility
                metadata_dict: dict[str, t.GeneralValueType] = dict(
                    metadata.model_dump().items()
                )
                self._data["metadata"] = metadata_dict
                return self

            def merge_config(
                self,
                config: t.Types.ConfigurationMapping,
            ) -> Self:
                """Merge additional config (fluent API)."""
                self._data.update(config)
                return self

            def build(self) -> FlextModelsCqrs.Handler:
                """Build and validate Handler instance."""
                return FlextModelsCqrs.Handler.model_validate(self._data)


__all__ = ["FlextModelsCqrs"]
