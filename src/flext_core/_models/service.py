"""Domain service patterns extracted from FlextModels.

This module contains the FlextModelsService class with all domain service-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Service instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from typing import Annotated

from pydantic import Field, field_validator

from flext_core._models.base import FlextModelsBase
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsService:
    """Domain service pattern container class.

    This class acts as a namespace container for domain service patterns.
    All nested classes are accessed via FlextModels.Service.* in the main models.py.
    """

    class DomainServiceExecutionRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service execution request with advanced validation."""

        service_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Service name",
        )
        method_name: str = Field(
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Method to execute",
        )
        parameters: t.ConfigurationDict = Field(default_factory=dict)
        context: t.ConfigurationDict = Field(default_factory=dict)
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextSettings (Config has priority over Constants)",
        )
        execution: bool = False
        enable_validation: bool = True

        @field_validator("context", mode="before")
        @classmethod
        def validate_context(cls, v: t.GeneralValueType) -> t.StringDict:
            """Ensure context has required fields (using FlextRuntime).

            Returns t.StringDict because ensure_trace_context generates string trace IDs.
            This is compatible with the field type t.ConfigurationDict since str is a subtype.
            """
            # Normalize input to dict
            if v is None:
                context_dict: t.StringDict = {}
            elif FlextRuntime.is_dict_like(v) and isinstance(v, dict):
                context_dict = {k: str(val) for k, val in v.items()}
            else:
                context_dict = {}

            # Ensure trace_id and span_id exist
            if "trace_id" not in context_dict:
                context_dict["trace_id"] = str(uuid.uuid4())
            if "span_id" not in context_dict:
                context_dict["span_id"] = str(uuid.uuid4())
            return context_dict

        @field_validator("timeout_seconds", mode="after")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            """Validate timeout is reasonable (using FlextRuntime)."""
            max_timeout_seconds = c.Performance.MAX_TIMEOUT_SECONDS
            if v <= 0:
                msg = "Timeout must be positive"
                raise ValueError(msg)
            if v > max_timeout_seconds:
                msg = f"Timeout {v}s exceeds maximum {max_timeout_seconds}s"
                raise ValueError(msg)
            return v

    class DomainServiceBatchRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: list[t.ConfigurationDict] = Field(
            default_factory=list,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            max_length=c.Performance.MAX_BATCH_OPERATIONS,
        )
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default=c.Performance.MAX_BATCH_SIZE,
            description="Batch size from FlextSettings (Config has priority over Constants)",
        )
        timeout_per_operation: float = Field(
            default=c.Defaults.TIMEOUT,
            description="Timeout per operation from FlextSettings (Config has priority over Constants)",
        )

    class DomainServiceMetricsRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[c.Cqrs.ServiceMetricTypeLiteral],
            Field(
                default_factory=lambda: list(c.Cqrs.DEFAULT_METRIC_CATEGORIES),
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = c.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: str = Field(
            default_factory=lambda: c.Cqrs.Aggregation.AVG,
        )
        group_by: list[str] = Field(default_factory=list)
        filters: t.ConfigurationDict = Field(default_factory=dict)

    class DomainServiceResourceRequest(FlextModelsBase.ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = c.Dispatcher.DEFAULT_SERVICE_NAME
        resource_type: str = Field(
            c.Dispatcher.DEFAULT_RESOURCE_TYPE,
            pattern=c.Platform.PATTERN_IDENTIFIER,
        )
        resource_id: str | None = None
        resource_limit: int = Field(c.Performance.MAX_BATCH_SIZE, gt=c.ZERO)
        action: str = Field(default_factory=lambda: c.Cqrs.Action.GET)
        data: t.ConfigurationDict = Field(default_factory=dict)
        filters: t.ConfigurationDict = Field(default_factory=dict)

    class AclResponse(FlextModelsBase.ArbitraryTypesModel):
        """ACL (Access Control List) response model."""

        resource: str = Field(description="Resource identifier")
        user: str = Field(description="User identifier")
        action: str = Field(description="Requested action")
        allowed: bool = Field(description="Whether access is allowed")
        permissions: list[str] = Field(
            default_factory=list,
            description="Granted permissions",
        )
        denied_permissions: list[str] = Field(
            default_factory=list,
            description="Denied permissions",
        )
        context: t.ConfigurationDict = Field(
            default_factory=dict,
            description="Additional context",
        )

    class OperationExecutionRequest(FlextModelsBase.ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str = Field(
            max_length=c.Performance.MAX_OPERATION_NAME_LENGTH,
            min_length=c.Reliability.RETRY_COUNT_MIN,
            description="Operation name",
        )
        operation_callable: p.VariadicCallable[p.ResultLike[t.GeneralValueType]]
        arguments: t.ConfigurationDict = Field(default_factory=dict)
        keyword_arguments: t.ConfigurationDict = Field(
            default_factory=dict,
        )
        timeout_seconds: float = Field(
            default=c.Defaults.TIMEOUT,
            gt=c.ZERO,
            le=c.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextSettings (Config has priority over Constants)",
        )
        retry_config: t.ConfigurationDict = Field(
            default_factory=dict,
        )

        @field_validator("operation_callable", mode="after")
        @classmethod
        def validate_operation_callable(
            cls,
            v: p.VariadicCallable[p.ResultLike[t.GeneralValueType]],
        ) -> p.VariadicCallable[p.ResultLike[t.GeneralValueType]]:
            """Validate operation is callable.

            With mode="after", Pydantic has already validated v as VariadicCallable type.
            Protocol types are always callable, so no additional check needed.
            """
            # v is already validated as VariadicCallable by Pydantic
            # Protocol types guarantee callable interface
            return v


__all__ = ["FlextModelsService"]
