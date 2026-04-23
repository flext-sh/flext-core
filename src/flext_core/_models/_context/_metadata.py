"""Context metadata and domain data models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from types import MappingProxyType
from typing import Annotated, Self

from flext_core import (
    FlextModelsBase as m,
    FlextModelsContextData,
    FlextModelsPydantic as mp,
    FlextUtilitiesPydantic as up,
    c,
    t,
)


class FlextModelsContextMetadata:
    """Namespace for context metadata models."""

    class ContextMetadata(m.FlexibleInternalModel):
        """Metadata storage for context objects with full tracing support."""

        user_id: Annotated[
            str | None,
            up.Field(default=None, description="Associated user ID"),
        ] = None
        correlation_id: Annotated[
            str | None,
            up.Field(
                default=None,
                description="Primary correlation ID for distributed tracing",
            ),
        ] = None
        parent_correlation_id: Annotated[
            str | None,
            up.Field(
                default=None,
                description="Parent request's correlation ID for nested calls",
            ),
        ] = None
        request_id: Annotated[
            str | None,
            up.Field(default=None, description="HTTP request identifier"),
        ] = None
        session_id: Annotated[
            str | None,
            up.Field(default=None, description="User session identifier"),
        ] = None
        tenant_id: Annotated[
            str | None,
            up.Field(default=None, description="Tenant/Organization ID"),
        ] = None
        handler_mode: Annotated[
            str | None,
            up.Field(default=None, description="Handler mode (command/query/event)"),
        ] = None
        message_type: Annotated[
            str | None,
            up.Field(default=None, description="Type of message being processed"),
        ] = None
        message_id: Annotated[
            str | None,
            up.Field(default=None, description="Unique message identifier"),
        ] = None
        custom_fields: Annotated[
            Mapping[str, t.RuntimeData],
            mp.BeforeValidator(
                lambda v: FlextModelsContextData.normalize_to_mapping(v)
            ),
            up.Field(
                description="Custom metadata attributes for caller-specific tracing and context.",
            ),
        ] = up.Field(default_factory=lambda: MappingProxyType({}))

        @up.model_validator(mode="after")
        def validate_context_protocol(self) -> Self:
            """Validate context instance has get() and set() methods."""
            context_field = None
            for field_name in self.__class__.model_fields:
                if c.FIELD_CONTEXT in field_name.lower():
                    context_field = getattr(self, field_name, None)
                    break
            if context_field is None:
                return self
            if hasattr(context_field, "get") and hasattr(context_field, "set"):
                return self
            raise ValueError(c.ERR_CONTEXT_FIELD_MUST_HAVE_GET_SET)

    class ContextDomainData(m.FlexibleInternalModel):
        """Domain-specific context data storage."""

        domain_name: Annotated[
            str | None,
            up.Field(default=None, description="Domain name/identifier"),
        ] = None
        domain_type: Annotated[
            str | None,
            up.Field(default=None, description="Type of domain"),
        ] = None
        domain_data: Annotated[
            Mapping[str, t.RuntimeData],
            up.Field(
                description="Domain payload values scoped to the current business context.",
            ),
        ] = up.Field(default_factory=lambda: MappingProxyType({}))
        domain_metadata: Annotated[
            t.JsonMapping,
            up.Field(
                description="Domain metadata attributes describing origin and processing state.",
            ),
        ] = up.Field(default_factory=dict)


__all__: list[str] = ["FlextModelsContextMetadata"]
