"""CQRS patterns extracted from FlextModels.

This module contains the FlextModelsCqrs class with all CQRS-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Cqrs instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
)

from flext_core import c, t
from flext_core._models.base import FlextModelsBase as m
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as ur,
)
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug


class _CqrsPagination(m.FlexibleInternalModel):
    """Pagination model for query results.

    Defined at module level so it can be referenced in Query annotations
    without forward-reference issues (Pydantic can resolve it statically).
    Exposed as FlextModelsCqrs.Pagination.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={
            "title": "Pagination",
            "description": "Pagination model for query results with computed fields",
        },
    )
    page: Annotated[
        t.PositiveInt,
        Field(
            description="Page number (1-based indexing)",
            examples=[1, 2, 10, 100],
        ),
    ] = c.DEFAULT_RETRY_DELAY_SECONDS
    size: Annotated[
        t.PositiveInt,
        Field(
            le=c.MAX_PAGE_SIZE,
            description="Number of items per page (max 1000)",
            examples=[10, 20, 50, 100],
        ),
    ] = c.DEFAULT_PAGE_SIZE

    @computed_field
    @property
    def limit(self) -> int:
        """Get limit (same as size)."""
        return self.size

    @computed_field
    @property
    def offset(self) -> int:
        """Calculate offset from page and size."""
        return (self.page - 1) * self.size


class FlextModelsCqrs:
    """CQRS pattern container class.

    This class acts as a namespace container for CQRS patterns.
    All nested classes can be accessed via FlextModels.Cqrs.* (type aliases) or
    directly via FlextModelsCqrs.*
    """

    class Command(m.ArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        tag: ClassVar[Literal["command"]] = "command"
        message_type: Annotated[
            Literal["command"],
            Field(
                frozen=True,
                description="Message type discriminator (always 'command')",
            ),
        ] = "command"
        command_type: Annotated[
            t.NonEmptyStr,
            Field(
                description="Command type identifier",
            ),
        ] = c.DEFAULT_COMMAND_TYPE
        command_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique command identifier used for tracing and idempotency checks.",
                title="Command Id",
                examples=["cmd_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: ug.generate_prefixed_id("cmd"),
        )
        issuer_id: Annotated[
            t.NonEmptyStr | None,
            Field(
                description="Identity of the principal that issued this command.",
            ),
        ] = None

    Pagination = _CqrsPagination

    class Query(m.ArbitraryTypesModel):
        """Query model for CQRS query operations."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations",
            },
        )
        tag: ClassVar[Literal["query"]] = "query"
        message_type: Annotated[
            Literal["query"],
            Field(
                frozen=True,
                description="Message type discriminator",
            ),
        ] = "query"
        filters: Annotated[
            t.MappingKV[str, t.Scalar],
            Field(
                description="Filter values that restrict which records are returned by the query.",
                title="Query Filters",
                examples=[{"status": "active", "tenant": "acme"}],
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        pagination: Annotated[
            _CqrsPagination,
            Field(
                description="Pagination settings controlling page number and page size for query results.",
                title="Pagination",
                examples=[{"page": 1, "size": 50}],
            ),
        ] = Field(default_factory=_CqrsPagination)
        query_id: Annotated[
            t.NonEmptyStr,
            Field(
                description="Unique query identifier used for tracing and cache correlation.",
                title="Query Id",
                examples=["query_01HZX7Q0P5N6M2"],
            ),
        ] = Field(
            default_factory=lambda: ug.generate_prefixed_id(
                "query",
            ),
        )
        query_type: Annotated[
            str | None,
            Field(
                description="Query type identifier for dispatcher routing.",
            ),
        ] = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls,
            v: BaseModel | t.MappingKV[str, t.Scalar] | None,
        ) -> BaseModel:
            """Convert pagination to Pagination instance."""
            # Allow subclasses to override Pagination via class attribute,
            # fallback to the default _CqrsPagination
            pagination_cls: type[BaseModel] = getattr(
                cls,
                "Pagination",
                _CqrsPagination,
            )
            normalized_input = ur.normalize_model_input_mapping(v)
            if normalized_input is None:
                return pagination_cls()
            try:
                return pagination_cls.model_validate(normalized_input)
            except ValidationError:
                return pagination_cls()


__all__: list[str] = ["FlextModelsCqrs"]
