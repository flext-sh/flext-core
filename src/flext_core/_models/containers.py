"""Container models for the FLEXT type system.

Pydantic RootModel-based container classes for dictionaries, lists, and validators.
These are the concrete implementations exposed through FlextModels (``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    ItemsView,
    ValuesView,
)
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

from flext_core import t


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``t.ConfigMap``, ``t.Dict``, etc.
    """

    class ErrorCodeMap(BaseModel):
        """Model for nested error code mappings."""

        codes: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                title="Error Codes",
                description="Mapping from error keys to numeric error codes.",
                examples=[{"timeout": 504, "invalid_payload": 400}],
            ),
        ]

    class ServiceMap(t.RootDictModel[type[BaseModel] | Callable[..., BaseModel]]):
        """Service registry map container. Use ``m.ServiceMap``."""

        # Used by: exported alias surface in `flext_core/models.py`; no direct runtime
        # constructor usage currently found by ast-grep. Intended payload represents
        # service classes or service factory callables, not generic container values.

        root: Annotated[
            dict[str, type[BaseModel] | Callable[..., BaseModel]],
            Field(
                default_factory=dict,
                title="Service Registry Map",
                description="Service registry entries keyed by service identifiers.",
                examples=[
                    {"user_service": "UserService", "session_factory": "build_session"},
                ],
            ),
        ]

    class ErrorMap(t.RootDictModel[int | str | BaseModel]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: Annotated[
            dict[str, int | str | BaseModel],
            Field(
                default_factory=dict,
                title="Error Map",
                description="Error catalog mapping keys to codes, messages, or nested code maps.",
                examples=[{"user_missing": 404, "bad_input": "invalid data"}],
            ),
        ]

    class FactoryMap(t.RootDictModel[t.FactoryCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: Annotated[
            dict[str, t.FactoryCallable],
            Field(
                default_factory=dict,
                title="Factory Map",
                description="Factory callables keyed by registration name.",
                examples=[{"db_client": "create_db_client"}],
            ),
        ]

    class ResourceMap(t.RootDictModel[t.ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: Annotated[
            dict[str, t.ResourceCallable],
            Field(
                default_factory=dict,
                title="Resource Map",
                description="Lifecycle resource factories keyed by resource name.",
                examples=[{"connection": "open_connection"}],
            ),
        ]

    class ValidatorCallable(RootModel[t.ValidatorCallable]):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: t.ValidatorCallable = Field(
            title="Validator Callable",
            description="Callable that validates or transforms one scalar/model input value.",
            examples=["identity_validator"],
        )

        def __call__(self, value: t.ScalarOrModel | None) -> t.ScalarOrModel | None:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(RootModel[dict[str, t.ValidatorCallable]]):
        """Shared API for validator map containers."""

        def items(self) -> ItemsView[str, t.ValidatorCallable]:
            """Get validator items."""
            validated: dict[str, t.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.items()

        def values(self) -> ValuesView[t.ValidatorCallable]:
            """Get validator values."""
            validated: dict[str, t.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: Annotated[
            dict[str, t.ValidatorCallable],
            Field(
                default_factory=dict,
                title="Field Validator Map",
                description="Field-level validators keyed by field name.",
                examples=[{"email": "validate_email"}],
            ),
        ]

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: Annotated[
            dict[str, t.ValidatorCallable],
            Field(
                default_factory=dict,
                title="Consistency Rule Map",
                description="Consistency rule callables keyed by rule identifier.",
                examples=[{"order_total": "validate_order_total"}],
            ),
        ]

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: Annotated[
            dict[str, t.ValidatorCallable],
            Field(
                default_factory=dict,
                title="Event Validator Map",
                description="Event validator callables keyed by event type or alias.",
                examples=[{"user.created": "validate_user_created"}],
            ),
        ]

    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )
        results: Annotated[
            list[t.Scalar | None],
            Field(
                default_factory=list,
                title="Batch Results",
                description="Batch result values in processing order.",
                examples=[["ok", 1, None]],
            ),
        ]

        errors: Annotated[
            list[tuple[int, str]],
            Field(
                default_factory=list,
                title="Batch Errors",
                description="Batch error tuples as (index, message).",
                examples=[[(0, "invalid payload")]],
            ),
        ]
        total: Annotated[
            int,
            Field(
                default=0,
                title="Total Items",
                description="Total number of batch items processed.",
                examples=[10],
            ),
        ] = 0
        success_count: Annotated[
            int,
            Field(
                default=0,
                title="Success Count",
                description="Number of batch items processed successfully.",
                examples=[8],
            ),
        ] = 0
        error_count: Annotated[
            int,
            Field(
                default=0,
                title="Error Count",
                description="Number of batch items that failed with errors.",
                examples=[2],
            ),
        ] = 0


__all__ = ["FlextModelsContainers"]
