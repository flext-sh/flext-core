"""Container models for the FLEXT type system.

Pydantic RootModel-based container classes for dictionaries, lists, and validators.
These are the concrete implementations exposed through FlextModels (``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    ItemsView,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from typing import Annotated

from pydantic import BaseModel, Field, RootModel

from flext_core import FlextModelFoundation, t


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``t.ConfigMap``, ``t.Dict``, etc.
    """

    class ErrorCodeMap(FlextModelFoundation.FlexibleInternalModel):
        """Model for nested error code mappings."""

        codes: Annotated[
            Mapping[str, int],
            Field(
                title="Error Codes",
                description="Mapping from error keys to numeric error codes.",
                examples=[{"timeout": 504, "invalid_payload": 400}],
            ),
        ] = Field(default_factory=dict)

    class ErrorMap(t.RootDictModel[int | str | BaseModel]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: Annotated[
            MutableMapping[str, int | str | BaseModel],
            Field(
                title="Error Map",
                description="Error catalog mapping keys to codes, messages, or nested code maps.",
                examples=[{"user_missing": 404, "bad_input": "invalid data"}],
            ),
        ] = Field(default_factory=dict)

    class ValidatorCallable(RootModel[t.ValidatorCallable]):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Annotated[
            t.ValidatorCallable,
            Field(
                title="Validator Callable",
                description="Callable that validates or transforms one scalar/model input value.",
                examples=["identity_validator"],
            ),
        ]

        def __call__(self, value: t.ScalarOrModel | None) -> t.ScalarOrModel | None:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(RootModel[Mapping[str, t.ValidatorCallable]]):
        """Shared API for validator map containers."""

        def items(self) -> ItemsView[str, t.ValidatorCallable]:
            """Get validator items."""
            validated: Mapping[str, t.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.items()

        def values(self) -> ValuesView[t.ValidatorCallable]:
            """Get validator values."""
            validated: Mapping[str, t.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: Annotated[
            Mapping[str, t.ValidatorCallable],
            Field(
                title="Field Validator Map",
                description="Field-level validators keyed by field name.",
                examples=[{"email": "validate_email"}],
            ),
        ] = Field(default_factory=dict)

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: Annotated[
            Mapping[str, t.ValidatorCallable],
            Field(
                title="Consistency Rule Map",
                description="Consistency rule callables keyed by rule identifier.",
                examples=[{"order_total": "validate_order_total"}],
            ),
        ] = Field(default_factory=dict)

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: Annotated[
            Mapping[str, t.ValidatorCallable],
            Field(
                title="Event Validator Map",
                description="Event validator callables keyed by event type or alias.",
                examples=[{"user.created": "validate_user_created"}],
            ),
        ] = Field(default_factory=dict)

    class BatchResultDict(FlextModelFoundation.ArbitraryTypesModel):
        """Result payload model for batch operation outputs."""

        results: Annotated[
            Sequence[t.Scalar | None],
            Field(
                title="Batch Results",
                description="Batch result values in processing order.",
                examples=[["ok", 1, None]],
            ),
        ] = Field(default_factory=list)

        errors: Annotated[
            Sequence[tuple[int, str]],
            Field(
                title="Batch Errors",
                description="Batch error tuples as (index, message).",
                examples=[[(0, "invalid payload")]],
            ),
        ] = Field(default_factory=list)
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
            t.NonNegativeInt,
            Field(
                default=0,
                title="Success Count",
                description="Number of batch items processed successfully.",
                examples=[8],
            ),
        ] = 0
        error_count: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                title="Error Count",
                description="Number of batch items that failed with errors.",
                examples=[2],
            ),
        ] = 0


__all__ = ["FlextModelsContainers"]
