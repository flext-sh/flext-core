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
    ValuesView,
)
from typing import Annotated

from pydantic import Field, RootModel

from flext_core._typings.services import FlextTypesServices


class FlextModelsContainers:
    """Pydantic container models for the FLEXT type system.

    Provides RootModel-based dict and list containers for type-safe
    configuration, service maps, and validator collections.
    Access via ``t.ConfigMap``, ``t.Dict``, etc.
    """

    class ValidatorCallable(RootModel[FlextTypesServices.ValidatorCallable]):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Annotated[
            FlextTypesServices.ValidatorCallable,
            Field(
                title="Validator Callable",
                description="Callable that validates or transforms one scalar/model input value.",
                examples=["identity_validator"],
            ),
        ]

        def __call__(
            self,
            value: FlextTypesServices.ScalarOrModel | None,
        ) -> FlextTypesServices.ScalarOrModel | None:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[Mapping[str, FlextTypesServices.ValidatorCallable]],
    ):
        """Shared API for validator map containers."""

        def items(self) -> ItemsView[str, FlextTypesServices.ValidatorCallable]:
            """Get validator items."""
            validated: Mapping[str, FlextTypesServices.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.items()

        def values(self) -> ValuesView[FlextTypesServices.ValidatorCallable]:
            """Get validator values."""
            validated: Mapping[str, FlextTypesServices.ValidatorCallable] = {
                key: value for key, value in self.root.items() if callable(value)
            }
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

        root: Annotated[
            Mapping[str, FlextTypesServices.ValidatorCallable],
            Field(
                title="Field Validator Map",
                description="Field-level validators keyed by field name.",
                examples=[{"email": "validate_email"}],
            ),
        ] = Field(default_factory=dict)


__all__ = ["FlextModelsContainers"]
