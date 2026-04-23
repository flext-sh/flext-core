"""Canonical builder primitives for ContractModel-backed DSLs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from typing import Self, override

from flext_core import FlextModelsBase as m, c, t


class FlextModelsBuilder:
    """Builder namespace for immutable ContractModel-backed DSLs."""

    class Builder:
        """Canonical builder DSL namespace exposed as ``m.Builder`` via MRO."""

        class Base[StateT: m.ContractModel, ProductT]:
            """Canonical builder that evolves immutable state and delegates build()."""

            state: StateT

            def __init__(self, *, state: StateT) -> None:
                self.state = state

            @property
            def state(self) -> StateT:
                """Return current immutable builder state."""
                return self.state

            def _replace(self, state: StateT) -> Self:
                """Replace current immutable state and preserve fluent chaining."""
                self.state = state
                return self

            def _set(
                self,
                **updates: t.JsonPayload | Sequence[t.JsonPayload],
            ) -> Self:
                """Apply one immutable ``model_copy(update=...)`` transition."""
                return self._replace(self.state.model_copy(update=updates))

            def _path(self, field_name: str, *parts: str) -> Self:
                """Set one tuple path field using immutable state updates."""
                return self._set(**{field_name: tuple(parts)})

            def _append(
                self,
                field_name: str,
                value: t.JsonValue,
            ) -> Self:
                """Append one value to a sequence field while preserving immutability."""
                current_values: t.VariadicTuple[t.JsonValue] = tuple(
                    getattr(self.state, field_name)
                )
                return self._set(**{field_name: (*current_values, value)})

            @staticmethod
            def _model[ModelT: m.ContractModel](
                model_type: type[ModelT],
                /,
                **data: t.JsonPayload | Sequence[t.JsonPayload],
            ) -> ModelT:
                """Build one ContractModel payload for DSL composition."""
                return model_type.model_validate(data)

            def _append_model[ModelT: m.ContractModel](
                self,
                field_name: str,
                model_type: type[ModelT],
                /,
                **data: t.JsonPayload | Sequence[t.JsonPayload],
            ) -> Self:
                """Build and append one ContractModel item to a sequence field."""
                model_item = self._model(model_type, **data)
                current_values: t.VariadicTuple[m.ContractModel] = tuple(
                    getattr(self.state, field_name)
                )
                return self._set(**{field_name: (*current_values, model_item)})

            def _build_product(self, state: StateT) -> ProductT:
                """Build one product from state. Subclasses must implement it."""
                raise NotImplementedError(c.ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED)

            def build(self) -> ProductT:
                """Build the final product from the current state."""
                return self._build_product(self.state)

        class Identity[StateT: m.ContractModel](Base[StateT, StateT]):
            """Canonical builder for DSLs whose final product is the state model."""

            @override
            def _build_product(self, state: StateT) -> StateT:
                return state


__all__: t.MutableSequenceOf[str] = ["FlextModelsBuilder"]
