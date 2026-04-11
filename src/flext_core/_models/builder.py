"""Canonical builder primitives for ContractModel-backed DSLs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Self, override

from flext_core import FlextModelsBase, c


class FlextModelsBuilder:
    """Builder namespace for immutable ContractModel-backed DSLs."""

    class Builder:
        """Canonical builder DSL namespace exposed as ``m.Builder`` via MRO."""

        class Base[StateT: FlextModelsBase.ContractModel, ProductT]:
            """Canonical builder that evolves immutable state and delegates build()."""

            _state: StateT

            def __init__(self, *, state: StateT) -> None:
                self._state = state

            @property
            def state(self) -> StateT:
                """Return current immutable builder state."""
                return self._state

            def _replace(self, state: StateT) -> Self:
                """Replace current immutable state and preserve fluent chaining."""
                self._state = state
                return self

            def _set(self, **updates: object) -> Self:
                """Apply one immutable ``model_copy(update=...)`` transition."""
                return self._replace(self._state.model_copy(update=updates))

            def _path(self, field_name: str, *parts: str) -> Self:
                """Set one tuple path field using immutable state updates."""
                return self._set(**{field_name: tuple(parts)})

            def _append(self, field_name: str, value: object) -> Self:
                """Append one value to a sequence field while preserving immutability."""
                current_values = tuple(getattr(self._state, field_name))
                return self._set(**{field_name: (*current_values, value)})

            @staticmethod
            def _model[ModelT: FlextModelsBase.ContractModel](
                model_type: type[ModelT],
                /,
                **data: object,
            ) -> ModelT:
                """Build one ContractModel payload for DSL composition."""
                return model_type.model_validate(data)

            def _append_model[ModelT: FlextModelsBase.ContractModel](
                self,
                field_name: str,
                model_type: type[ModelT],
                /,
                **data: object,
            ) -> Self:
                """Build and append one ContractModel item to a sequence field."""
                return self._append(field_name, self._model(model_type, **data))

            def _build_product(self, state: StateT) -> ProductT:
                """Build one product from state. Subclasses must implement it."""
                raise NotImplementedError(c.ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED)

            def build(self) -> ProductT:
                """Build the final product from the current state."""
                return self._build_product(self._state)

        class Identity[StateT: FlextModelsBase.ContractModel](Base[StateT, StateT]):
            """Canonical builder for DSLs whose final product is the state model."""

            @override
            def _build_product(self, state: StateT) -> StateT:
                return state


__all__ = ["FlextModelsBuilder"]
