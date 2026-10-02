"""Canonical builder primitives for ContractModel-backed DSLs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._models.base import FlextModelsBase as m
from flext_core._typings.base import FlextTypingBase as tb

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices as ts


class FlextModelsBuilder:
    """Builder namespace for immutable ContractModel-backed DSLs.

    Exposed flat on ``m`` via MRO (``m.Base``/``m.Identity``) after the
    namespace-holder flattening (operator decision 2026-09-24).
    """

    class Base[StateT: m.ContractModel, ProductT]:
        """Canonical builder that evolves immutable state and delegates build()."""

        state: StateT

        def __init__(self, *, state: StateT) -> None:
            self.state = state

        def _replace(self, state: StateT) -> Self:
            """Replace current immutable state and preserve fluent chaining.

            Returns:
                The resulting ``Self``.

            """
            self.state = state
            return self

        def _set(
            self,
            **updates: ts.JsonPayload
            | tb.SequenceOf[ts.JsonPayload]
            | tb.SequenceOf[m.ContractModel],
        ) -> Self:
            """Apply one immutable ``model_copy(update=...)`` transition.

            Returns:
                The resulting ``Self``.

            """
            return self._replace(self.state.model_copy(update=updates))

        def _path(self, field_name: str, *parts: str) -> Self:
            """Set one tuple path field using immutable state updates.

            Returns:
                The resulting ``Self``.

            """
            return self._set(**{field_name: tuple(parts)})

        def _append(self, field_name: str, value: tb.JsonValue) -> Self:
            """Append one value to a sequence field while preserving immutability.

            Returns:
                The resulting ``Self``.

            """
            current_values: tb.VariadicTuple[tb.JsonValue] = tuple(
                getattr(self.state, field_name),
            )
            return self._set(**{field_name: (*current_values, value)})

        @staticmethod
        def _model[ModelT: m.ContractModel](
            model_type: type[ModelT],
            /,
            **data: ts.JsonPayload | tb.SequenceOf[ts.JsonPayload],
        ) -> ModelT:
            """Build one ContractModel payload for DSL composition.

            Returns:
                The resulting ``ModelT``.

            """
            model: ModelT = model_type.model_validate(data)
            return model

        def _append_model[ModelT: m.ContractModel](
            self,
            field_name: str,
            model_type: type[ModelT],
            /,
            **data: ts.JsonPayload | tb.SequenceOf[ts.JsonPayload],
        ) -> Self:
            """Build and append one ContractModel item to a sequence field.

            Returns:
                The resulting ``Self``.

            """
            model_item = self._model(model_type, **data)
            current_values: tb.VariadicTuple[m.ContractModel] = tuple(
                getattr(self.state, field_name),
            )
            updated: tb.SequenceOf[m.ContractModel] = (*current_values, model_item)
            return self._set(**{field_name: updated})

        def _build_product(self, state: StateT) -> ProductT:
            """Build one product from state. Subclasses must implement it."""
            msg = (
                f"{ce.ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED}: "
                f"{type(state).__name__}"
            )
            raise NotImplementedError(msg)

        def build(self) -> ProductT:
            """Build the final product from the current state.

            Returns:
                The resulting ``ProductT``.

            """
            return self._build_product(self.state)

    class Identity[StateT: m.ContractModel](Base[StateT, StateT]):
        """Canonical builder for DSLs whose final product is the state model."""

        @override
        def _build_product(self, state: StateT) -> StateT:
            return state


__all__: tb.MutableSequenceOf[str] = ["FlextModelsBuilder"]
