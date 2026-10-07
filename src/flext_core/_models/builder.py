"""Canonical builder primitives for ContractModel-backed DSLs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

from flext_core._constants import FlextConstantsErrors
from flext_core._models.base import FlextModelsBase
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices


class FlextModelsBuilder:
    """Builder namespace for immutable ContractModel-backed DSLs.

    Exposed flat on ``m`` via MRO (``m.Base``/``m.Identity``) after the
    namespace-holder flattening (operator decision 2026-09-24).
    """

    class Base[StateT: FlextModelsBase.ContractModel, ProductT]:
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
            **updates: FlextTypesServices.JsonPayload
            | FlextTypingBase.SequenceOf[FlextTypesServices.JsonPayload]
            | FlextTypingBase.SequenceOf[FlextModelsBase.ContractModel],
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

        def _append(self, field_name: str, value: FlextTypingBase.JsonValue) -> Self:
            """Append one value to a sequence field while preserving immutability.

            Returns:
                The resulting ``Self``.

            """
            current_values: FlextTypingBase.VariadicTuple[FlextTypingBase.JsonValue] = (
                tuple(
                    getattr(self.state, field_name),
                )
            )
            return self._set(**{field_name: (*current_values, value)})

        @staticmethod
        def _model[ModelT: FlextModelsBase.ContractModel](
            model_type: type[ModelT],
            /,
            **data: FlextTypesServices.JsonPayload
            | FlextTypingBase.SequenceOf[FlextTypesServices.JsonPayload],
        ) -> ModelT:
            """Build one ContractModel payload for DSL composition.

            Returns:
                The resulting ``ModelT``.

            """
            model: ModelT = model_type.model_validate(data)
            return model

        def _append_model[ModelT: FlextModelsBase.ContractModel](
            self,
            field_name: str,
            model_type: type[ModelT],
            /,
            **data: FlextTypesServices.JsonPayload
            | FlextTypingBase.SequenceOf[FlextTypesServices.JsonPayload],
        ) -> Self:
            """Build and append one ContractModel item to a sequence field.

            Returns:
                The resulting ``Self``.

            """
            model_item = self._model(model_type, **data)
            current_values: FlextTypingBase.VariadicTuple[
                FlextModelsBase.ContractModel
            ] = tuple(
                getattr(self.state, field_name),
            )
            updated: FlextTypingBase.SequenceOf[FlextModelsBase.ContractModel] = (
                *current_values,
                model_item,
            )
            return self._set(**{field_name: updated})

        def _build_product(self, state: StateT) -> ProductT:
            """Build one product from state. Subclasses must implement it."""
            msg = (
                f"{FlextConstantsErrors.ERR_BUILDER_BUILD_PRODUCT_NOT_IMPLEMENTED}: "
                f"{type(state).__name__}"
            )
            raise NotImplementedError(msg)

        def build(self) -> ProductT:
            """Build the final product from the current state.

            Returns:
                The resulting ``ProductT``.

            """
            return self._build_product(self.state)

    class Identity[StateT: FlextModelsBase.ContractModel](Base[StateT, StateT]):
        """Canonical builder for DSLs whose final product is the state model."""

        @override
        def _build_product(self, state: StateT) -> StateT:
            return state


__all__: FlextTypingBase.MutableSequenceOf[str] = ["FlextModelsBuilder"]
