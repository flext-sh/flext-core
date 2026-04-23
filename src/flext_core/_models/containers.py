"""Container RootModel wrappers for the FLEXT runtime.

Pydantic v2 ``RootModel`` + ``MutableMapping`` / ``Sequence`` ABCs.  Only
the abstract hooks required by the ABCs are implemented - ``items``,
``keys``, ``values``, ``get``, ``update``, ``__contains__`` etc. come for
free from the ABCs.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterator,
    MutableMapping,
    Sequence,
)
from typing import Annotated, no_type_check, override

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.services import FlextTypesServices
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up


@no_type_check
class FlextModelsContainers:
    """Pydantic RootModel container namespace."""

    @no_type_check
    class ValidatorCallable(mp.RootModel[FlextTypesServices.ValidatorCallable]):
        """Callable validator container rooted in a scalar-or-model transform."""

        root: Annotated[
            FlextTypesServices.ValidatorCallable,
            up.Field(
                title="Validator Callable",
                description="Callable that validates or transforms one scalar/model input value.",
                examples=["identity_validator"],
            ),
        ]

        def __call__(
            self,
            value: FlextTypesServices.ScalarOrModel,
        ) -> FlextTypesServices.ScalarOrModel:
            return self.root(value)

    @no_type_check
    class Dict(
        mp.RootModel[dict[str, FlextTypesServices.RuntimeData]],
        MutableMapping[str, FlextTypesServices.RuntimeData],
    ):
        """Runtime dictionary container rooted in validated values."""

        root: Annotated[
            dict[str, FlextTypesServices.RuntimeData],
            up.Field(description="Validated runtime key-value mapping."),
        ]

        @override
        def __iter__(self) -> Iterator[str]:
            return iter(self.root)

        @override
        def __len__(self) -> int:
            return len(self.root)

        @override
        def __getitem__(self, key: str) -> FlextTypesServices.RuntimeData:
            return self.root[key]

        @override
        def __setitem__(self, key: str, value: FlextTypesServices.RuntimeData) -> None:
            self.root[key] = value

        @override
        def __delitem__(self, key: str) -> None:
            del self.root[key]

    @no_type_check
    class ConfigMap(
        mp.RootModel[dict[str, FlextTypesServices.RuntimeData]],
        MutableMapping[str, FlextTypesServices.RuntimeData],
    ):
        """Runtime configuration mapping rooted in validated values."""

        root: Annotated[
            dict[str, FlextTypesServices.RuntimeData],
            up.Field(description="Validated runtime configuration mapping."),
        ]

        @override
        def __iter__(self) -> Iterator[str]:
            return iter(self.root)

        @override
        def __len__(self) -> int:
            return len(self.root)

        @override
        def __getitem__(self, key: str) -> FlextTypesServices.RuntimeData:
            return self.root[key]

        @override
        def __setitem__(self, key: str, value: FlextTypesServices.RuntimeData) -> None:
            self.root[key] = value

        @override
        def __delitem__(self, key: str) -> None:
            del self.root[key]

    @no_type_check
    class ObjectList(
        mp.RootModel[list[FlextTypesServices.RuntimeData]],
        Sequence[FlextTypesServices.RuntimeData],
    ):
        """Runtime list container rooted in validated values."""

        root: Annotated[
            list[FlextTypesServices.RuntimeData],
            up.Field(description="Validated runtime sequence."),
        ]

        @override
        def __iter__(self) -> Iterator[FlextTypesServices.RuntimeData]:
            return iter(self.root)

        @override
        def __len__(self) -> int:
            return len(self.root)

        @override
        def __getitem__(self, index: int) -> FlextTypesServices.RuntimeData:
            return self.root[index]


__all__: list[str] = ["FlextModelsContainers"]
