# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/operation_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._exception_params import FlextProtocolsExceptionParamsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypingBase


class FlextProtocolsExceptionParamsOperationErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class OperationErrorParams(FlextProtocolsExceptionParamsBase.ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        operation: str | None
        reason: str | None

    @runtime_checkable
    class ServiceLookupParams(
        FlextProtocolsExceptionParamsBase.ExpectedActualTypeParams, Protocol
    ):
        """Structural data contract projected from the owning model."""

        service_name: str | None

    @runtime_checkable
    class RegistryPluginParams(FlextProtocolsExceptionParamsBase.ParamsModel, Protocol):
        """Structural data contract projected from the owning model."""

        category: str | None
        name: str | None
        scope: str | None

    @runtime_checkable
    class AttributeAccessErrorParams(
        FlextProtocolsExceptionParamsBase.ParamsModel, Protocol
    ):
        """Structural data contract projected from the owning model."""

        attribute_name: str | None

        @property
        def attribute_context(self) -> FlextTypingBase.RuntimeData | None:
            """Context payload describing the state during access failure."""
            ...


__all__: list[str] = [
    "FlextProtocolsExceptionParamsOperationErrors",
]
