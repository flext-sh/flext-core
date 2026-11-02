# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/operation_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextProtocolsExceptionParamsOperationErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class OperationErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def operation(self) -> str | None:
            """Operation name associated with the failure."""
            ...

        @property
        def reason(self) -> str | None:
            """Short reason explaining the operation failure."""
            ...

    @runtime_checkable
    class ServiceLookupParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def expected_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Expected runtime type name for the failing value."""
            ...

        @property
        def actual_type(self) -> FlextTypesPydantic.StrictStr | None:
            """Actual runtime type name received at runtime."""
            ...

        @property
        def service_name(self) -> str | None:
            """Service identifier requested from the container/registry."""
            ...

    @runtime_checkable
    class RegistryPluginParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def category(self) -> str | None:
            """Registry category for plugin registration."""
            ...

        @property
        def name(self) -> str | None:
            """Plugin name within a category."""
            ...

        @property
        def scope(self) -> str | None:
            """Registration scope used by registry operations."""
            ...

    @runtime_checkable
    class AttributeAccessErrorParams(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def attribute_name(self) -> str | None:
            """Attribute name that could not be accessed safely."""
            ...

        @property
        def attribute_context(self) -> FlextTypingBase.RuntimeData | None:
            """Context payload describing the state during access failure."""
            ...


__all__: list[str] = [
    "FlextProtocolsExceptionParamsOperationErrors",
]
