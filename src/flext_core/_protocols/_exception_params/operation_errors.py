# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_exception_params/operation_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t


class FlextProtocolsExceptionParamsOperationErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class OperationErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        operation: str | None
        reason: str | None

    @runtime_checkable
    class ServiceLookupParams(Protocol):
        """Structural data contract projected from the owning model."""

        expected_type: t.StrictStr | None
        actual_type: t.StrictStr | None
        service_name: str | None

    @runtime_checkable
    class RegistryPluginParams(Protocol):
        """Structural data contract projected from the owning model."""

        category: str | None
        name: str | None
        scope: str | None

    @runtime_checkable
    class AttributeAccessErrorParams(Protocol):
        """Structural data contract projected from the owning model."""

        attribute_name: str | None
        attribute_context: t.RuntimeData | None


__all__: list[str] = [
    "FlextProtocolsExceptionParamsOperationErrors",
]
