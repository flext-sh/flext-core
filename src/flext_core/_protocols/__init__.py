"""Internal protocol modules for flext_core."""

from __future__ import annotations

from flext_core._protocols.typed_protocols import (
    FlextTypedProtocols,
    SupportsConfigAccess,
    SupportsLogging,
    SupportsSerialization,
)

__all__ = [
    "FlextTypedProtocols",
    "SupportsConfigAccess",
    "SupportsLogging",
    "SupportsSerialization",
]
