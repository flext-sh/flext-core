"""Compatibility re-export module for protocol interfaces."""

from __future__ import annotations

# Re-export selected protocols for compatibility
from flext_core.protocols import FlextConfigurable, FlextValidator

__all__: list[str] = [
    "FlextConfigurable",
    "FlextValidator",
]
