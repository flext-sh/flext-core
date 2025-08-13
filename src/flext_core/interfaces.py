"""Compatibility re-export module for protocol interfaces.

This thin module exists solely for backward-compatibility with tests and external
code that import `flext_core.interfaces`. All interfaces have been consolidated
in `flext_core.protocols`. We re-export a small curated set used by tests.
"""

from __future__ import annotations

# Re-export selected protocols for compatibility
from flext_core.protocols import FlextConfigurable, FlextValidator

__all__: list[str] = [
    "FlextConfigurable",
    "FlextValidator",
]
