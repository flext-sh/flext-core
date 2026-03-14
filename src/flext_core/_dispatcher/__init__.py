# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Expose dispatcher reliability and timeout helpers.

This package houses helper classes previously nested within
``FlextDispatcher`` so orchestration code remains concise while retaining
identical behavior and typed exports for downstream consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._dispatcher.config import FlextModelsConfig
    from flext_core._dispatcher.reliability import (
        CircuitBreakerManager,
        RateLimiterManager,
        RetryPolicy,
    )
    from flext_core._dispatcher.timeout import TimeoutEnforcer
    from flext_core.typings import FlextTypes

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CircuitBreakerManager": (
        "flext_core._dispatcher.reliability",
        "CircuitBreakerManager",
    ),
    "FlextModelsConfig": ("flext_core._dispatcher.config", "FlextModelsConfig"),
    "RateLimiterManager": ("flext_core._dispatcher.reliability", "RateLimiterManager"),
    "RetryPolicy": ("flext_core._dispatcher.reliability", "RetryPolicy"),
    "TimeoutEnforcer": ("flext_core._dispatcher.timeout", "TimeoutEnforcer"),
}

__all__ = [
    "CircuitBreakerManager",
    "FlextModelsConfig",
    "RateLimiterManager",
    "RetryPolicy",
    "TimeoutEnforcer",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
