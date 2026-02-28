"""Expose dispatcher reliability and timeout helpers.

This package houses helper classes previously nested within
``FlextDispatcher`` so orchestration code remains concise while retaining
identical behavior and typed exports for downstream consumers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes, FlextTypes as t, m
    from flext_core._dispatcher.reliability import (
        CircuitBreakerManager,
        RateLimiterManager,
        RetryPolicy,
    )
    from flext_core._dispatcher.timeout import TimeoutEnforcer

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CircuitBreakerManager": (
        "flext_core._dispatcher.reliability",
        "CircuitBreakerManager",
    ),
    "FlextTypes": ("flext_core", "FlextTypes"),
    "RateLimiterManager": ("flext_core._dispatcher.reliability", "RateLimiterManager"),
    "RetryPolicy": ("flext_core._dispatcher.reliability", "RetryPolicy"),
    "TimeoutEnforcer": ("flext_core._dispatcher.timeout", "TimeoutEnforcer"),
    "m": ("flext_core", "m"),
    "t": ("flext_core", "FlextTypes"),
}

__all__ = [
    "CircuitBreakerManager",
    "FlextTypes",
    "RateLimiterManager",
    "RetryPolicy",
    "TimeoutEnforcer",
    "m",
    "t",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
