"""Shared deprecation warning helpers for _utilities modules.

This module provides centralized deprecation warning utilities
for internal modules that should be accessed via the FlextUtilities facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from typing import Final

# Approved modules that can import directly (for testing, internal use)
APPROVED_MODULES: Final[frozenset[str]] = frozenset({
    "flext_core.utilities",
    "flext_core._utilities",
    "tests.",
})


def warn_direct_module_access(
    module_name: str, facade_method: str | None = None
) -> None:
    """Warn if internal module is accessed from non-approved caller.

    Args:
        module_name: Name of the internal module being accessed (e.g., "conversion")
        facade_method: Optional facade method suggestion (e.g., "u.conversion(...)")
                      If not provided, defaults to "u.{ModuleName}"

    Example:
        >>> warn_direct_module_access("conversion", "u.conversion(...)")
        # Emits: "Direct import from _utilities.conversion is deprecated.
        #         Use 'from flext_core import u; u.conversion(...)' instead."

    """
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        caller_module = frame.f_back.f_back.f_globals.get("__name__", "")
        if not any(caller_module.startswith(approved) for approved in APPROVED_MODULES):
            # Build the suggestion message
            if facade_method is None:
                # Convert module_name to title case for class name
                facade_method = f"u.{module_name.title()}"

            warnings.warn(
                f"Direct import from _utilities.{module_name} is deprecated. "
                f"Use 'from flext_core import u; {facade_method}' instead.",
                DeprecationWarning,
                stacklevel=4,
            )


__all__ = [
    "APPROVED_MODULES",
    "warn_direct_module_access",
]
