"""Deprecation utilities for FLEXT ecosystem.

This module provides centralized deprecation warning functionality
with once-per-key tracking to prevent warning spam.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from typing import ClassVar


class FlextUtilitiesDeprecation:
    """Centralized deprecation warning system with once-per-key tracking.

    This class provides a mechanism to emit deprecation warnings that will
    only be shown once per unique key, preventing warning spam in loops or
    repeated calls while still alerting users to deprecated patterns.

    Usage:
        from flext_core.utilities import u
        u.Deprecation.warn_once(
            "subclass:FlextLdifModels",
            "Subclassing FlextLdifModels is deprecated. Use FlextModels.Ldif."
        )
    """

    _warned: ClassVar[set[str]] = set()

    @classmethod
    def warn_once(cls, key: str, message: str) -> None:
        """Emit a deprecation warning once per unique key.

        Args:
            key: Unique identifier for this deprecation (e.g., "subclass:ClassName").
            message: The deprecation warning message to display.

        Note:
            The warning is only emitted once per unique key. Subsequent calls
            with the same key will be silently ignored.

        """
        if key not in cls._warned:
            cls._warned.add(key)
            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=3,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the warned keys set (primarily for testing).

        This method clears all previously warned keys, allowing warnings
        to be emitted again. Useful for testing deprecation warnings.
        """
        cls._warned.clear()

    @classmethod
    def _generate_approved_modules(
        cls,
        facade_module: str | None = None,
        internal_module: str | None = None,
    ) -> frozenset[str]:
        """Generate approved modules list automatically.

        Args:
            facade_module: Public facade module name (e.g., "flext_core.models").
                          If None, will be auto-detected from caller's module.
            internal_module: Internal module name (e.g., "flext_core._models").
                            If None, will be auto-detected from caller's module.

        Returns:
            Frozenset of approved module prefixes including:
            - facade_module (public facade)
            - internal_module (internal directory)
            - tests. (always included)

        """
        approved = {"tests."}  # Always include tests

        # Auto-detect from caller if not provided
        if facade_module is None or internal_module is None:
            caller_module = cls._get_caller_module()
            if caller_module:
                detected = cls._detect_modules_from_caller(caller_module)
                if facade_module is None and detected["facade"]:
                    facade_module = detected["facade"]
                if internal_module is None and detected["internal"]:
                    internal_module = detected["internal"]

        if facade_module:
            approved.add(facade_module)
        if internal_module:
            approved.add(internal_module)

        return frozenset(approved)

    @classmethod
    def _get_caller_module(cls) -> str | None:
        """Get the caller's module name by inspecting the call stack.

        Goes back through the stack to find the actual caller, skipping:
        - This helper method
        - _generate_approved_modules
        - is_approved_module
        """
        frame = inspect.currentframe()
        if not frame:
            return None

        # Skip: _get_caller_module, _generate_approved_modules, is_approved_module
        # So we need to go back 3 frames
        current = frame
        for _ in range(3):
            if current and current.f_back:
                current = current.f_back
            else:
                return None

        if current:
            return current.f_globals.get("__name__", "")
        return None

    @classmethod
    def _detect_modules_from_caller(cls, caller_module: str) -> dict[str, str | None]:
        """Detect facade and internal modules from caller module name.

        Args:
            caller_module: Full module name of the caller (e.g., "flext_core._models.container").

        Returns:
            Dictionary with "facade" and "internal" keys containing detected module names.

        """
        parts = caller_module.split(".")
        if not parts:
            return {"facade": None, "internal": None}

        # Find internal directory (part starting with "_")
        internal_idx = None
        for i, part in enumerate(parts):
            if part.startswith("_"):
                internal_idx = i
                break

        if internal_idx is not None:
            # We're in an internal directory
            # Facade: remove "_" prefix from internal directory name
            # e.g., "flext_core._models.container" -> facade: "flext_core.models"
            facade_parts = parts[:internal_idx] + [
                parts[internal_idx][1:]
            ]  # Remove leading "_"
            facade = ".".join(facade_parts)

            # Internal: use up to and including the internal directory
            # e.g., "flext_core._models.container" -> internal: "flext_core._models"
            internal = ".".join(parts[: internal_idx + 1])
            return {"facade": facade, "internal": internal}

        # Not in an internal directory, use package root as facade
        return {"facade": parts[0], "internal": None}

    @classmethod
    def is_approved_module(
        cls,
        module_name: str,
        facade_module: str | None = None,
        internal_module: str | None = None,
    ) -> bool:
        """Check if a module is approved for direct internal access.

        Args:
            module_name: The full module name to check.
            facade_module: Public facade module name (e.g., "flext_core.models").
                          If None, will be auto-detected from caller's module.
            internal_module: Internal module name (e.g., "flext_core._models").
                            If None, will be auto-detected from caller's module.

        Returns:
            True if the module is approved for direct access to internal modules.

        The approved modules are automatically generated to include:
        - Public facade module (auto-detected or provided)
        - Internal module directory (auto-detected or provided)
        - tests. (always included)

        """
        approved_modules = cls._generate_approved_modules(
            facade_module, internal_module
        )
        return any(module_name.startswith(approved) for approved in approved_modules)
