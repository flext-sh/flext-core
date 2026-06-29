"""Composed dependency-injector runtime bridge.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._runtime._dependency_bindings import FlextRuntimeDependencyBindings


class FlextRuntimeDependencyIntegration:
    """Expose dependency-injector integration under the runtime namespace."""

    class DependencyIntegration(FlextRuntimeDependencyBindings):
        """Centralize dependency-injector wiring with provider helpers."""


__all__: list[str] = ["FlextRuntimeDependencyIntegration"]
