"""Public runtime facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._runtime import FlextRuntimeContainer


class FlextRuntime(FlextRuntimeContainer):
    """Expose runtime normalization and validation helpers.

    The ``_runtime`` family package owns the implementation; this facade
    publishes the composed namespace consumed as ``FlextRuntime`` by the
    other flext-core modules and downstream packages.
    """


__all__: list[str] = ["FlextRuntime"]
