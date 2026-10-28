"""Public runtime facade namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities


class FlextRuntime(
    FlextUtilities,
):
    """Expose runtime normalization and validation helpers.

    Runtime primitives and Pydantic normalization retain their existing
    owners; this namespace adds no implementations or compatibility methods.
    """


__all__: list[str] = ["FlextRuntime"]
