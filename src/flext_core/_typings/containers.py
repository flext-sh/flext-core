"""FlextTypingContainers - pure typing-only container aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)

from flext_core import FlextTypingBase as t


class FlextTypingContainers:
    """Container aliases for type contracts only (no runtime behavior)."""

    type Dict = Mapping[str, t.Container]
    type ConfigMap = Mapping[str, t.Container]
    type ObjectList = Sequence[t.Container]


__all__: list[str] = ["FlextTypingContainers"]
