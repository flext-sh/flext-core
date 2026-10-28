"""Context scope and statistics models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from flext_core._models._context._scope import FlextModelsContextScopePart03


class FlextModelsContextScope(FlextModelsContextScopePart03):
    """Public facade for FlextModelsContextScope."""


__all__: list[str] = ["FlextModelsContextScope"]
