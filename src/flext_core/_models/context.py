"""Extracted facade class for context.

Copyright (c) 2026 FLEXT Team. All rights reserved.
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

from flext_core._models._context import (
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextMetadata,
    FlextModelsContextScope,
    FlextModelsContextTokens,
)


class FlextModelsContext(
    FlextModelsContextData,
    FlextModelsContextExport,
    FlextModelsContextMetadata,
    FlextModelsContextScope,
    FlextModelsContextTokens,
):
    """Facade for context."""


__all__: list[str] = ["FlextModelsContext"]
