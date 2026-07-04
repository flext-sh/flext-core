"""Context metadata and domain data models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes as t


class FlextModelsContextMetadata:
    """Namespace for context metadata models."""


__all__: t.MutableSequenceOf[str] = ["FlextModelsContextMetadata"]
