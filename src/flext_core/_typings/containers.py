"""FlextTypingContainers - pure typing-only container aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import FlextTypingBase as t
from .pydantic import FlextTypesPydantic as tp


class FlextTypingContainers:
    """Protocol-independent value and container aliases."""

    type JsonPayloadLeaf = t.Scalar | Path | tp.JsonValue | tp.BaseModelType
    type JsonPayloadCollectionValue = (
        JsonPayloadLeaf
        | t.MappingKV[str, JsonPayloadLeaf]
        | t.SequenceOf[JsonPayloadLeaf]
    )
    type JsonPayload = (
        JsonPayloadLeaf
        | t.MappingKV[str, JsonPayloadCollectionValue]
        | t.SequenceOf[JsonPayloadCollectionValue]
    )
    type ScalarOrModel = t.Scalar | tp.BaseModelType
    type RegistrablePlugin = ScalarOrModel | Callable[..., ScalarOrModel]


__all__: list[str] = ["FlextTypingContainers"]
