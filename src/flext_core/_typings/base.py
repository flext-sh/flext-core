"""Core type aliases and container typing conventions for Flext.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from pathlib import Path


class FlextTypingBase:
    """Base type alias namespace for Flext core type-safe contracts."""

    type Primitives = str | int | float | bool
    type Numeric = int | float
    type Scalar = Primitives | datetime
    type Container = Scalar | Path
    type NormalizedValue = (
        Container
        | Sequence[FlextTypingBase.NormalizedValue]
        | Mapping[str, FlextTypingBase.NormalizedValue]
        | tuple[FlextTypingBase.NormalizedValue, ...]
        | None
    )
    type ContainerMapping = Mapping[str, NormalizedValue]
    type ContainerList = Sequence[NormalizedValue]
    type MutableContainerMapping = MutableMapping[str, NormalizedValue]
    type MutableContainerList = MutableSequence[NormalizedValue]

    PRIMITIVES_TYPES: tuple[type[str], type[int], type[float], type[bool]] = (
        str,
        int,
        float,
        bool,
    )
    NUMERIC_TYPES: tuple[type[int], type[float]] = (int, float)
    SCALAR_TYPES: tuple[
        type[str],
        type[int],
        type[float],
        type[bool],
        type[datetime],
    ] = (str, int, float, bool, datetime)
    CONTAINER_TYPES: tuple[
        type[str],
        type[int],
        type[float],
        type[bool],
        type[datetime],
        type[Path],
    ] = (str, int, float, bool, datetime, Path)
    CONTAINER_AND_COLLECTION_TYPES: tuple[type, ...] = (
        *CONTAINER_TYPES,
        list,
        dict,
        tuple,
    )
