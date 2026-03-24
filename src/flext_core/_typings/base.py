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

    type Numeric = int | float
    type Primitives = str | Numeric | bool
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

    # Flat (non-recursive) mapping/list aliases for high-frequency patterns
    type StrMapping = Mapping[str, str]
    type StrSequence = Sequence[str]
    type ScalarMapping = Mapping[str, Scalar]
    type ScalarList = Sequence[Scalar]
    type FlatContainerList = Sequence[Container]

    class ContainerMappingBase(Mapping[str, "FlextTypingBase.NormalizedValue"]):
        """Concrete base for Mapping[str, NormalizedValue] inheritance.

        PEP 695 ``type X = ...`` aliases cannot be subclassed (CPython limitation).
        Use ``t.*Base`` classes when inheriting, ``t.*`` aliases for annotations.
        """

    class ContainerListBase(Sequence["FlextTypingBase.NormalizedValue"]):
        """Concrete base for Sequence[NormalizedValue] inheritance."""

    class MutableContainerMappingBase(
        MutableMapping[str, "FlextTypingBase.NormalizedValue"]
    ):
        """Concrete base for MutableMapping[str, NormalizedValue] inheritance."""

    class MutableContainerListBase(MutableSequence["FlextTypingBase.NormalizedValue"]):
        """Concrete base for MutableSequence[NormalizedValue] inheritance."""

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
