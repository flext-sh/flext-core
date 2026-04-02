"""Core type aliases and container typing conventions for Flext.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import SecretBytes, SecretStr


class FlextTypingBase:
    """Base type alias namespace for Flext core type-safe contracts."""

    type Numeric = int | float
    type Primitives = str | Numeric | bool
    type Scalar = Primitives | datetime
    type Container = Scalar | Path
    type RecursiveContainerMapping = Mapping[str, FlextTypingBase.RecursiveContainer]
    type RecursiveContainerList = Sequence[FlextTypingBase.RecursiveContainer]
    type MutableRecursiveContainerMapping = MutableMapping[
        str,
        FlextTypingBase.RecursiveContainer,
    ]
    type MutableRecursiveContainerList = MutableSequence[
        FlextTypingBase.RecursiveContainer,
    ]
    type RecursiveContainer = (
        Container
        | FlextTypingBase.RecursiveContainerMapping
        | FlextTypingBase.RecursiveContainerList
        | tuple[FlextTypingBase.RecursiveContainer, ...]
        | None
    )
    type ContainerMapping = RecursiveContainerMapping
    type ContainerList = RecursiveContainerList
    type MutableContainerMapping = MutableRecursiveContainerMapping
    type MutableContainerList = MutableRecursiveContainerList
    type NormalizedValue = RecursiveContainer
    type SecretValue = SecretStr | SecretBytes
    type SettingsValue = RecursiveContainer | SecretValue

    # Flat (non-recursive) mapping/list aliases for high-frequency patterns
    type StrMapping = Mapping[str, str]
    type StrSequence = Sequence[str]

    type ScalarMapping = Mapping[str, Scalar]
    type ScalarList = Sequence[Scalar]
    type FlatContainerList = Sequence[Container]

    class ContainerMappingBase(Mapping[str, "FlextTypingBase.RecursiveContainer"]):
        """Concrete base for Mapping[str, RecursiveContainer] inheritance.

        PEP 695 ``type X = ...`` aliases cannot be subclassed (CPython limitation).
        Use ``t.*Base`` classes when inheriting, ``t.*`` aliases for annotations.
        """

    class ContainerListBase(Sequence["FlextTypingBase.RecursiveContainer"]):
        """Concrete base for Sequence[RecursiveContainer] inheritance."""

    class MutableContainerMappingBase(
        MutableMapping[str, "FlextTypingBase.RecursiveContainer"],
    ):
        """Concrete base for MutableMapping[str, RecursiveContainer] inheritance."""

    class MutableContainerListBase(
        MutableSequence["FlextTypingBase.RecursiveContainer"]
    ):
        """Concrete base for MutableSequence[RecursiveContainer] inheritance."""

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

    type Pair[LeftT, RightT] = tuple[LeftT, RightT]
    type Triple[FirstT, SecondT, ThirdT] = tuple[FirstT, SecondT, ThirdT]
    type VariadicTuple[ItemT] = tuple[ItemT, ...]
    type IntPair = Pair[int, int]
