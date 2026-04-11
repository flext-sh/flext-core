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
    type OpaqueValue = object
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
    type MappingKV[KeyT, ValueT] = Mapping[KeyT, ValueT]
    type MutableMappingKV[KeyT, ValueT] = dict[KeyT, ValueT]
    type SequenceOf[ItemT] = Sequence[ItemT]
    type MutableSequenceOf[ItemT] = list[ItemT]
    type ContainerMapping = RecursiveContainerMapping
    type ContainerList = RecursiveContainerList
    type MutableContainerMapping = MutableRecursiveContainerMapping
    type MutableContainerList = MutableRecursiveContainerList
    type NormalizedValue = RecursiveContainer
    type SecretValue = SecretStr | SecretBytes
    type SettingsValue = RecursiveContainer | SecretValue

    # Flat (non-recursive) mapping/list aliases for high-frequency patterns
    type StrMapping = Mapping[str, str]
    type MutableStrMapping = MutableMapping[str, str]
    type StrSequence = Sequence[str]

    type ScalarMapping = Mapping[str, Scalar]
    type MutableScalarMapping = MutableMapping[str, Scalar]
    type ScalarList = Sequence[Scalar]
    type FlatContainerList = Sequence[Container]

    type IntMapping = Mapping[str, int]
    type MutableIntMapping = MutableMapping[str, int]
    type BoolMapping = Mapping[str, bool]
    type MutableBoolMapping = MutableMapping[str, bool]
    type FrozensetMapping = Mapping[str, frozenset[str]]
    type MutableFrozensetMapping = MutableMapping[str, frozenset[str]]
    type StrSequenceMapping = Mapping[str, Sequence[str]]
    type MutableStrSequenceMapping = MutableMapping[str, MutableSequence[str]]

    # Recurring domain-specific flat mapping aliases
    type AttributeMapping = Mapping[str, str | MutableSequence[str]]
    type MutableAttributeMapping = MutableMapping[str, str | MutableSequence[str]]
    type ConfigValueMapping = Mapping[str, str | int | float]
    type OptionalStrMapping = Mapping[str, str | None]
    type MutableOptionalStrMapping = MutableMapping[str, str | None]
    type HeaderMapping = Mapping[str, int | str]
    type FeatureFlagMapping = Mapping[str, str | bool]
    type MutableFeatureFlagMapping = MutableMapping[str, str | bool]
    type OptionalFeatureFlagMapping = Mapping[str, str | bool | None]
    type MutableOptionalFeatureFlagMapping = MutableMapping[str, str | bool | None]
    type MutableHeaderMapping = MutableMapping[str, int | str]
    type MutableConfigValueMapping = MutableMapping[str, str | int | float]
    type OptionalBoolMapping = Mapping[str, bool | None]
    type MutableOptionalBoolMapping = MutableMapping[str, bool | None]

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
        MutableSequence["FlextTypingBase.RecursiveContainer"],
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
    type Quad[FirstT, SecondT, ThirdT, FourthT] = tuple[
        FirstT,
        SecondT,
        ThirdT,
        FourthT,
    ]
    type Quint[FirstT, SecondT, ThirdT, FourthT, FifthT] = tuple[
        FirstT,
        SecondT,
        ThirdT,
        FourthT,
        FifthT,
    ]
    type VariadicTuple[ItemT] = tuple[ItemT, ...]
    type IntPair = Pair[int, int]
