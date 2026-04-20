"""Core type aliases and container typing conventions for Flext.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from datetime import datetime
from pathlib import Path

from flext_core import FlextTypesPydantic as tp


class FlextTypingBase:
    """Base type alias namespace for Flext core type-safe contracts."""

    type Numeric = int | float
    type Primitives = str | Numeric | bool
    type Scalar = Primitives | bytes | datetime
    type JsonValue = tp.JsonValue
    type JsonMapping = Mapping[str, JsonValue]
    type JsonList = Sequence[JsonValue]
    type FlatScalarMapping = Mapping[str, Scalar]
    type FlatScalarSequence = Sequence[Scalar]
    type Container = Scalar | Path | FlatScalarMapping | FlatScalarSequence | JsonValue
    type OpaqueValue = Container
    type MappingKV[KeyT, ValueT] = Mapping[KeyT, ValueT]
    type MutableMappingKV[KeyT, ValueT] = dict[KeyT, ValueT]
    type SequenceOf[ItemT] = Sequence[ItemT]
    type MutableSequenceOf[ItemT] = list[ItemT]
    type SecretValue = tp.SecretStr | tp.SecretBytes
    type SettingsValue = Scalar | SecretValue

    # Flat (non-recursive) mapping/list aliases for high-frequency patterns
    type StrMapping = Mapping[str, str]
    type MutableStrMapping = MutableMapping[str, str]
    type OptionalStrMapping = Mapping[str, str | None]
    type MutableOptionalStrMapping = MutableMapping[str, str | None]
    type StrSequence = Sequence[str]

    type ScalarMapping = Mapping[str, Scalar]
    type MutableScalarMapping = MutableMapping[str, Scalar]
    type ScalarList = Sequence[Scalar]
    type FlatContainerList = Sequence[Container]
    type MutableFlatContainerList = MutableSequence[Container]
    type FlatContainerMapping = Mapping[str, Container]
    type MutableFlatContainerMapping = MutableMapping[str, Container]
    # Canonical consumer aliases (flat; no recursion — JsonValue carries depth)
    type ContainerValue = Container
    type ContainerValueMapping = FlatContainerMapping
    type MutableContainerValueMapping = MutableFlatContainerMapping
    type ContainerValueList = FlatContainerList
    type MutableContainerValueList = MutableFlatContainerList
    type RecursiveValue = JsonValue
    type RecursiveContainerMapping = FlatContainerMapping
    type MutableRecursiveContainerMapping = MutableFlatContainerMapping
    type RecursiveContainerList = FlatContainerList
    type MutableRecursiveContainerList = MutableFlatContainerList
    type OptionalContainerValue = Container | None
    type OptionalContainerValueMapping = FlatContainerMapping | None
    type OptionalScalar = Scalar | None
    type OptionalPrimitive = Primitives | None
    type MutableOptionalFeatureFlagMapping = MutableMapping[str, str | bool | None]
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
    type HeaderMapping = Mapping[str, int | str]
    type FeatureFlagMapping = Mapping[str, str | bool]
    type MutableFeatureFlagMapping = MutableMapping[str, str | bool]
    type MutableHeaderMapping = MutableMapping[str, int | str]
    type MutableConfigValueMapping = MutableMapping[str, str | int | float]

    class ContainerMappingBase(Mapping[str, Container]):
        """Concrete base for Mapping[str, Container] inheritance.

        PEP 695 ``type X = ...`` aliases cannot be subclassed (CPython limitation).
        Use ``t.*Base`` classes when inheriting, ``t.*`` aliases for annotations.
        """

    class ContainerListBase(Sequence[Container]):
        """Concrete base for Sequence[Container] inheritance."""

    class MutableContainerMappingBase(
        MutableMapping[str, Container],
    ):
        """Concrete base for MutableMapping[str, Container] inheritance."""

    class MutableContainerListBase(
        MutableSequence[Container],
    ):
        """Concrete base for MutableSequence[Container] inheritance."""

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
