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
from types import GenericAlias, UnionType
from typing import ForwardRef, TypeAliasType

from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextTypingBase:
    """Base type alias namespace for Flext core type-safe contracts."""

    type Numeric = tp.StrictInt | tp.StrictFloat

    type Primitives = tp.StrictStr | Numeric | tp.StrictBool

    type Scalar = Primitives | tp.StrictBytes | datetime
    type ScalarMapping = Mapping[str, Scalar]
    type ScalarList = Sequence[Scalar]
    type MutableScalarMapping = MutableMapping[str, Scalar]

    type StrMapping = Mapping[str, str]
    type StrSequence = Sequence[str]
    type MutableStrMapping = MutableMapping[str, str]
    type OptionalStrMapping = Mapping[str, str | None]
    type MutableOptionalStrMapping = MutableMapping[str, str | None]

    type SecretValue = tp.SecretStr | tp.SecretBytes
    type SettingsValue = tp.JsonValue | SecretValue | Path

    type JsonMapping = Mapping[str, tp.JsonValue]
    type JsonList = Sequence[tp.JsonValue]
    type MutableJsonMapping = MutableMapping[str, tp.JsonValue]
    type MutableJsonList = MutableSequence[tp.JsonValue]
    type FlatContainerList = Sequence[tp.JsonValue]
    type MutableFlatContainerList = MutableSequence[tp.JsonValue]
    type FlatContainerMapping = Mapping[str, tp.JsonValue]
    type FlatContainer = FlatContainerMapping | Sequence[tp.JsonValue]
    type MutableFlatContainerMapping = MutableMapping[str, tp.JsonValue]
    type MutableFlatContainer = (
        MutableFlatContainerMapping | MutableSequence[tp.JsonValue]
    )
    type MappingKV[KeyT, ValueT] = Mapping[KeyT, ValueT]
    type MutableMappingKV[KeyT, ValueT] = MutableMapping[KeyT, ValueT]
    type SequenceOf[ItemT] = Sequence[ItemT]
    type MutableSequenceOf[ItemT] = MutableSequence[ItemT]

    # Canonical consumer aliases (flat; no recursion — tp.JsonValue carries depth)
    type MutableOptionalFeatureFlagMapping = MutableMapping[str, str | bool | None]
    type IntMapping = Mapping[str, int]
    type MutableIntMapping = MutableMapping[str, int]
    type BoolMapping = Mapping[str, bool]
    type MutableBoolMapping = MutableMapping[str, bool]
    type OptionalBoolMapping = Mapping[str, bool | None]
    type MutableOptionalBoolMapping = MutableMapping[str, bool | None]
    type FrozensetMapping = Mapping[str, frozenset[str]]
    type MutableFrozensetMapping = MutableMapping[str, frozenset[str]]
    type StrSequenceMapping = Mapping[str, StrSequence]
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

    class ContainerMappingBase(Mapping[str, tp.JsonValue]):
        """Concrete base for JsonMapping inheritance.

        PEP 695 ``type X = ...`` aliases cannot be subclassed (CPython limitation).
        Use ``t.*Base`` classes when inheriting, ``t.*`` aliases for annotations.
        """

    class ContainerListBase(Sequence[tp.JsonValue]):
        """Concrete base for JsonList inheritance."""

    class MutableContainerMappingBase(
        MutableMapping[str, tp.JsonValue],
    ):
        """Concrete base for MutableMapping[str, tp.JsonValue] inheritance."""

    class MutableContainerListBase(
        MutableSequence[tp.JsonValue],
    ):
        """Concrete base for MutableSequence[tp.JsonValue] inheritance."""

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

    type TypeHintSpecifier = (
        type | str | UnionType | GenericAlias | TypeAliasType | ForwardRef
    )
