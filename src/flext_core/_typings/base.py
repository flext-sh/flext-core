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

from flext_core._typings.annotateds import FlextTypesAnnotateds as ta
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextTypingBase(tp, ta):
    """Base type alias namespace for Flext core type-safe contracts."""

    type MappingKV[KeyT, ValueT] = Mapping[KeyT, ValueT]
    type MutableMappingKV[KeyT, ValueT] = MutableMapping[KeyT, ValueT]
    type SequenceOf[ItemT] = Sequence[ItemT]
    type MutableSequenceOf[ItemT] = MutableSequence[ItemT]

    type Numeric = tp.StrictInt | tp.StrictFloat

    type Primitives = tp.StrictStr | Numeric | tp.StrictBool

    type Scalar = Primitives | tp.StrictBytes | datetime
    type ScalarMapping = MappingKV[str, Scalar]
    type ScalarList = SequenceOf[Scalar]
    type MutableScalarMapping = MutableMapping[str, Scalar]

    type StrMapping = MappingKV[str, str]
    type StrSequence = SequenceOf[str]
    type MutableStrMapping = MutableMapping[str, str]
    type OptionalStrMapping = MappingKV[str, str | None]
    type MutableOptionalStrMapping = MutableMapping[str, str | None]

    type SecretValue = tp.SecretStr | tp.SecretBytes
    type SettingsValue = tp.JsonValue | SecretValue | Path

    type JsonValue = tp.JsonValue
    type JsonMapping = MappingKV[str, tp.JsonValue]
    type JsonList = SequenceOf[tp.JsonValue]
    type MutableJsonMapping = MutableMapping[str, tp.JsonValue]
    type MutableJsonList = MutableSequenceOf[tp.JsonValue]
    type FlatContainerList = SequenceOf[tp.JsonValue]
    type MutableFlatContainerList = MutableSequenceOf[tp.JsonValue]
    type FlatContainerMapping = MappingKV[str, tp.JsonValue]
    type FlatContainer = FlatContainerMapping | SequenceOf[tp.JsonValue]
    type MutableFlatContainerMapping = MutableMapping[str, tp.JsonValue]
    type MutableFlatContainer = (
        MutableFlatContainerMapping | MutableSequenceOf[tp.JsonValue]
    )
    # Canonical consumer aliases (flat; no recursion — tp.JsonValue carries depth)
    type MutableOptionalFeatureFlagMapping = MutableMapping[str, str | bool | None]
    type IntMapping = MappingKV[str, int]
    type MutableIntMapping = MutableMapping[str, int]
    type BoolMapping = MappingKV[str, bool]
    type MutableBoolMapping = MutableMapping[str, bool]
    type OptionalBoolMapping = MappingKV[str, bool | None]
    type MutableOptionalBoolMapping = MutableMapping[str, bool | None]
    type FrozensetMapping = MappingKV[str, frozenset[str]]
    type MutableFrozensetMapping = MutableMapping[str, frozenset[str]]
    type StrSequenceMapping = MappingKV[str, StrSequence]
    type MutableStrSequenceMapping = MutableMapping[str, MutableSequenceOf[str]]
    type ScalarOrStrSequenceMapping = MappingKV[str, Scalar | StrSequence]

    # Recurring domain-specific flat mapping aliases
    type AttributeMapping = MappingKV[str, str | MutableSequenceOf[str]]
    type MutableAttributeMapping = MutableMapping[str, str | MutableSequenceOf[str]]
    type ConfigValueMapping = MappingKV[str, str | int | float]
    type HeaderMapping = MappingKV[str, int | str]
    type FeatureFlagMapping = MappingKV[str, str | bool]
    type MutableFeatureFlagMapping = MutableMapping[str, str | bool]
    type MutableHeaderMapping = MutableMapping[str, int | str]
    type MutableConfigValueMapping = MutableMapping[str, str | int | float]

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
