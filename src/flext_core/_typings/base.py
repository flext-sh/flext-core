"""Canonical base declarations and type-family MRO for Flext.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from datetime import datetime
from pathlib import Path
from types import GenericAlias, UnionType
from typing import ForwardRef, TypeAliasType

import typing_extensions

from flext_core._typings import FlextTypesPydantic


class FlextTypingBase(FlextTypesPydantic):
    """Base type alias namespace for Flext core type-safe contracts."""

    type RegexPattern = re.Pattern[str]
    type RegexMatch = re.Match[str]

    type Numeric = FlextTypesPydantic.StrictInt | FlextTypesPydantic.StrictFloat

    type Primitives = (
        FlextTypesPydantic.StrictStr | Numeric | FlextTypesPydantic.StrictBool
    )

    type Scalar = Primitives | FlextTypesPydantic.StrictBytes | datetime

    type ScalarMapping = Mapping[str, Scalar]
    type MutableStrMapping = MutableMapping[str, str]
    type OptionalStrMapping = Mapping[str, str | None]
    type MutableOptionalStrMapping = MutableMapping[str, str | None]

    type SecretValue = FlextTypesPydantic.SecretStr | FlextTypesPydantic.SecretBytes
    type SettingsValue = FlextTypesPydantic.JsonValue | SecretValue | Path

    type JsonMapping = Mapping[str, FlextTypesPydantic.JsonValue]
    type JsonDict = dict[str, FlextTypesPydantic.JsonValue]
    type JsonValueList = list[FlextTypesPydantic.JsonValue]
    type JsonList = Sequence[FlextTypesPydantic.JsonValue]
    type MutableJsonMapping = MutableMapping[str, FlextTypesPydantic.JsonValue]
    type MutableJsonList = MutableSequence[FlextTypesPydantic.JsonValue]
    type FlatContainerList = Sequence[FlextTypesPydantic.JsonValue]
    type MutableFlatContainerList = MutableSequence[FlextTypesPydantic.JsonValue]
    type FlatContainerMapping = Mapping[str, FlextTypesPydantic.JsonValue]
    type FlatContainer = FlatContainerMapping | Sequence[FlextTypesPydantic.JsonValue]
    type MutableFlatContainerMapping = MutableMapping[str, FlextTypesPydantic.JsonValue]
    type MutableFlatContainer = (
        MutableFlatContainerMapping | MutableSequence[FlextTypesPydantic.JsonValue]
    )
    type RuntimeData = FlextTypesPydantic.JsonValue | Path | datetime
    type MutableOptionalFeatureFlagMapping = MutableMapping[str, str | bool | None]
    # Family mapping/sequence vocabulary, restored fix-forward per ADR-028
    # Decision 7 (amended 2026-10-29): the first removal wave deleted these
    # while non-migrated consumers still spoke them. Managed transition — the
    # native-annotation cutover resumes; each alias retires when its consumer
    # census reaches zero.
    type StrMapping = Mapping[str, str]
    type StrSequence = Sequence[str]
    type SequenceOf[T] = Sequence[T]
    type MutableSequenceOf[T] = MutableSequence[T]
    type MappingKV[K, V] = Mapping[K, V]
    type MutableMappingKV[K, V] = MutableMapping[K, V]
    type IntMapping = Mapping[str, int]
    type MutableIntMapping = MutableMapping[str, int]
    type BoolMapping = Mapping[str, bool]
    type MutableBoolMapping = MutableMapping[str, bool]
    type OptionalBoolMapping = Mapping[str, bool | None]
    type MutableOptionalBoolMapping = MutableMapping[str, bool | None]
    type FrozensetMapping = Mapping[str, frozenset[str]]
    type MutableFrozensetMapping = MutableMapping[str, frozenset[str]]
    type StrSequenceMapping = Mapping[str, Sequence[str]]
    type MutableStrSequenceMapping = MutableMapping[str, MutableSequence[str]]
    type ScalarOrStrSequenceMapping = Mapping[str, Scalar | Sequence[str]]

    # Recurring domain-specific flat mapping aliases
    type AttributeMapping = Mapping[str, str | MutableSequence[str]]
    type MutableAttributeMapping = MutableMapping[str, str | MutableSequence[str]]
    type ConfigValueMapping = Mapping[str, str | int | float]
    type HeaderMapping = Mapping[str, int | str]
    type FeatureFlagMapping = Mapping[str, str | bool]
    type MutableFeatureFlagMapping = MutableMapping[str, str | bool]
    type MutableHeaderMapping = MutableMapping[str, int | str]
    type MutableConfigValueMapping = MutableMapping[str, str | int | float]

    type VariadicTuple[ItemT] = tuple[ItemT, ...]
    type StrTuple = VariadicTuple[str]
    type StrPairSequence = Sequence[tuple[str, str]]
    type MutableStrPairSequence = MutableSequence[tuple[str, str]]
    type StrPairTuple = VariadicTuple[tuple[str, str]]
    type StrPairMapping = Mapping[str, tuple[str, str]]
    type MutableStrPairMapping = MutableMapping[str, tuple[str, str]]
    type StrPairTupleMapping = Mapping[str, StrPairTuple]
    type MutableStrPairTupleMapping = MutableMapping[str, StrPairTuple]
    type OptionalStrPairList = list[tuple[str, str] | None]
    type OptionalStrPairMapping = Mapping[str, tuple[str, str] | None]
    type OptionalStrPairCollection = OptionalStrPairList | OptionalStrPairMapping
    type MutableOptionalStrPairSequence = MutableSequence[tuple[str, str] | None]
    type StrSequencePair = tuple[str, Sequence[str]]
    type StrSequencePairSequence = Sequence[StrSequencePair]
    type StrSequencePairTuple = VariadicTuple[StrSequencePair]
    type StrPairSequencePair = tuple[str, StrPairSequence]
    type StrPairSequencePairSequence = Sequence[StrPairSequencePair]
    type LazyImportEntry = str | tuple[str, str]
    type LazyImportMap = Mapping[str, LazyImportEntry]
    type LazyImportDict = dict[str, LazyImportEntry]
    type MutableLazyImportMap = MutableMapping[str, LazyImportEntry]
    type LazyImportAliasGroups = Mapping[str, StrPairSequence]
    type LazyAliasMap = Mapping[str, tuple[str, str]]
    type MutableLazyAliasMap = MutableMapping[str, tuple[str, str]]

    type TypeHintSpecifier = (
        type[object] | str | UnionType | GenericAlias | TypeAliasType | ForwardRef
    )
    type TypeFormSpecifier = TypeHintSpecifier | typing_extensions.TypeForm[object]

    TypeForm = typing_extensions.TypeForm
    """PEP 747 ``TypeForm`` re-bound from its canonical owner (ADR-028, operator
    order 2026-10-29): usable both in annotations and in call position
    (``t.TypeForm(Mapping[str, int])``), without importing typing_extensions
    outside the facades. A PEP 695 ``type`` alias is not callable; the owner's
    special form is."""


__all__: list[str] = ["FlextTypingBase"]
