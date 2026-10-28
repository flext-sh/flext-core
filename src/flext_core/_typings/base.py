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
    type IntMapping = Mapping[str, int]
    type IntDict = dict[str, int]
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
    type LazyAliasDict = dict[str, tuple[str, str]]
    type MutableLazyAliasMap = MutableMapping[str, tuple[str, str]]
    type IntPair = tuple[int, int]

    type TypeHintSpecifier = (
        type[object] | str | UnionType | GenericAlias | TypeAliasType | ForwardRef
    )
    type TypeFormSpecifier = TypeHintSpecifier | typing_extensions.TypeForm[object]

    type TypeForm[T] = typing_extensions.TypeForm[T]
    """PEP 747 ``TypeForm`` for annotations: a ``t.TypeForm[T]`` parameter takes a
    bare type expression (``u.type_adapter(Mapping[str, int])``) without importing
    typing_extensions outside the facades (ADR-028). A generic type alias, never
    called: a class attribute bound to the special form is not a type form."""


__all__: list[str] = ["FlextTypingBase"]
