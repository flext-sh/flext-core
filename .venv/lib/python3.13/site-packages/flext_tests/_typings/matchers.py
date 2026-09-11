"""Types extraction for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Set as AbstractSet
from typing import TypeAliasType

from _pytest.python_api import ApproxBase
from flext_core import m, p, t
from flext_infra import t as it
from flext_tests._typings.base import FlextTestsBaseTypesMixin as tb


class FlextTestsMatchersTypesMixin:
    type MatcherEqTarget = (
        m.BaseModel
        | t.MappingKV[str, tb.TestobjectSerializable]
        | t.SequenceOf[tb.TestobjectSerializable]
        | bytes
        | str
        | int
        | float
        | bool
        | TypeAliasType
        | ApproxBase
    )
    """Expected-value target for ``Ok``/``Fail`` matcher ``eq`` / ``ne`` fields.

    The ``| None`` nullability is attached by the field defaulting to
    ``None`` rather than by the alias itself — Pydantic rejects
    ``union_mode`` on nullable schemas, so the alias stays non-nullable.
    """

    type MatchRuleLeaf = (
        tb.TestobjectSerializable
        | m.BaseModel
        | type
        | tuple[type, ...]
        | TypeAliasType
        | Callable[..., tb.Testobject]
    )
    type MatchRuleKwargs = t.MappingKV[
        str,
        Callable[..., tb.Testobject]
        | tb.TestobjectSerializable
        | it.Infra.RegexPattern,
    ]
    type MatchRuleValue = MatchRuleLeaf | MatchRuleKwargs
    type MatcherKwargValue = (
        p.AttributeProbe
        | ApproxBase
        | TypeAliasType
        | tuple[type, ...]
        | it.Infra.RegexPattern
        | Callable[..., p.AttributeProbe]
        | AbstractSet[p.AttributeProbe]
        | t.SequenceOf[p.AttributeProbe]
        | t.MappingKV[int | str, p.AttributeProbe]
    )
    type MatcherRuntimeValue = (
        MatchRuleLeaf
        | m.BaseModel
        | set[tb.TestobjectSerializable]
        | it.Infra.RegexPattern
        | Callable[..., tb.Testobject]
        | MatchRuleKwargs
        | t.MappingKV[int, MatchRuleValue]
        | t.MappingKV[str, MatchRuleValue]
        | t.MappingKV[FlextTestsMatchersTypesMixin.ItemSelector, MatchRuleValue]
    )
    type LengthSpec = int | tuple[int, int]
    type ComparableScalar = float | int | str
    """Comparable scalar arms for matcher ``gt``/``gte``/``lt``/``lte`` fields.

    Centralized to satisfy AGENTS.md § Model governance rule against
    inline 3+-arm unions in Pydantic field annotations.
    """
    type DeepSpec = t.MappingKV[
        str, Callable[[tb.Testobject], bool] | tb.TestobjectSerializable
    ]
    type PathMatchSpec = t.MappingKV[str, m.BaseModel]
    type ItemSelector = int | str
    type ItemMatchSpec = (
        t.SequenceOf[m.BaseModel]
        | t.MappingKV[FlextTestsMatchersTypesMixin.ItemSelector, m.BaseModel]
    )
    type AttributeMatchSpec = t.MappingKV[str, m.BaseModel]
    type PathSpec = str | t.StrSequence
    type PredicateSpec = Callable[[tb.Testobject], bool]
    type ContainmentSpec = tb.Testobject | t.SequenceOf[tb.TestobjectSerializable]
    type ExclusionSpec = str | t.StrSequence
    type SequencePredicate = type | Callable[[tb.Testobject], bool]
    type SortKey = bool | Callable[[tb.Testobject], tb.Testobject]
    type KeySpec = t.StrSequence | set[str]
    type KeyValueSpec = (
        tuple[str, tb.TestobjectSerializable]
        | t.MappingKV[str, tb.TestobjectSerializable]
    )
    type AttributeSpec = str | t.StrSequence
    type AttributeValueSpec = (
        tuple[str, tb.TestobjectSerializable]
        | t.MappingKV[str, tb.TestobjectSerializable]
    )
    type ErrorCodeSpec = str | t.StrSequence
    type ErrorDataSpec = m.ConfigMap
    type CleanupSpec = t.SequenceOf[Callable[[], None]]
    type EnvironmentSpec = t.StrMapping
