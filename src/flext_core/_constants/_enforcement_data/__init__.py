"""JSON package data for enforcement: catalog, runtime predicates and smells.

``catalog.json`` is the enforcement rule catalog, validated into
``m.EnforcementCatalog`` by its consumer. ``predicates.json`` holds every
runtime rule keyed by tag: its category, optional layer, problem/fix text, the
predicate kind and the predicate parameters (validated into
``m.EnforcementPredicateSpec`` by the runtime engine). ``smells.json`` carries
the smell thresholds, tags and the remaining smell rule text.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.resources
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue

from .._enforcement_parts.flextconstantsenforcement_part_04 import (
    FlextConstantsEnforcementRules,
)

if TYPE_CHECKING:
    from ..._typings.base import FlextTypingBase as t


class _SmellThresholds(BaseModel):
    params: int
    returns: int
    nesting: int
    fn_cx: int
    file_cx: int


class _SmellData(BaseModel):
    thresholds: _SmellThresholds
    tags: tuple[str, ...]
    rules_text: dict[str, tuple[str, str]]


class _PredicateRule(BaseModel):
    category: FlextConstantsEnforcementRules.EnforcementCategory
    layer: str = ""
    problem: str
    fix: str
    predicate: str
    params: dict[str, JsonValue]


class _PredicateData(BaseModel):
    predicates: dict[str, _PredicateRule]


def _resource_text(name: str) -> str:
    """Read one enforcement package-data resource."""
    return (
        importlib.resources
        .files(__package__)
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


_SMELL_DATA: _SmellData = _SmellData.model_validate_json(_resource_text("smells.json"))
_PREDICATES: dict[str, _PredicateRule] = _PredicateData.model_validate_json(
    _resource_text("predicates.json")
).predicates

ENFORCEMENT_SMELL_TAGS: tuple[str, ...] = _SMELL_DATA.tags
SMELL_THRESHOLDS: t.MappingKV[str, int] = MappingProxyType(
    _SMELL_DATA.thresholds.model_dump()
)
SMELL_RULES_TEXT: t.MappingKV[str, tuple[str, str]] = MappingProxyType(
    _SMELL_DATA.rules_text
)
ENFORCEMENT_TAG_CATEGORY: t.MappingKV[
    str, FlextConstantsEnforcementRules.EnforcementCategory
] = MappingProxyType({tag: rule.category for tag, rule in _PREDICATES.items()})
ENFORCEMENT_TAG_LAYER: t.MappingKV[str, str] = MappingProxyType({
    tag: rule.layer for tag, rule in _PREDICATES.items() if rule.layer
})
ENFORCEMENT_RULES_TEXT: t.MappingKV[str, tuple[str, str]] = MappingProxyType({
    **{tag: (rule.problem, rule.fix) for tag, rule in _PREDICATES.items()},
    **_SMELL_DATA.rules_text,
})
ENFORCEMENT_PREDICATE_SPECS: t.MappingKV[str, t.JsonMapping] = MappingProxyType({
    tag: {"predicate": rule.predicate, "params": rule.params}
    for tag, rule in _PREDICATES.items()
})

__all__: list[str] = [
    "ENFORCEMENT_PREDICATE_SPECS",
    "ENFORCEMENT_RULES_TEXT",
    "ENFORCEMENT_SMELL_TAGS",
    "ENFORCEMENT_TAG_CATEGORY",
    "ENFORCEMENT_TAG_LAYER",
    "SMELL_RULES_TEXT",
    "SMELL_THRESHOLDS",
]
