"""JSON package data for enforcement: catalog, runtime predicates and smells.

``catalog.json`` is the enforcement rule catalog, validated into
``m.EnforcementCatalog`` by its consumer. ``predicates.json`` holds every
runtime rule keyed by tag: its category, optional layer, problem/fix text, the
predicate kind and the predicate parameters (validated into
``m.EnforcementPredicateSpec`` by the runtime engine). ``exemptions.json``
holds the exemption, allowance and rename tables the runtime rules consult,
each exemption with its reason. ``smells.json`` carries the smell thresholds,
tags and smell rule text.

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
    collect: str = ""
    recursive: bool = False
    layer: str = ""
    problem: str
    fix: str
    predicate: str
    params: dict[str, JsonValue]


class _PredicateData(BaseModel):
    predicates: dict[str, _PredicateRule]


class _ExemptionData(BaseModel):
    relaxed_extra_bases: dict[str, str]
    infrastructure_bases: tuple[str, ...]
    constants_skip_attrs: tuple[str, ...]
    utilities_exempt_methods: tuple[str, ...]
    layer_allows: dict[str, tuple[str, ...]]
    value_object_bases: tuple[str, ...]
    nested_mro_min_depth: int
    classvar_exempt_names: dict[str, str]
    core_path_markers: tuple[str, ...]
    non_workspace_path_markers: tuple[str, ...]
    accessor_renames: dict[str, tuple[str, str]]
    accessor_external_contracts: dict[str, str]


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

_EXEMPTIONS: _ExemptionData = _ExemptionData.model_validate_json(
    _resource_text("exemptions.json")
)

ENFORCEMENT_RELAXED_EXTRA_BASES: frozenset[str] = frozenset(
    _EXEMPTIONS.relaxed_extra_bases
)
ENFORCEMENT_INFRASTRUCTURE_BASES: frozenset[str] = frozenset(
    _EXEMPTIONS.infrastructure_bases
)
ENFORCEMENT_CONSTANTS_SKIP_ATTRS: frozenset[str] = frozenset(
    _EXEMPTIONS.constants_skip_attrs
)
ENFORCEMENT_UTILITIES_EXEMPT_METHODS: frozenset[str] = frozenset(
    _EXEMPTIONS.utilities_exempt_methods
)
ENFORCEMENT_LAYER_ALLOWS: t.MappingKV[str, frozenset[str]] = MappingProxyType({
    layer: frozenset(kinds) for layer, kinds in _EXEMPTIONS.layer_allows.items()
})
ENFORCEMENT_VALUE_OBJECT_BASES: frozenset[str] = frozenset(
    _EXEMPTIONS.value_object_bases
)
ENFORCEMENT_NESTED_MRO_MIN_DEPTH: int = _EXEMPTIONS.nested_mro_min_depth
ENFORCEMENT_CLASSVAR_EXEMPT_NAMES: frozenset[str] = frozenset(
    _EXEMPTIONS.classvar_exempt_names
)
ENFORCE_FLEXT_CORE_PATH_MARKERS: frozenset[str] = frozenset(
    _EXEMPTIONS.core_path_markers
)
ENFORCE_NON_WORKSPACE_PATH_MARKERS: frozenset[str] = frozenset(
    _EXEMPTIONS.non_workspace_path_markers
)
ENFORCEMENT_ACCESSOR_RENAMES: t.MappingKV[str, tuple[str, str]] = MappingProxyType(
    _EXEMPTIONS.accessor_renames
)
ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS: frozenset[str] = frozenset(
    _EXEMPTIONS.accessor_external_contracts
)

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
ENFORCEMENT_TAG_COLLECT: t.MappingKV[str, str] = MappingProxyType({
    tag: rule.collect for tag, rule in _PREDICATES.items() if rule.collect
})
ENFORCEMENT_RECURSIVE_TAGS: frozenset[str] = frozenset(
    tag for tag, rule in _PREDICATES.items() if rule.recursive
)
ENFORCEMENT_RULES_TEXT: t.MappingKV[str, tuple[str, str]] = MappingProxyType({
    **{tag: (rule.problem, rule.fix) for tag, rule in _PREDICATES.items()},
    **_SMELL_DATA.rules_text,
})
ENFORCEMENT_PREDICATE_SPECS: t.MappingKV[str, t.JsonMapping] = MappingProxyType({
    tag: {"predicate": rule.predicate, "params": rule.params}
    for tag, rule in _PREDICATES.items()
})

__all__: list[str] = [
    "ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS",
    "ENFORCEMENT_ACCESSOR_RENAMES",
    "ENFORCEMENT_CLASSVAR_EXEMPT_NAMES",
    "ENFORCEMENT_CONSTANTS_SKIP_ATTRS",
    "ENFORCEMENT_INFRASTRUCTURE_BASES",
    "ENFORCEMENT_LAYER_ALLOWS",
    "ENFORCEMENT_NESTED_MRO_MIN_DEPTH",
    "ENFORCEMENT_PREDICATE_SPECS",
    "ENFORCEMENT_RELAXED_EXTRA_BASES",
    "ENFORCEMENT_UTILITIES_EXEMPT_METHODS",
    "ENFORCEMENT_VALUE_OBJECT_BASES",
    "ENFORCE_FLEXT_CORE_PATH_MARKERS",
    "ENFORCE_NON_WORKSPACE_PATH_MARKERS",
    "ENFORCEMENT_RECURSIVE_TAGS",
    "ENFORCEMENT_RULES_TEXT",
    "ENFORCEMENT_SMELL_TAGS",
    "ENFORCEMENT_TAG_CATEGORY",
    "ENFORCEMENT_TAG_COLLECT",
    "ENFORCEMENT_TAG_LAYER",
    "SMELL_RULES_TEXT",
    "SMELL_THRESHOLDS",
]
