"""Validation and publication of enforcement JSON package data.

``catalog.json`` is the enforcement rule catalog, validated into
``m.EnforcementCatalog`` by its consumer. ``predicates.json`` holds every
runtime rule keyed by tag: its category, optional layer, problem/fix text, the
predicate kind and the predicate parameters (validated into
``m.EnforcementPredicateSpec`` by the runtime engine). ``exemptions.json``
holds the exemption, allowance and rename tables the runtime rules consult,
each exemption with its reason. ``smells.json`` carries the smell thresholds,
tags and smell rule text.

The package initializer re-exports this facade; the declarations live here
(ADR-018: an initializer propagates and never declares).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.resources
from types import MappingProxyType
from typing import ClassVar

from pydantic import BaseModel

from flext_core._constants._enforcement import FlextConstantsEnforcementRules
from flext_core._constants._enforcement_data import (
    FlextPredicateRule,
    FlextSmellThresholds,
)
from flext_core._typings import FlextTypingBase


class FlextConstantsEnforcementData(BaseModel):
    """Validated enforcement package data published to the constants facade."""

    _DATA_PACKAGE: ClassVar[str] = "flext_core._constants._enforcement_data"

    class SmellData(BaseModel):
        """Smell thresholds, tags and rule text."""

        thresholds: FlextSmellThresholds
        tags: tuple[str, ...]
        rules_text: FlextTypingBase.StrPairMapping

    class PredicateData(BaseModel):
        """Runtime predicates keyed by tag."""

        predicates: FlextTypingBase.MappingKV[str, FlextPredicateRule]

    class ExemptionData(BaseModel):
        """Exemption, allowance and rename tables consulted by runtime rules."""

        relaxed_extra_bases: FlextTypingBase.StrMapping
        infrastructure_bases: tuple[str, ...]
        constants_skip_attrs: tuple[str, ...]
        utilities_exempt_methods: tuple[str, ...]
        utilities_stateful_adapters: FlextTypingBase.StrMapping
        layer_allows: FlextTypingBase.StrSequenceMapping
        value_object_bases: tuple[str, ...]
        nested_mro_min_depth: int
        classvar_exempt_names: FlextTypingBase.StrMapping
        core_path_markers: tuple[str, ...]
        non_workspace_path_markers: tuple[str, ...]
        accessor_renames: FlextTypingBase.StrPairMapping
        accessor_external_contracts: FlextTypingBase.StrMapping

    SMELL: ClassVar[SmellData] = SmellData.model_validate_json(
        importlib.resources
        .files(_DATA_PACKAGE)
        .joinpath("smells.json")
        .read_text(encoding="utf-8"),
    )
    PREDICATES: ClassVar[FlextTypingBase.MappingKV[str, FlextPredicateRule]] = (
        MappingProxyType(
            dict(
                PredicateData.model_validate_json(
                    importlib.resources
                    .files(_DATA_PACKAGE)
                    .joinpath("predicates.json")
                    .read_text(encoding="utf-8"),
                ).predicates,
            ),
        )
    )
    EXEMPTIONS: ClassVar[ExemptionData] = ExemptionData.model_validate_json(
        importlib.resources
        .files(_DATA_PACKAGE)
        .joinpath("exemptions.json")
        .read_text(encoding="utf-8"),
    )

    ENFORCEMENT_RELAXED_EXTRA_BASES: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.relaxed_extra_bases,
    )
    ENFORCEMENT_INFRASTRUCTURE_BASES: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.infrastructure_bases,
    )
    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.constants_skip_attrs,
    )
    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.utilities_exempt_methods,
    )
    ENFORCEMENT_UTILITIES_STATEFUL_ADAPTERS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.utilities_stateful_adapters,
    )
    ENFORCEMENT_LAYER_ALLOWS: ClassVar[
        FlextTypingBase.MappingKV[str, frozenset[str]]
    ] = MappingProxyType({
        layer: frozenset(kinds) for layer, kinds in EXEMPTIONS.layer_allows.items()
    })
    ENFORCEMENT_VALUE_OBJECT_BASES: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.value_object_bases,
    )
    ENFORCEMENT_NESTED_MRO_MIN_DEPTH: ClassVar[int] = EXEMPTIONS.nested_mro_min_depth
    ENFORCEMENT_CLASSVAR_EXEMPT_NAMES: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.classvar_exempt_names,
    )
    ENFORCE_FLEXT_CORE_PATH_MARKERS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.core_path_markers,
    )
    ENFORCE_NON_WORKSPACE_PATH_MARKERS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.non_workspace_path_markers,
    )
    ENFORCEMENT_ACCESSOR_RENAMES: ClassVar[
        FlextTypingBase.MappingKV[str, tuple[str, str]]
    ] = MappingProxyType(EXEMPTIONS.accessor_renames)
    ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS: ClassVar[frozenset[str]] = frozenset(
        EXEMPTIONS.accessor_external_contracts,
    )

    ENFORCEMENT_SMELL_TAGS: ClassVar[tuple[str, ...]] = SMELL.tags
    SMELL_THRESHOLDS: ClassVar[FlextTypingBase.MappingKV[str, int]] = MappingProxyType(
        SMELL.thresholds.model_dump(),
    )
    SMELL_RULES_TEXT: ClassVar[FlextTypingBase.MappingKV[str, tuple[str, str]]] = (
        MappingProxyType(SMELL.rules_text)
    )
    ENFORCEMENT_TAG_CATEGORY: ClassVar[
        FlextTypingBase.MappingKV[
            str,
            FlextConstantsEnforcementRules.EnforcementCategory,
        ]
    ] = MappingProxyType({tag: rule.category for tag, rule in PREDICATES.items()})
    ENFORCEMENT_TAG_LAYER: ClassVar[FlextTypingBase.MappingKV[str, str]] = (
        MappingProxyType({
            tag: rule.layer for tag, rule in PREDICATES.items() if rule.layer
        })
    )
    ENFORCEMENT_TAG_COLLECT: ClassVar[FlextTypingBase.MappingKV[str, str]] = (
        MappingProxyType({
            tag: rule.collect for tag, rule in PREDICATES.items() if rule.collect
        })
    )
    ENFORCEMENT_RECURSIVE_TAGS: ClassVar[frozenset[str]] = frozenset(
        tag for tag, rule in PREDICATES.items() if rule.recursive
    )
    ENFORCEMENT_RULES_TEXT: ClassVar[
        FlextTypingBase.MappingKV[str, tuple[str, str]]
    ] = MappingProxyType({
        **{tag: (rule.problem, rule.fix) for tag, rule in PREDICATES.items()},
        **SMELL.rules_text,
    })
    ENFORCEMENT_PREDICATE_SPECS: ClassVar[
        FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping]
    ] = MappingProxyType({
        tag: {"predicate": rule.predicate, "params": rule.params}
        for tag, rule in PREDICATES.items()
    })


__all__: tuple[str, ...] = ("FlextConstantsEnforcementData",)
