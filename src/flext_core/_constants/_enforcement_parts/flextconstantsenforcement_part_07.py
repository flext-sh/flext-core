"""Enforcement rule constants loaded from JSON package data.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_core._constants._enforcement_data import FlextConstantsEnforcementData
from flext_core._constants._enforcement_parts import FlextConstantsEnforcementRules

if TYPE_CHECKING:
    from flext_core._typings import FlextTypingBase


class FlextConstantsEnforcementSmellData:
    """Runtime rules, exemptions, smell thresholds and rule text from package data."""

    ENFORCEMENT_CATALOG_RESOURCE: ClassVar[str] = "catalog.json"
    """Package-data resource holding the enforcement rule catalog."""

    ENFORCEMENT_SMELL_TAGS: ClassVar[FlextTypingBase.VariadicTuple[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_SMELL_TAGS
    )
    SMELL_THRESHOLDS: ClassVar[FlextTypingBase.IntMapping] = (
        FlextConstantsEnforcementData.SMELL_THRESHOLDS
    )
    SMELL_RULES_TEXT: ClassVar[FlextTypingBase.StrPairMapping] = (
        FlextConstantsEnforcementData.SMELL_RULES_TEXT
    )
    ENFORCEMENT_RULES_TEXT: ClassVar[FlextTypingBase.StrPairMapping] = (
        FlextConstantsEnforcementData.ENFORCEMENT_RULES_TEXT
    )
    """Problem/fix text templates per runtime tag; consumed by the emitter."""
    ENFORCEMENT_TAG_CATEGORY: ClassVar[
        FlextTypingBase.MappingKV[
            str,
            FlextConstantsEnforcementRules.EnforcementCategory,
        ]
    ] = FlextConstantsEnforcementData.ENFORCEMENT_TAG_CATEGORY
    """Runtime tag → category dispatching its item collection."""
    ENFORCEMENT_TAG_LAYER: ClassVar[FlextTypingBase.StrMapping] = (
        FlextConstantsEnforcementData.ENFORCEMENT_TAG_LAYER
    )
    """Runtime tag → facade layer for ATTR-category rules."""
    ENFORCEMENT_TAG_COLLECT: ClassVar[FlextTypingBase.StrMapping] = (
        FlextConstantsEnforcementData.ENFORCEMENT_TAG_COLLECT
    )
    """NAMESPACE tag → item collection strategy of the runtime engine."""
    ENFORCEMENT_RECURSIVE_TAGS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_RECURSIVE_TAGS
    )
    """Tags whose scan recurses into inner namespace classes."""
    ENFORCEMENT_PREDICATE_SPECS: ClassVar[
        FlextTypingBase.MappingKV[str, FlextTypingBase.JsonMapping]
    ] = FlextConstantsEnforcementData.ENFORCEMENT_PREDICATE_SPECS
    """Runtime tag → raw predicate kind and parameters (typed by the engine)."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_RELAXED_EXTRA_BASES
    )
    """Base model names allowed a relaxed ``extra=`` policy."""
    ENFORCEMENT_INFRASTRUCTURE_BASES: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_INFRASTRUCTURE_BASES
    )
    """FLEXT infrastructure base class names exempt from enforcement checks."""
    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_CONSTANTS_SKIP_ATTRS
    )
    """Class-level attributes skipped during constants enforcement."""
    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_UTILITIES_EXEMPT_METHODS
    )
    """Methods exempt from static/classmethod enforcement on utilities."""
    ENFORCEMENT_UTILITIES_STATEFUL_ADAPTERS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_UTILITIES_STATEFUL_ADAPTERS
    )
    """Utilities classes whose protocol-adapter state is their contract
    (reasons in the package data); the static-shape rule skips them."""
    ENFORCEMENT_LAYER_ALLOWS: ClassVar[
        FlextTypingBase.MappingKV[str, frozenset[str]]
    ] = FlextConstantsEnforcementData.ENFORCEMENT_LAYER_ALLOWS
    """Per-layer inner-class kinds the cross-layer checks permit."""
    ENFORCEMENT_VALUE_OBJECT_BASES: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_VALUE_OBJECT_BASES
    )
    """Base-class names that require ``frozen=True`` configuration."""
    ENFORCEMENT_NESTED_MRO_MIN_DEPTH: ClassVar[int] = (
        FlextConstantsEnforcementData.ENFORCEMENT_NESTED_MRO_MIN_DEPTH
    )
    """Minimum qualname depth for a class to count as nested in a container."""
    ENFORCEMENT_CLASSVAR_EXEMPT_NAMES: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_CLASSVAR_EXEMPT_NAMES
    )
    """ClassVar names that are framework idioms (reasons in the package data)."""
    ENFORCE_FLEXT_CORE_PATH_MARKERS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCE_FLEXT_CORE_PATH_MARKERS
    )
    """Path fragments identifying flext-core source (ENFORCE-039 exemption)."""
    ENFORCE_NON_WORKSPACE_PATH_MARKERS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCE_NON_WORKSPACE_PATH_MARKERS
    )
    """Filesystem path fragments identifying third-party source."""
    ENFORCEMENT_ACCESSOR_RENAMES: ClassVar[
        FlextTypingBase.MappingKV[str, FlextTypingBase.StrPair]
    ] = FlextConstantsEnforcementData.ENFORCEMENT_ACCESSOR_RENAMES
    """Legacy accessor name → (canonical replacement, reason)."""
    ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS: ClassVar[frozenset[str]] = (
        FlextConstantsEnforcementData.ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS
    )
    """Accessor names owned by external framework contracts (reasons in data)."""


__all__: list[str] = ["FlextConstantsEnforcementSmellData"]
