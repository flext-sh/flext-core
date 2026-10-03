"""Enforcement rule constants loaded from JSON package data.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._constants._enforcement_data import (
    ENFORCE_FLEXT_CORE_PATH_MARKERS,
    ENFORCE_NON_WORKSPACE_PATH_MARKERS,
    ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS,
    ENFORCEMENT_ACCESSOR_RENAMES,
    ENFORCEMENT_CLASSVAR_EXEMPT_NAMES,
    ENFORCEMENT_CONSTANTS_SKIP_ATTRS,
    ENFORCEMENT_INFRASTRUCTURE_BASES,
    ENFORCEMENT_LAYER_ALLOWS,
    ENFORCEMENT_NESTED_MRO_MIN_DEPTH,
    ENFORCEMENT_PREDICATE_SPECS,
    ENFORCEMENT_RECURSIVE_TAGS,
    ENFORCEMENT_RELAXED_EXTRA_BASES,
    ENFORCEMENT_RULES_TEXT,
    ENFORCEMENT_SMELL_TAGS,
    ENFORCEMENT_TAG_CATEGORY,
    ENFORCEMENT_TAG_COLLECT,
    ENFORCEMENT_TAG_LAYER,
    ENFORCEMENT_UTILITIES_EXEMPT_METHODS,
    ENFORCEMENT_VALUE_OBJECT_BASES,
    SMELL_RULES_TEXT,
    SMELL_THRESHOLDS,
)
from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_04 import (
    FlextConstantsEnforcementRules,
)

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t

class FlextConstantsEnforcementSmellData:
    """Runtime rules, exemptions, smell thresholds and rule text from package data."""

    ENFORCEMENT_CATALOG_RESOURCE: ClassVar[str] = "catalog.json"
    """Package-data resource holding the enforcement rule catalog."""

    ENFORCEMENT_SMELL_TAGS: ClassVar[t.VariadicTuple[str]] = ENFORCEMENT_SMELL_TAGS
    SMELL_THRESHOLDS: ClassVar[t.IntMapping] = SMELL_THRESHOLDS
    SMELL_RULES_TEXT: ClassVar[t.StrPairMapping] = SMELL_RULES_TEXT
    ENFORCEMENT_RULES_TEXT: ClassVar[t.StrPairMapping] = ENFORCEMENT_RULES_TEXT
    """Problem/fix text templates per runtime tag; consumed by the emitter."""
    ENFORCEMENT_TAG_CATEGORY: ClassVar[
        t.MappingKV[str, FlextConstantsEnforcementRules.EnforcementCategory]
    ] = ENFORCEMENT_TAG_CATEGORY
    """Runtime tag → category dispatching its item collection."""
    ENFORCEMENT_TAG_LAYER: ClassVar[t.StrMapping] = ENFORCEMENT_TAG_LAYER
    """Runtime tag → facade layer for ATTR-category rules."""
    ENFORCEMENT_TAG_COLLECT: ClassVar[t.StrMapping] = ENFORCEMENT_TAG_COLLECT
    """NAMESPACE tag → item collection strategy of the runtime engine."""
    ENFORCEMENT_RECURSIVE_TAGS: ClassVar[frozenset[str]] = ENFORCEMENT_RECURSIVE_TAGS
    """Tags whose scan recurses into inner namespace classes."""
    ENFORCEMENT_PREDICATE_SPECS: ClassVar[t.MappingKV[str, t.JsonMapping]] = (
        ENFORCEMENT_PREDICATE_SPECS
    )
    """Runtime tag → raw predicate kind and parameters (typed by the engine)."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: ClassVar[frozenset[str]] = (
        ENFORCEMENT_RELAXED_EXTRA_BASES
    )
    """Base model names allowed a relaxed ``extra=`` policy."""
    ENFORCEMENT_INFRASTRUCTURE_BASES: ClassVar[frozenset[str]] = (
        ENFORCEMENT_INFRASTRUCTURE_BASES
    )
    """FLEXT infrastructure base class names exempt from enforcement checks."""
    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: ClassVar[frozenset[str]] = (
        ENFORCEMENT_CONSTANTS_SKIP_ATTRS
    )
    """Class-level attributes skipped during constants enforcement."""
    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: ClassVar[frozenset[str]] = (
        ENFORCEMENT_UTILITIES_EXEMPT_METHODS
    )
    """Methods exempt from static/classmethod enforcement on utilities."""
    ENFORCEMENT_LAYER_ALLOWS: ClassVar[t.MappingKV[str, frozenset[str]]] = (
        ENFORCEMENT_LAYER_ALLOWS
    )
    """Per-layer inner-class kinds the cross-layer checks permit."""
    ENFORCEMENT_VALUE_OBJECT_BASES: ClassVar[frozenset[str]] = (
        ENFORCEMENT_VALUE_OBJECT_BASES
    )
    """Base-class names that require ``frozen=True`` configuration."""
    ENFORCEMENT_NESTED_MRO_MIN_DEPTH: ClassVar[int] = ENFORCEMENT_NESTED_MRO_MIN_DEPTH
    """Minimum qualname depth for a class to count as nested in a container."""
    ENFORCEMENT_CLASSVAR_EXEMPT_NAMES: ClassVar[frozenset[str]] = (
        ENFORCEMENT_CLASSVAR_EXEMPT_NAMES
    )
    """ClassVar names that are framework idioms (reasons in the package data)."""
    ENFORCE_FLEXT_CORE_PATH_MARKERS: ClassVar[frozenset[str]] = (
        ENFORCE_FLEXT_CORE_PATH_MARKERS
    )
    """Path fragments identifying flext-core source (ENFORCE-039 exemption)."""
    ENFORCE_NON_WORKSPACE_PATH_MARKERS: ClassVar[frozenset[str]] = (
        ENFORCE_NON_WORKSPACE_PATH_MARKERS
    )
    """Filesystem path fragments identifying third-party source."""
    ENFORCEMENT_ACCESSOR_RENAMES: ClassVar[t.MappingKV[str, t.StrPair]] = (
        ENFORCEMENT_ACCESSOR_RENAMES
    )
    """Legacy accessor name → (canonical replacement, reason)."""
    ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS: ClassVar[frozenset[str]] = (
        ENFORCEMENT_ACCESSOR_EXTERNAL_CONTRACTS
    )
    """Accessor names owned by external framework contracts (reasons in data)."""


__all__: list[str] = ["FlextConstantsEnforcementSmellData"]
