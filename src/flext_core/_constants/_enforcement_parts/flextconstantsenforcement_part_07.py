"""Enforcement rule constants loaded from JSON package data."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._enforcement_data import (
    ENFORCEMENT_PREDICATE_SPECS,
    ENFORCEMENT_RULES_TEXT,
    ENFORCEMENT_SMELL_TAGS,
    ENFORCEMENT_TAG_CATEGORY,
    ENFORCEMENT_TAG_LAYER,
    SMELL_RULES_TEXT,
    SMELL_THRESHOLDS,
)
from .flextconstantsenforcement_part_04 import FlextConstantsEnforcementRules

if TYPE_CHECKING:
    from ..._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementSmellData:
    """Runtime rules, smell thresholds and rule text from package data."""

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
    ENFORCEMENT_PREDICATE_SPECS: ClassVar[t.MappingKV[str, t.JsonMapping]] = (
        ENFORCEMENT_PREDICATE_SPECS
    )
    """Runtime tag → raw predicate kind and parameters (typed by the engine)."""


__all__: list[str] = ["FlextConstantsEnforcementSmellData"]
