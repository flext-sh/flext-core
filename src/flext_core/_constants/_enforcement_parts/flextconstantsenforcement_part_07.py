"""Smell-rule enforcement constants loaded from JSON package-data."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .._enforcement_data import (
    ENFORCEMENT_SMELL_TAGS,
    SMELL_RULES_TEXT,
    SMELL_THRESHOLDS,
)

if TYPE_CHECKING:
    from ..._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementSmellData:
    """JSON-loaded smell enforcement thresholds, tags and rule text."""

    ENFORCEMENT_CATALOG_RESOURCE: ClassVar[str] = "catalog.json"
    """Package-data resource holding the enforcement rule catalog."""

    ENFORCEMENT_SMELL_TAGS: ClassVar[t.VariadicTuple[str]] = ENFORCEMENT_SMELL_TAGS
    SMELL_THRESHOLDS: ClassVar[t.IntMapping] = SMELL_THRESHOLDS
    SMELL_RULES_TEXT: ClassVar[t.StrPairMapping] = SMELL_RULES_TEXT


__all__: list[str] = ["FlextConstantsEnforcementSmellData"]
