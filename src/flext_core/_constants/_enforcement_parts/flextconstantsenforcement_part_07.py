"""Smell-rule enforcement constants loaded from JSON package-data."""

from __future__ import annotations

from typing import Final

from flext_core._constants._enforcement_data.loader import (
    ENFORCEMENT_SMELL_TAGS,
    SMELL_BEARTYPE_ROWS,
    SMELL_CODE_SMELL_ROWS,
    SMELL_FIX_STRATEGIES,
    SMELL_RULES_TEXT,
    SMELL_THRESHOLDS,
)

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsEnforcementSmellData:
    """JSON-loaded smell enforcement rules and thresholds."""

    ENFORCEMENT_SMELL_TAGS: Final[tuple[str, ...]] = ENFORCEMENT_SMELL_TAGS
    SMELL_THRESHOLDS: Final[t.IntMapping] = SMELL_THRESHOLDS
    ENFORCEMENT_SMELL_FIX_STRATEGIES: Final[
        t.MappingKV[str, t.MappingKV[str, t.JsonValue]]
    ] = {tag: strategy.model_dump() for tag, strategy in SMELL_FIX_STRATEGIES.items()}
    SMELL_FIX_STRATEGIES: Final[t.MappingKV[str, t.MappingKV[str, t.JsonValue]]] = {
        tag: strategy.model_dump() for tag, strategy in SMELL_FIX_STRATEGIES.items()
    }
    SMELL_RULES_TEXT: Final[t.StrPairMapping] = SMELL_RULES_TEXT
    SMELL_BEARTYPE_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], str], ...]
    ] = SMELL_BEARTYPE_ROWS
    SMELL_CODE_SMELL_ROWS: Final[
        tuple[tuple[str, str, str, str, tuple[str, ...], str], ...]
    ] = SMELL_CODE_SMELL_ROWS


__all__: list[str] = ["FlextConstantsEnforcementSmellData"]
