"""JSON package data for enforcement: the rule catalog and the smell rules.

``catalog.json`` is the enforcement rule catalog; flext-core validates it into
``m.EnforcementCatalog`` at its consumer. ``smells.json`` carries the smell
thresholds, tags and rule text loaded here.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.resources
from types import MappingProxyType
from typing import TYPE_CHECKING

from pydantic import BaseModel

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


def _load_smell_data() -> _SmellData:
    """Load and validate smells.json from package data."""
    text = (
        importlib.resources
        .files(__package__)
        .joinpath("smells.json")
        .read_text(encoding="utf-8")
    )
    return _SmellData.model_validate_json(text)


_SMELL_DATA: _SmellData = _load_smell_data()

ENFORCEMENT_SMELL_TAGS: tuple[str, ...] = _SMELL_DATA.tags
SMELL_THRESHOLDS: t.MappingKV[str, int] = MappingProxyType(
    _SMELL_DATA.thresholds.model_dump()
)
SMELL_RULES_TEXT: t.MappingKV[str, tuple[str, str]] = MappingProxyType(
    _SMELL_DATA.rules_text
)

__all__: list[str] = ["ENFORCEMENT_SMELL_TAGS", "SMELL_RULES_TEXT", "SMELL_THRESHOLDS"]
