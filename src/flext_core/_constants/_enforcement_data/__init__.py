"""JSON package-data loader for enforcement smell rules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.resources
from typing import ClassVar, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..._typings.base import FlextTypingBase as t


class _SmellThresholds(BaseModel):
    params: int
    returns: int
    nesting: int
    fn_cx: int
    file_cx: int


class _SmellFixStrategy(BaseModel):
    auto: bool
    fixer: str | None
    description: str


class _SmellCatalogRow(BaseModel):
    id: str
    severity: str
    tag: str
    anchor: str
    skills: tuple[str, ...]
    description: str


class _SmellData(BaseModel):
    thresholds: _SmellThresholds
    tags: tuple[str, ...]
    fix_strategy: dict[str, _SmellFixStrategy]
    rules_text: dict[str, tuple[str, str]]
    beartype_rows: tuple[_SmellCatalogRow, ...] = Field(alias="beartype_rows")
    code_smell_rows: tuple[_SmellCatalogRow, ...] = Field(alias="code_smell_rows")

    model_config = {"populate_by_name": True}


def _load_smell_data() -> _SmellData:
    """Load and validate smells.json from package data."""
    text = (
        importlib.resources
        .files(__package__)
        .joinpath("smells.json")
        .read_text(encoding="utf-8")
    )
    return _SmellData.model_validate_json(text)


_SMELL_DATA: ClassVar[_SmellData] = _load_smell_data()

ENFORCEMENT_SMELL_TAGS: ClassVar[tuple[str, ...]] = _SMELL_DATA.tags
SMELL_THRESHOLDS: ClassVar[t.MappingKV[str, int]] = _SMELL_DATA.thresholds.model_dump()
SMELL_FIX_STRATEGIES: ClassVar[t.MappingKV[str, _SmellFixStrategy]] = (
    _SMELL_DATA.fix_strategy
)
SMELL_RULES_TEXT: ClassVar[t.MappingKV[str, tuple[str, str]]] = _SMELL_DATA.rules_text
SMELL_BEARTYPE_ROWS: ClassVar[
    tuple[tuple[str, str, str, str, tuple[str, ...], str], ...]
] = tuple(
    (row.id, row.severity, row.tag, row.anchor, row.skills, row.description)
    for row in _SMELL_DATA.beartype_rows
)
SMELL_CODE_SMELL_ROWS: ClassVar[
    tuple[tuple[str, str, str, str, tuple[str, ...], str], ...]
] = tuple(
    (row.id, row.severity, row.tag, row.anchor, row.skills, row.description)
    for row in _SMELL_DATA.code_smell_rows
)

__all__: list[str] = [
    "ENFORCEMENT_SMELL_TAGS",
    "SMELL_FIX_STRATEGIES",
    "SMELL_RULES_TEXT",
    "SMELL_THRESHOLDS",
]
