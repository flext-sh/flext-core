"""Tests for collections models full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, cast

import pytest
from pydantic import Field

from flext_core import r
from tests import c, m, t, u


class _Stats(m.Statistics):
    value: int | None = None


class _Rules(m.Rules):
    name: str = ""
    count: int = 0


class _Results(m.Results):
    value: int | bool | None = None
    data: dict[str, t.Tests.object] = Field(default_factory=dict)


def _default_tags() -> list[str]:
    return []


class _Options(m.Options):
    score: int | float | bool | None = None
    tags: Annotated[list[str], Field(default_factory=_default_tags)]


class _Config(m.Config):
    value: int = 0


def test_categories_clear_and_symbols_are_available() -> None:
    categories = m.Categories(categories={})
    categories.add_entries("x", ["a"])
    categories.clear()
    assert categories.categories == {}
    assert c.Errors.UNKNOWN_ERROR
    assert r[int].ok(1).is_success
    result = u.find([1], lambda value: value == 1)
    assert result.is_success and result.value == 1


def test_statistics_from_dict_and_none_conflict_resolution() -> None:
    config_map = t.ConfigMap({"value": 5})
    loaded = _Stats.from_mapping(cast("dict[str, t.MetadataValue]", config_map.root))
    assert loaded.value == 5
    assert _Stats._resolve_conflict(None, None) is None


def test_rules_merge_combines_model_dump_values() -> None:
    merged = _Rules.merge(_Rules(name="a", count=1), _Rules(name="b", count=2))
    assert merged.name == "b"
    assert merged.count == 2


def test_results_internal_conflict_paths_and_combine() -> None:
    entries: list[t.MetadataValue] = [
        {"ok": "v", "xs": [1, "a"]},
        {"ys": [2, 3.5]},
    ]
    merged_dict = _Results._merge_dicts(
        entries,
    )
    assert merged_dict["ok"] == "v"
    assert merged_dict["xs"] == [1, "a"]
    assert merged_dict["ys"] == [2, 3.5]
    assert _Results._resolve_conflict(None, None) is None
    assert _Results._resolve_conflict(True, False) is False
    combined = _Results.combine(_Results(value=1), _Results(value=2))
    assert combined.value == 2


def test_options_merge_conflict_paths_and_empty_merge_options() -> None:
    assert _Options._resolve_conflict(None, None) is None
    assert _Options._resolve_conflict(2, 3) == 5
    assert _Options._resolve_conflict([1], [2, "x"]) == [1, 2, "x"]
    assert _Options._resolve_conflict("a", "b") == "b"
    empty = _Options.merge_options()
    assert isinstance(empty, _Options)


def test_config_hash_from_mapping_and_non_hashable() -> None:
    loaded = _Config.from_mapping(t.ConfigMap(root={"value": 7}))
    assert loaded.value == 7
    with pytest.raises(TypeError, match="_Config objects are not hashable") as exc_info:
        hash(loaded)
    assert exc_info.value is not None
    assert "not hashable" in str(exc_info.value)
