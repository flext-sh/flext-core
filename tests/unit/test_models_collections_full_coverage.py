from __future__ import annotations

from pydantic import Field
from typing import cast

from flext_core import c, m, r, t, u


class _Stats(m.CollectionsStatistics):
    value: int | None = None


class _Rules(m.Rules):
    name: str = ""
    count: int = 0


class _Results(m.CollectionsResults):
    value: int | bool | None = None
    data: dict[str, t.GeneralValueType] = Field(default_factory=dict)


class _Options(m.CollectionsOptions):
    score: int | float | bool | None = None
    tags: list[t.GeneralValueType] = Field(default_factory=list)


class _Config(m.CollectionsConfig):
    value: int = 0


def test_categories_clear_and_symbols_are_available() -> None:
    categories = m.Categories()
    categories.add_entries("x", ["a"])
    categories.clear()
    assert categories.categories == {}
    assert c.Errors.UNKNOWN_ERROR
    assert r[int].ok(1).is_success
    assert isinstance(u.Collection.find([1], lambda value: value == 1), int)


def test_statistics_from_dict_and_none_conflict_resolution() -> None:
    config_map = t.ConfigMap.model_validate({"value": 5})
    loaded = _Stats.from_dict(config_map)
    assert loaded.value == 5
    assert _Stats._resolve_aggregate_conflict(None, None) is None


def test_rules_merge_combines_model_dump_values() -> None:
    merged = _Rules.merge(_Rules(name="a", count=1), _Rules(name="b", count=2))
    assert merged.name == "b"
    assert merged.count == 2


def test_results_internal_conflict_paths_and_combine() -> None:
    merged_dict = _Results._merge_dicts(
        cast(
            "list[t.ConfigMapValue]",
            [
                {"ok": "v", "xs": [1, "a", object()]},
                {"ys": [2, None, 3.5]},
            ],
        )
    )
    assert merged_dict["ok"] == "v"
    assert merged_dict["xs"] == [1, "a"]
    assert merged_dict["ys"] == [2, None, 3.5]

    assert _Results._resolve_aggregate_conflict(None, None) is None
    assert _Results._resolve_aggregate_conflict(True, False) is False

    combined = _Results.combine(_Results(value=1), _Results(value=2))
    assert combined.value == 2


def test_options_merge_conflict_paths_and_empty_merge_options() -> None:
    assert _Options._resolve_merge_conflict(None, None) is None
    assert _Options._resolve_merge_conflict(2, 3) == 5
    assert _Options._resolve_merge_conflict([1], [2, "x"]) == [1, 2, "x"]
    assert _Options._resolve_merge_conflict("a", "b") == "b"

    empty = _Options.merge_options()
    assert isinstance(empty, _Options)


def test_config_hash_from_mapping_and_non_hashable() -> None:
    loaded = _Config.from_mapping(t.ConfigMap(root={"value": 7}))
    assert loaded.value == 7

    try:
        hash(loaded)
    except TypeError as exc:
        assert "not hashable" in str(exc)
    else:
        msg = "Expected TypeError for mutable config hash"
        raise AssertionError(msg)
