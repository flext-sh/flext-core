"""Behavior contract for flext_core collection utilities — public API only."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from flext_core import u
from flext_tests import tm
from tests.models import m

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.typings import t


class TestsFlextCoreUtilitiesCollection:
    """Behavior contract for u.map / u.find / u.filter / u.count / u.process / u.merge_mappings."""

    def test_normalize_domain_event_data_flattens_public_payloads(
        self, tmp_path: Path
    ) -> None:
        workspace_root = tmp_path / "flext"
        workspace_root.mkdir()
        config_payload = m.ConfigMap.model_validate({
            "event": "sync-users",
            "workspace_root": workspace_root,
            "attempt_count": 2,
            "ignored": None,
        })
        mapping_payload: t.JsonMapping = {"tenant": "acme", "retry": 1, "skip": None}

        normalized_config = u.normalize_domain_event_data(config_payload)
        normalized_mapping_payload = u.normalize_domain_event_data(mapping_payload)
        normalized_none = u.normalize_domain_event_data(None)

        tm.that(normalized_none, eq={})
        tm.that(
            normalized_config,
            eq={
                "event": "sync-users",
                "workspace_root": str(workspace_root),
                "attempt_count": 2,
            },
        )
        tm.that(normalized_mapping_payload, eq={"tenant": "acme", "retry": 1})

    # --- map -------------------------------------------------------------

    @pytest.mark.parametrize(
        ("items", "mapper", "expected"),
        [
            ([1, 2, 3], lambda x: x * 2, [2, 4, 6]),
            ((1, 2), lambda x: x + 1, (2, 3)),
            ({"a": 1, "b": 2}, lambda x: x * 10, {"a": 10, "b": 20}),
        ],
    )
    def test_map_applies_function_to_each_element(
        self,
        items: t.JsonList | tuple[t.JsonValue, ...] | t.JsonMapping,
        mapper: Callable[[t.JsonValue], t.JsonValue],
        expected: t.JsonValue,
    ) -> None:
        tm.that(u.map(items, mapper), eq=expected)

    # --- find ------------------------------------------------------------

    @pytest.mark.parametrize(
        ("items", "predicate", "expected", "expect_found"),
        [
            ([1, 2, 3], lambda x: x == 2, 2, True),
            ({"a": 1, "b": 4}, lambda x: x == 4, 4, True),
            ([1, 3, 5], lambda x: x == 2, None, False),
        ],
    )
    def test_find_returns_matching_element_or_failure(
        self,
        *,
        items: t.JsonList | tuple[t.JsonValue, ...] | t.JsonMapping,
        predicate: Callable[[t.JsonValue], bool],
        expected: t.JsonValue,
        expect_found: bool,
    ) -> None:
        result = u.find(items, predicate)
        if expect_found:
            tm.ok(result)
            tm.that(result.value, eq=expected)
        else:
            tm.fail(result)

    def test_find_returns_failure_when_mapping_has_no_matching_value(self) -> None:
        result = u.find(
            {"tenant": "acme", "mode": "full"}, lambda value: value == "delta"
        )

        tm.fail(result)
        tm.that(result.error, eq="No matching item found")

    # --- filter ----------------------------------------------------------

    @pytest.mark.parametrize(
        ("items", "predicate", "mapper", "expected"),
        [
            ([1, 2, 3, 4], lambda x: x % 2 == 0, None, [2, 4]),
            ([1, 2, 3, 4], lambda x: x > 2, lambda x: x * 2, [6, 8]),
            ({"a": 1, "b": 2, "c": 3}, lambda v: v % 2 != 0, None, {"a": 1, "c": 3}),
            ({"a": 1, "b": 4}, lambda v: v > 2, lambda v: v * 2, {"b": 8}),
            ([1, 3, 5], lambda x: x > 10, None, []),
            ([2, 4, 6], lambda x: x % 2 == 0, None, [2, 4, 6]),
        ],
    )
    def test_filter_keeps_matching_and_optionally_maps(
        self,
        items: t.JsonList | tuple[t.JsonValue, ...] | t.JsonMapping,
        predicate: Callable[[t.JsonValue], bool],
        mapper: Callable[[t.JsonValue], t.JsonValue] | None,
        expected: t.JsonValue,
    ) -> None:
        tm.that(u.filter(items, predicate, mapper=mapper), eq=expected)

    # --- count -----------------------------------------------------------

    @pytest.mark.parametrize(
        ("items", "predicate", "expected"),
        [([1, 2, 3, 4], None, 4), ([1, 2, 3, 4], lambda x: x % 2 == 0, 2)],
    )
    def test_count_returns_total_or_matching(
        self,
        items: t.JsonList,
        predicate: Callable[[t.JsonValue], bool] | None,
        expected: int,
    ) -> None:
        tm.that(u.count(items, predicate), eq=expected)

    # --- process ---------------------------------------------------------

    @pytest.mark.parametrize(
        ("items", "processor", "predicate", "expected"),
        [
            ([1, 2, 3], lambda x: x * 2, None, [2, 4, 6]),
            ([1, 2, 3], lambda x: x * 2, lambda x: x > 1, [4, 6]),
            (["a", "b", "c"], lambda x: x.upper(), None, ["A", "B", "C"]),
            ([], lambda x: x * 2, None, []),
        ],
    )
    def test_process_applies_processor_with_optional_predicate(
        self,
        items: t.JsonList,
        processor: Callable[[t.JsonValue], t.JsonValue],
        predicate: Callable[[t.JsonValue], bool] | None,
        expected: t.JsonList,
    ) -> None:
        result = u.process(items, processor, on_error="collect", predicate=predicate)
        tm.ok(result)
        tm.that(result.value, eq=expected)

    def test_process_supports_skip_and_fail_error_modes(self) -> None:
        def project_identifier(value: t.JsonValue) -> str:
            if value == 2:
                error_message = "cannot process item"
                raise ValueError(error_message)
            return f"item:{value}"

        skipped = u.process([1, 2, 3], project_identifier, on_error="skip")
        failed = u.process([1, 2, 3], project_identifier, on_error="fail")

        tm.ok(skipped)
        tm.that(skipped.value, eq=["item:1", "item:3"])
        tm.fail(failed)
        tm.that(failed.error, eq="Processing failed for item: 2")

    # --- merge_mappings --------------------------------------------------

    def test_merge_mappings_deep_combines_nested_keys(self) -> None:
        base: t.MappingKV[str, t.JsonValue] = {"a": 1, "b": {"x": 1}}
        other: t.MappingKV[str, t.JsonValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base, other)
        tm.ok(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)

    def test_merge_mappings_override_replaces_values(self) -> None:
        base: t.MappingKV[str, t.JsonValue] = {"a": 1, "b": {"x": 1}}
        other: t.MappingKV[str, t.JsonValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base, other, strategy="override")
        tm.ok(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)


__all__: list[str] = ["TestsFlextCoreUtilitiesCollection"]
