"""Behavior contract for flext_core collection utilities — public API only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests import c, m, t, tm, u


def _is_same(expected: object) -> Callable[[object], bool]:
    """Build a typed identity predicate for ``tm.that(where=...)``.

    Returns:
        The resulting ``Callable[[object], bool]``.

    """

    def matches(candidate: object) -> bool:
        return candidate is expected

    return matches


class TestsFlextCoreUtilitiesCollection:
    """Behavior contract for the collection utilities.

    Covers ``u.map``, ``u.find``, ``u.filter``, ``u.count``, ``u.process``,
    and ``u.merge_mappings``.
    """

    @staticmethod
    def test_domain_event_retains_its_validated_payload(
        tmp_path: Path,
    ) -> None:
        """Keep the validated payload and its native values without normalization."""
        repository_root = tmp_path / "flext"
        repository_root.mkdir()
        config_payload = m.ConfigMap.model_validate({
            c.HandlerType.EVENT: "sync-users",
            "repository_root": repository_root,
            "attempt_count": 2,
            "ignored": None,
        })
        event = m.DomainEvent(
            event_type="sync-users",
            aggregate_id="acme",
            data=config_payload,
        )
        tm.that(event.data, where=_is_same(config_payload))
        tm.that(
            event.data.root["repository_root"],
            where=_is_same(repository_root),
        )
        tm.that(event.data.root["ignored"], none=True)

    # --- map -------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("items", "mapper", "expected"),
        [
            ([1, 2, 3], u.Tests.double, [2, 4, 6]),
            ((1, 2), (1).__add__, (2, 3)),
            ({"a": 1, "b": 2}, (10).__mul__, {"a": 10, "b": 20}),
            ({1, 2}, u.Tests.double, {2, 4}),
            (frozenset({1, 2}), u.Tests.double, frozenset({2, 4})),
            (range(3), u.Tests.double, [0, 2, 4]),
        ],
    )
    def test_map_applies_function_to_each_element(
        items: Sequence[int] | Mapping[str, int] | set[int] | frozenset[int],
        mapper: Callable[[int], int],
        expected: Sequence[int] | Mapping[str, int] | set[int] | frozenset[int],
    ) -> None:
        """Test map applies function to each element."""
        tm.that(u.map(items, mapper), eq=expected)

    # --- find ------------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("items", "predicate", "expected", "expect_found"),
        [
            ([1, 2, 3], (2).__eq__, 2, True),
            ({"a": 1, "b": 4}, (4).__eq__, 4, True),
            ([1, 3, 5], (2).__eq__, None, False),
        ],
    )
    def test_find_returns_matching_element_or_failure(
        *,
        items: Sequence[int] | Mapping[str, int],
        predicate: Callable[[int], bool],
        expected: int | None,
        expect_found: bool,
    ) -> None:
        """Test find returns matching element or failure."""
        result = u.find(items, predicate)
        if expect_found:
            tm.ok(result, eq=expected)
        else:
            tm.fail(result)

    @staticmethod
    def test_find_returns_failure_when_mapping_has_no_matching_value() -> None:
        """Test find returns failure when mapping has no matching value."""
        result = u.find(
            {"tenant": "acme", "mode": "full"},
            "delta".__eq__,
        )

        tm.fail(result)
        tm.that(result.error, eq="No matching item found")

    # --- filter ----------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("items", "predicate", "mapper", "expected"),
        [
            ([1, 2, 3, 4], u.Tests.is_even, None, [2, 4]),
            ([1, 2, 3, 4], (2).__lt__, u.Tests.double, [6, 8]),
            (
                {"a": 1, "b": 2, "c": 3},
                u.Tests.is_odd,
                None,
                {"a": 1, "c": 3},
            ),
            (
                {"a": 1, "b": 4},
                (2).__lt__,
                u.Tests.double,
                {"b": 8},
            ),
            ([1, 3, 5], (10).__lt__, None, []),
            ([2, 4, 6], u.Tests.is_even, None, [2, 4, 6]),
            ((1, 2, 3, 4), u.Tests.is_even, None, (2, 4)),
            ((1, 2, 3, 4), (2).__lt__, u.Tests.double, (6, 8)),
            (range(4), u.Tests.is_even, None, [0, 2]),
        ],
    )
    def test_filter_keeps_matching_and_optionally_maps(
        items: Sequence[int] | Mapping[str, int],
        predicate: Callable[[int], bool],
        mapper: Callable[[int], int] | None,
        expected: Sequence[int] | Mapping[str, int],
    ) -> None:
        """Test filter keeps matching and optionally maps."""
        tm.that(u.filter(items, predicate, mapper=mapper), eq=expected)

    # --- count -----------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("items", "predicate", "expected"),
        [([1, 2, 3, 4], None, 4), ([1, 2, 3, 4], u.Tests.is_even, 2)],
    )
    def test_count_returns_total_or_matching(
        items: Sequence[int],
        predicate: Callable[[int], bool] | None,
        expected: int,
    ) -> None:
        """Test count returns total or matching."""
        tm.that(u.count(items, predicate), eq=expected)

    # --- process ---------------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        ("items", "processor", "predicate", "expected"),
        [
            ([1, 2, 3], u.Tests.double, None, [2, 4, 6]),
            ([1, 2, 3], u.Tests.double, (1).__lt__, [4, 6]),
            ([], u.Tests.double, None, []),
        ],
    )
    def test_process_applies_processor_with_optional_predicate(
        items: Sequence[int],
        processor: Callable[[int], int],
        predicate: Callable[[int], bool] | None,
        expected: Sequence[int],
    ) -> None:
        """Test process applies processor with optional predicate."""
        result = u.process(items, processor, predicate=predicate)
        tm.ok(result, eq=expected)

    @staticmethod
    def test_process_uppercases_each_string() -> None:
        """Apply a string processor without widening its input contract."""
        tm.ok(u.process(["a", "b", "c"], str.upper), eq=["A", "B", "C"])

    @staticmethod
    def test_process_first_failure_ends_run_carrying_its_exception() -> None:
        """Test process first failure ends run carrying its exception."""
        visited: list[t.JsonValue] = []
        raised: list[ValueError] = []

        def project_identifier(value: t.JsonValue) -> str:
            visited.append(value)
            if value in {2, 3}:
                error = ValueError(f"cannot process item {value}")
                raised.append(error)
                raise error
            return f"item:{value}"

        failed = u.process([1, 2, 3], project_identifier)

        tm.fail(failed)
        tm.that(visited, eq=[1, 2])
        tm.ok(failed.exception is raised[0])
        tm.that(
            failed.error,
            eq=c.ERR_COLLECTION_PROCESSING_FAILED_FOR_ITEM.format(
                item=2,
                error=str(raised[0]),
            ),
        )

    # --- merge_mappings --------------------------------------------------

    @staticmethod
    def test_merge_mappings_deep_combines_nested_keys() -> None:
        """Test merge mappings deep combines nested keys."""
        base: Mapping[str, t.JsonValue] = {"a": 1, "b": {"x": 1}}
        other: Mapping[str, t.JsonValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base, other)
        tm.ok(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)

    @staticmethod
    def test_merge_mappings_override_replaces_values() -> None:
        """Test merge mappings override replaces values."""
        base: Mapping[str, t.JsonValue] = {"a": 1, "b": {"x": 1}}
        other: Mapping[str, t.JsonValue] = {"b": {"y": 2}, "c": 3}
        result = u.merge_mappings(base, other, strategy=c.MergeStrategy.OVERRIDE)
        tm.ok(result)
        tm.that(result.value["a"], eq=1)
        tm.that(result.value["c"], eq=3)
        tm.that(result.value["b"], is_=dict)
