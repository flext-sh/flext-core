"""Real API tests for FlextUtilities facade."""

from __future__ import annotations

from time import perf_counter

import pytest
from flext_tests import tm
from hypothesis import given, strategies as st

from flext_core import u
from tests import m, t


class TestAutomatedFlextUtilities:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, True),
            ("", True),
            ([], True),
            ({}, True),
            ("x", False),
            ([1], False),
            ({"a": 1}, False),
            (0, True),
        ],
    )
    def test_empty(self, value: t.NormalizedValue | None, expected: bool) -> None:
        tm.that(u.empty(value), eq=expected)

    def test_generate_ulid_and_uuid(self) -> None:
        generated_ulid = u.generate("ulid")
        generated_uuid = u.generate()
        tm.that(generated_ulid, is_=str, none=False)
        tm.that(generated_uuid, is_=str, none=False)
        tm.that(len(generated_ulid), gt=0)
        tm.that(len(generated_uuid), gt=0)

    def test_type_guards_and_collection_helpers(self) -> None:
        model = m.Tests.User(id="1", name="Test", email="test@test.com")
        numbers = [1, 2, 3]
        tm.that(u.is_base_model(model), eq=True)
        tm.that(u.is_scalar(42), eq=True)
        tm.that(u.is_config_value({"alpha": 1}), eq=True)

        mapped = u.map(numbers, lambda v: v * 2)
        chunks = u.chunk(numbers, 2)
        unique = u.unique([1, 1, 2, 3, 3])
        tm.that(mapped, eq=[2, 4, 6])
        tm.that(chunks, eq=[[1, 2], [3]])
        tm.that(unique, eq=[1, 2, 3])

    @given(
        st.one_of(
            st.none(),
            st.text(),
            st.lists(st.integers()),
            st.dictionaries(st.text(), st.integers()),
        )
    )
    def test_hypothesis_empty_returns_bool(
        self, value: t.NormalizedValue | None
    ) -> None:
        result = u.empty(value)
        tm.that(result, is_=bool)

    @given(st.text())
    def test_hypothesis_generate_always_non_empty(self, _value: str) -> None:
        generated = u.generate("ulid")
        tm.that(generated, is_=str, none=False)
        tm.that(len(generated), gt=0)

    def test_benchmark_generate(self) -> None:
        users = [
            m.Tests.User(id="1", name="Test", email="test@test.com") for _ in range(3)
        ]
        tm.that(len(users), gt=0)
        start = perf_counter()
        for _ in range(2000):
            generated = u.generate("ulid")
            tm.that(len(generated), gt=0)
        elapsed = perf_counter() - start
        tm.that(elapsed, gt=0.0)
