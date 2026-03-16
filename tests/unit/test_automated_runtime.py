"""Real API tests for FlextRuntime."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest
from flext_tests import tb, tm, tt
from hypothesis import given, strategies as st
from pydantic import BaseModel

from flext_core import FlextRuntime


class TestAutomatedFlextRuntime:
    def test_is_base_model_and_collections(self) -> None:
        model = tb.Tests.Model.user()
        tm.that(FlextRuntime.is_base_model(model), eq=True)
        tm.that(FlextRuntime.is_base_model("value"), eq=False)

        dict_like = {"alpha": 1}
        list_like = [1, 2, 3]
        tm.that(FlextRuntime.is_dict_like(dict_like), eq=True)
        tm.that(FlextRuntime.is_dict_like(list_like), eq=False)
        tm.that(FlextRuntime.is_list_like(list_like), eq=True)
        tm.that(FlextRuntime.is_list_like("abc"), eq=False)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("valid_name", True), ("invalid-name", False)],
    )
    def test_is_valid_identifier(self, value: str, expected: bool) -> None:
        tm.that(FlextRuntime.is_valid_identifier(value), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [('{"key":"value"}', True), ("not-json", False)],
    )
    def test_is_valid_json(self, value: str, expected: bool) -> None:
        tm.that(FlextRuntime.is_valid_json(value), eq=expected)

    def test_normalize_container_and_metadata(self) -> None:
        payload = {"path": Path("/tmp/test"), "num": 7}
        normalized_container = FlextRuntime.normalize_to_container(payload)
        normalized_metadata = FlextRuntime.normalize_to_metadata(payload)
        tm.that(FlextRuntime.is_base_model(normalized_container), eq=True)
        tm.that(FlextRuntime.is_dict_like(normalized_metadata), eq=True)

    def test_safe_get_attribute_and_structlog_state(self) -> None:
        class RuntimeProbe(BaseModel):
            marker: str = "ok"

        probe = RuntimeProbe()
        tm.that(FlextRuntime.safe_get_attribute(probe, "marker"), eq="ok")
        tm.that(
            FlextRuntime.safe_get_attribute(probe, "missing", default="fallback"),
            eq="fallback",
        )

        FlextRuntime.reset_structlog_state_for_testing()
        tm.that(FlextRuntime.is_structlog_configured(), eq=False)
        FlextRuntime.ensure_structlog_configured()
        tm.that(FlextRuntime.is_structlog_configured(), eq=True)

    @given(st.text())
    def test_hypothesis_identifier_guard_returns_bool(self, value: str) -> None:
        result = FlextRuntime.is_valid_identifier(value)
        tm.that(result, is_=bool)

    @given(
        st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
        )
    )
    def test_hypothesis_type_guards_return_bool(
        self, value: float | str | bool
    ) -> None:
        tm.that(FlextRuntime.is_dict_like(value), is_=bool)
        tm.that(FlextRuntime.is_list_like(value), is_=bool)
        tm.that(FlextRuntime.is_valid_json(value), is_=bool)

    def test_benchmark_type_guards(self) -> None:
        models = tt.batch("user", count=8)
        dict_like = {"alpha": 1, "beta": 2}
        start = perf_counter()
        for _ in range(800):
            for model in models:
                _ = FlextRuntime.is_base_model(model)
                _ = FlextRuntime.is_dict_like(dict_like)
        elapsed = perf_counter() - start
        tm.that(elapsed, gt=0.0)
