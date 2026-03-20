"""Real FlextContext API tests using flext_tests infrastructure."""

from __future__ import annotations

from time import perf_counter

import pytest
from flext_tests import tm
from hypothesis import given, settings, strategies as st

from flext_core import FlextContext


class TestAutomatedFlextContext:
    def test_create_set_get_has_remove_clear(self) -> None:
        _ = {"seed": "yes"}
        ctx = FlextContext.create()
        tm.ok(ctx.set("alpha", "value"), eq=True)
        tm.ok(ctx.get("alpha"), eq="value")
        tm.that(ctx.has("alpha"), eq=True)
        ctx.remove("alpha")
        tm.that(ctx.has("alpha"), eq=False)
        ctx.clear()
        tm.that(ctx.keys(), length=0)

    def test_create_metadata_overload_and_validate(self) -> None:
        ctx = FlextContext.create(
            operation_id="op-1",
            user_id="user-1",
        )
        tm.ok(ctx.get("operation_id"), eq="op-1")
        tm.ok(ctx.get("user_id"), eq="user-1")
        tm.ok(ctx.validate_context(), eq=True)

    def test_clone_merge_export_items_values(self) -> None:
        ctx = FlextContext.create()
        _ = ctx.set("k1", "v1")
        _ = ctx.set("k2", 2)
        cloned = ctx.clone()
        tm.ok(cloned.get("k1"), eq="v1")
        extra = {"k3": "v3"}
        merged = cloned.merge(extra)
        tm.that(merged is cloned, eq=True)
        tm.ok(cloned.get("k3"), eq="v3")
        exported = cloned.export()
        tm.that(exported, is_=dict)
        tm.that(exported, has=["global"])
        tm.that(cloned.keys(), has=["k1", "k2", "k3"])
        tm.that(cloned.values(), length_gte=3)
        tm.that(cloned.items(), length_gte=3)

    @given(
        key=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
        ),
        value=st.one_of(
            st.text(
                min_size=1,
                max_size=30,
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            ),
            st.integers(min_value=0, max_value=1000),
            st.booleans(),
        ),
    )
    @settings(max_examples=50)
    def test_set_get_roundtrip_property(
        self, key: str, value: str | int | bool
    ) -> None:
        ctx = FlextContext.create()
        tm.ok(ctx.set(key, value), eq=True)
        tm.ok(ctx.get(key), eq=value)

    @pytest.mark.performance
    def test_set_get_benchmark_cycle(self) -> None:
        ctx = FlextContext.create()
        keys = ["a", "b", "c", "d", "e"]
        value_factory = u.Tests.Factory.format_operation
        tm.that(callable(value_factory), eq=True)
        start = perf_counter()
        for i in range(400):
            key = str(keys[i % len(keys)])
            value = str(value_factory("v", i))
            tm.ok(ctx.set(key, value), eq=True)
            tm.ok(ctx.get(key), eq=value)
        tm.that(perf_counter() - start, gte=0.0)
