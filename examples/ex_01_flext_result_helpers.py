"""Advanced result example sections kept under the module LOC cap."""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
    Sequence,
)

from flext_core import p, r

from .shared import ExamplesFlextShared


class Ex01ResultAdvancedSections(ExamplesFlextShared):
    """Advanced ``r`` sections shared by the executable example."""

    def collections_and_resource(self) -> None:
        """Exercise collection helpers and resource management wrapper."""
        self.section("collections_and_resource")

        def to_even(n: int) -> p.Result[int]:
            if n % 2 == 0:
                return r[int].ok(n)
            return r[int].fail(f"odd:{n}")

        self.audit_check(
            "traverse.success",
            r[Sequence[int]].traverse([2, 4, 6], to_even, fail_fast=True).unwrap_or([]),
        )
        self.audit_check(
            "traverse.fail_fast",
            r[Sequence[int]].traverse([2, 3, 4], to_even, fail_fast=True).error,
        )
        self.audit_check(
            "traverse.collect",
            r[Sequence[int]].traverse([1, 3, 5], to_even, fail_fast=False).error,
        )
        acc_ok = r.accumulate_errors(r[int].ok(1), r[int].ok(2))
        acc_fail = r.accumulate_errors(
            r[int].ok(1),
            r[int].fail("e1"),
            r[int].fail("e2"),
        )
        self.audit_check("accumulate_errors.success", acc_ok.unwrap_or([]))
        self.audit_check("accumulate_errors.failure", acc_fail.error)
        cleaned_values: MutableSequence[int] = []

        def make_handle() -> ExamplesFlextShared.Handle:
            return self.Handle(value=21)

        def clean_handle(handle: ExamplesFlextShared.Handle) -> None:
            handle.cleaned = True
            cleaned_values.append(handle.value)

        success_resource = r[int].with_resource(
            make_handle,
            lambda handle: r[int].ok(handle.value * 2),
            cleanup=clean_handle,
        )
        failure_resource = r[int].with_resource(
            make_handle,
            lambda _: r[int].fail("resource op failed"),
            cleanup=clean_handle,
        )
        no_cleanup_resource = r[int].with_resource(
            make_handle,
            lambda handle: r[int].ok(handle.value + 1),
        )
        self.audit_check("with_resource.success", success_resource.unwrap_or(-1))
        self.audit_check("with_resource.failure", failure_resource.error)
        self.audit_check("with_resource.no_cleanup", no_cleanup_resource.unwrap_or(-1))
        self.audit_check("with_resource.cleanup_calls", cleaned_values)

    def side_effects_and_folds(self) -> None:
        """Exercise side-effect helpers, map_or, fold, and filter."""
        self.section("side_effects_and_folds")
        side_effects: MutableSequence[int] = []
        error_effects: MutableSequence[str] = []
        ok_value = r[int].ok(7)
        fail_value = r[int].fail("oops")
        self.audit_check(
            "tap.success",
            ok_value.tap(lambda n: side_effects.append(n)).success,
        )
        self.audit_check(
            "tap.failure",
            fail_value.tap(lambda n: side_effects.append(n)).failure,
        )
        self.audit_check("tap.log", side_effects)
        self.audit_check(
            "tap_error.failure",
            fail_value.tap_error(lambda e: error_effects.append(e)).failure,
        )
        self.audit_check(
            "tap_error.success",
            ok_value.tap_error(lambda e: error_effects.append(e)).success,
        )
        self.audit_check("tap_error.log", error_effects)
        self.audit_check("map_or.success_default", ok_value.map_or(0))
        self.audit_check("map_or.failure_default", fail_value.map_or(0))
        self.audit_check(
            "map_or.success_func", ok_value.map_or("none", lambda n: f"n={n}")
        )
        self.audit_check(
            "map_or.failure_func", fail_value.map_or("none", lambda n: f"n={n}")
        )
        self.audit_check(
            "fold.success",
            ok_value.fold(
                on_failure=lambda e: f"fail:{e}",
                on_success=lambda n: f"ok:{n}",
            ),
        )
        self.audit_check(
            "fold.failure",
            fail_value.fold(
                on_failure=lambda e: f"fail:{e}",
                on_success=lambda n: f"ok:{n}",
            ),
        )
        self.audit_check(
            "filter.success_pass", ok_value.filter(lambda n: n > 0).success
        )
        self.audit_check(
            "filter.success_fail", ok_value.filter(lambda n: n < 0).failure
        )
        self.audit_check(
            "filter.failure_stays",
            fail_value.filter(lambda n: n > 0).failure,
        )

    def transform_chain_and_recover(self) -> None:
        """Exercise transformation and chaining APIs for success/failure paths."""
        self.section("transform_chain_and_recover")
        base_ok = r[int].ok(5)
        base_fail = r[int].fail("bad-number")
        self.audit_check("map.success", base_ok.map(lambda n: n + 1).unwrap_or(-1))
        self.audit_check("map.failure", base_fail.map(lambda n: n + 1).failure)
        self.audit_check(
            "map.exception_to_failure",
            base_ok.map(
                lambda _: (_ for _ in ()).throw(ValueError("map exploded")),
            ).error,
        )
        self.audit_check(
            "flat_map.success",
            base_ok.flat_map(lambda n: r[int].ok(n * 2)).unwrap_or(-1),
        )
        self.audit_check(
            "flat_map.failure",
            base_ok.flat_map(lambda _: r[int].fail("flat failed")).error,
        )
        self.audit_check(
            "flat_map_chain.success",
            base_ok.flat_map(lambda n: r[int].ok(n - 2)).unwrap_or(-1),
        )
        self.audit_check(
            "flat_map_chain.failure",
            base_fail.flat_map(lambda n: r[int].ok(n)).failure,
        )
        bind_ok = self.bind_probe(base_ok, 3)
        bind_fail = self.bind_probe(base_fail, 3)
        self.audit_check("bind.success", self.bind_status(bind_ok))
        self.audit_check("bind.failure", self.bind_status(bind_fail))
        self.audit_check(
            "map_error.success_unchanged",
            base_ok.map_error(lambda e: f"alt:{e}").unwrap_or(-1),
        )
        self.audit_check(
            "map_error.failure_changed",
            base_fail.map_error(lambda e: f"alt:{e}").error,
        )
        self.audit_check(
            "map_error.failure_changed",
            base_fail.map_error(lambda e: f"mapped:{e}").error,
        )
        self.audit_check(
            "map_error.success_unchanged",
            base_ok.map_error(lambda e: f"mapped:{e}").unwrap_or(-1),
        )
        self.audit_check(
            "lash.failure_recovered",
            base_fail.lash(lambda e: r[int].ok(len(e))).unwrap_or(-1),
        )
        self.audit_check(
            "lash.success_unchanged",
            base_ok.lash(lambda _: r[int].ok(99)).unwrap_or(-1),
        )
        self.audit_check(
            "recover.failure", base_fail.recover(lambda e: len(e)).unwrap_or(-1)
        )
        self.audit_check("recover.success", base_ok.recover(lambda _: 0).unwrap_or(-1))
