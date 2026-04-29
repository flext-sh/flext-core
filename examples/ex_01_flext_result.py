"""r (r) — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
    Sequence,
)
from typing import override

from flext_core import p, r, t

from .models import m
from .shared import ExamplesFlextShared


class Ex01r(ExamplesFlextShared):
    """Golden-file tests for ``r`` / ``r`` public API."""

    def __init__(self) -> None:
        """Initialise harness bound to this script's golden file."""
        super().__init__(__file__)

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

    def conversions_and_models(self) -> None:
        """Exercise conversion APIs and Pydantic model integration."""
        self.section("conversions_and_models")
        ok_value = r[int].ok(8)
        fail_value = r[int].fail("io-err")
        self.audit_check("value.success", ok_value.value)
        self.audit_check("value.failure.unwrap_or", fail_value.unwrap_or(123))
        self.audit_check(
            "map.success", ok_value.map(lambda value: value + 1).unwrap_or(-1)
        )
        self.audit_check("map.failure", fail_value.map(lambda value: value + 1).failure)
        self.audit_check(
            "flat_map.success",
            ok_value.flat_map(lambda value: r[int].ok(value * 2)).unwrap_or(-1),
        )
        self.audit_check(
            "map_error.failure",
            fail_value.map_error(lambda err: f"mapped:{err}").error,
        )
        self.audit_check(
            "lash.failure",
            fail_value.lash(lambda err: r[int].ok(len(err))).unwrap_or(-1),
        )
        valid_data: t.JsonMapping = {"name": "Ada", "age": 30}
        invalid_data: t.JsonMapping = {"name": "Ada", "age": "bad"}
        person_model = ExamplesFlextShared.Person
        from_validation_ok = r[ExamplesFlextShared.Person].from_validation(
            valid_data,
            person_model,
        )
        from_validation_fail = r[ExamplesFlextShared.Person].from_validation(
            invalid_data,
            person_model,
        )
        self.audit_check("from_validation.success", from_validation_ok.success)
        self.audit_check("from_validation.failure", from_validation_fail.failure)
        self.audit_check(
            "to_model.success",
            r[t.JsonMapping].ok(valid_data).to_model(self.Person).value.age,
        )
        self.audit_check(
            "to_model.from_failure",
            r[t.JsonMapping].fail("missing").to_model(self.Person).error,
        )
        self.audit_check(
            "to_model.validation_failure",
            r[t.JsonMapping].ok(invalid_data).to_model(self.Person).failure,
        )

    def factories_and_guards(self) -> None:
        """Exercise factory constructors, decorator wrapping, and type guards."""
        self.section("factories_and_guards")
        ok_result = r[int].ok(10)
        self.audit_check("ok.value", ok_result.value)
        failed = r[int].fail(
            "boom",
            error_code="E_DEMO",
            error_data={"stage": "factory"},
            exception=ValueError("bad input"),
        )
        self.audit_check("fail.error", failed.error)
        self.audit_check("fail.error_code", failed.error_code)
        self.audit_check(
            "fail.error_data",
            failed.error_data
            if failed.error_data is not None
            else m.ConfigMap(root={}),
        )

        @r.safe
        def parse_int(value: str) -> int:
            return int(value)

        safe_ok = parse_int("42")
        safe_fail = parse_int("x")
        self.audit_check("safe.success.unwrap_or", safe_ok.unwrap_or(0))
        self.audit_check("safe.failure.error", safe_fail.error)

        def func_fail() -> str | None:
            msg = m.Examples.ErrorMessages.CALLABLE_FAILED
            raise RuntimeError(msg)

        def func_none() -> str | None:
            return None

        callable_ok = r[str].create_from_callable(lambda: "created")
        callable_fail = r[str].create_from_callable(func_fail, error_code="E_CALL")
        callable_none = r[str].create_from_callable(func_none, error_code="E_NONE")
        self.audit_check(
            "create_from_callable.success", callable_ok.unwrap_or("fallback")
        )
        self.audit_check("create_from_callable.failure.code", callable_fail.error_code)
        self.audit_check("create_from_callable.none.error", callable_none.error)
        self.audit_check("is_success_result.scalar", r.successful_result("ok"))
        self.audit_check("is_success_result.int", r.successful_result(1))
        self.audit_check("is_failure_result.scalar", r.failed_result("plain"))
        self.audit_check("is_failure_result.string", r.failed_result("plain"))

    def properties_and_unwrap(self) -> None:
        """Exercise result properties and unwrap behavior for both states."""
        self.section("properties_and_unwrap")
        success = r[str].ok("value")
        failure = r[str].fail("missing", error_code="E_PROP", error_data={"x": 1})
        self.audit_check("prop.success.is_success", success.success)
        self.audit_check("prop.success.is_failure", success.failure)
        self.audit_check("prop.failure.is_success", failure.success)
        self.audit_check("prop.failure.is_failure", failure.failure)
        self.audit_check("prop.success.value", success.value)
        self.audit_check("prop.success.value_readable", success.value == "value")
        self.audit_check("prop.failure.error", failure.error)
        self.audit_check("prop.failure.error_code", failure.error_code)
        self.audit_check(
            "prop.failure.error_data",
            failure.error_data
            if failure.error_data is not None
            else m.ConfigMap(root={}),
        )
        self.audit_check("unwrap.success", success.unwrap())
        try:
            _ = failure.unwrap()
            self.audit_check("unwrap.failure.raises", False)
        except RuntimeError as exc:
            self.audit_check("unwrap.failure.raises", True)
            self.audit_check("unwrap.failure.type", type(exc).__name__)
        self.audit_check("unwrap_or.success", success.unwrap_or("default"))
        self.audit_check("unwrap_or.failure", failure.unwrap_or("default"))

    @override
    def run(self) -> None:
        """Run all sections and verify against the golden file."""
        self.factories_and_guards()
        self.properties_and_unwrap()
        self.transform_chain_and_recover()
        self.side_effects_and_folds()
        self.conversions_and_models()
        self.collections_and_resource()
        self.verify()

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


def main() -> None:
    """Entry point."""
    Ex01r().run()


if __name__ == "__main__":
    main()
