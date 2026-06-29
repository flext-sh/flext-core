"""r (r) — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import r, t

from .ex_01_flext_result_helpers import Ex01ResultAdvancedSections
from .models import m
from .shared import ExamplesFlextShared


class Ex01r(Ex01ResultAdvancedSections):
    """Golden-file tests for ``r`` / ``r`` public API."""

    def __init__(self) -> None:
        """Initialise harness bound to this script's golden file."""
        super().__init__(caller_file=Path(__file__))

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


def main() -> None:
    """Entry point."""
    Ex01r().run()


if __name__ == "__main__":
    main()
