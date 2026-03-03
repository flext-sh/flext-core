"""FlextResult (r) — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from returns.io import IOFailure, IOSuccess
from returns.maybe import Nothing, Some
from shared import Examples

from flext_core import FlextExceptions, FlextResult, r, t, u


class Ex01FlextResult(Examples):
    """Exercise FlextResult public API."""

    def exercise(self) -> None:
        """Run all sections and verify against the golden file."""
        self.demo_factories_and_guards()
        self.demo_properties_and_unwrap()
        self.demo_transform_chain_and_recover()
        self.demo_side_effects_and_folds()
        self.demo_conversions_and_models()
        self.demo_collections_and_resource()

    def demo_factories_and_guards(self) -> None:
        """Exercise factory constructors, decorator wrapping, and type guards."""
        self.section("factories_and_guards")

        ok_input = self.rand_int()
        ok_result = r[int].ok(ok_input)
        self.check("ok.value", ok_result.value)
        self.check("ok.value_matches_input", ok_result.value == ok_input)

        err_msg = self.rand_str(10)
        err_code = self.rand_str(6)
        err_data = self.rand_dict(2)
        ex_msg = self.rand_str(9)
        failed = r[int].fail(
            err_msg,
            error_code=err_code,
            error_data=err_data,
            exception=ValueError(ex_msg),
        )
        self.check("fail.error", failed.error)
        self.check("fail.error_matches_input", failed.error == err_msg)
        self.check("fail.error_code", failed.error_code)
        self.check("fail.error_code_matches_input", failed.error_code == err_code)
        self.check("fail.error_data", failed.error_data)
        self.check(
            "fail.error_data_matches_input",
            u.is_dict_like(failed.error_data)
            and self.ser(failed.error_data) == self.ser(err_data),
        )

        @r.safe
        def parse_int(value: str) -> int:
            return int(value)

        safe_input = str(self.rand_int(1, 10_000))
        safe_invalid = self.rand_str(7)
        safe_ok = parse_int(safe_input)
        safe_fail = parse_int(safe_invalid)
        self.check("safe.success.unwrap_or", safe_ok.unwrap_or(0))
        self.check(
            "safe.success.matches_input", safe_ok.unwrap_or(0) == int(safe_input)
        )
        self.check("safe.failure.error", safe_fail.error)
        self.check("safe.failure.is_failure", safe_fail.is_failure)

        callable_msg = self.rand_str(12)
        callable_ok = r[str].create_from_callable(lambda: callable_msg)
        callable_fail_msg = self.rand_str(11)
        callable_fail_code = self.rand_str(6)
        callable_fail = r[str].create_from_callable(
            lambda: (_ for _ in ()).throw(RuntimeError(callable_fail_msg)),
            error_code=callable_fail_code,
        )
        callable_none_code = self.rand_str(6)
        callable_none = r[str].create_from_callable(
            lambda: None,
            error_code=callable_none_code,
        )
        self.check("create_from_callable.success", callable_ok.unwrap_or(""))
        self.check(
            "create_from_callable.success_matches_input",
            callable_ok.unwrap_or("") == callable_msg,
        )
        self.check("create_from_callable.failure.code", callable_fail.error_code)
        self.check(
            "create_from_callable.failure.code_matches_input",
            callable_fail.error_code == callable_fail_code,
        )
        self.check("create_from_callable.none.error", callable_none.error)
        self.check(
            "create_from_callable.none.code_matches_input",
            callable_none.error_code == callable_none_code,
        )

        is_success_result = getattr(r, "is_success_result")
        is_failure_result = getattr(r, "is_failure_result")
        self.check("is_success_result.ok", is_success_result(ok_result))
        self.check("is_success_result.ok_matches", is_success_result(ok_result) is True)
        self.check("is_success_result.fail", is_success_result(failed))
        self.check("is_success_result.fail_matches", is_success_result(failed) is False)
        self.check("is_failure_result.fail", is_failure_result(failed))
        self.check("is_failure_result.fail_matches", is_failure_result(failed) is True)
        self.check("is_failure_result.string", is_failure_result(self.rand_str(5)))
        self.check(
            "is_failure_result.string_matches",
            is_failure_result(self.rand_str(5)) is False,
        )

    def demo_properties_and_unwrap(self) -> None:
        """Exercise result properties and unwrap behavior for both states."""
        self.section("properties_and_unwrap")

        success_val = self.rand_str(9)
        fail_msg = self.rand_str(10)
        fail_code = self.rand_str(6)
        fail_data = self.rand_dict(2)
        success = r[str].ok(success_val)
        failure = r[str].fail(fail_msg, error_code=fail_code, error_data=fail_data)

        self.check("prop.success.is_success", success.is_success)
        self.check("prop.success.is_failure", success.is_failure)
        self.check("prop.failure.is_success", failure.is_success)
        self.check("prop.failure.is_failure", failure.is_failure)
        self.check("prop.success.value", success.value)
        self.check("prop.success.value_matches", success.value == success_val)
        self.check("prop.success.data", success.data)
        self.check("prop.success.data_matches", success.data == success_val)
        self.check("prop.success.result_self", success.result is success)
        self.check("prop.failure.error", failure.error)
        self.check("prop.failure.error_matches", failure.error == fail_msg)
        self.check("prop.failure.error_code", failure.error_code)
        self.check("prop.failure.error_code_matches", failure.error_code == fail_code)
        self.check("prop.failure.error_data", failure.error_data)
        self.check(
            "prop.failure.error_data_matches",
            u.is_dict_like(failure.error_data)
            and self.ser(failure.error_data) == self.ser(fail_data),
        )

        self.check("unwrap.success", success.unwrap())
        self.check("unwrap.success_matches", success.unwrap() == success_val)
        try:
            failure.unwrap()
            self.check("unwrap.failure.raises", False)
        except RuntimeError as exc:
            self.check("unwrap.failure.raises", True)
            self.check("unwrap.failure.type", type(exc).__name__)

        self.check("unwrap_or.success", success.unwrap_or(""))
        self.check("unwrap_or.success_matches", success.unwrap_or("") == success_val)
        fallback = self.rand_str(8)
        self.check("unwrap_or.failure", failure.unwrap_or(fallback))
        self.check("unwrap_or.failure_matches", failure.unwrap_or(fallback) == fallback)

    def demo_transform_chain_and_recover(self) -> None:
        """Exercise transformation and chaining APIs for success/failure paths."""
        self.section("transform_chain_and_recover")

        base_value = self.rand_int(2, 1000)
        base_ok = r[int].ok(base_value)
        fail_err = self.rand_str(9)
        base_fail = r[int].fail(fail_err)
        plus_delta = self.rand_int(1, 9)

        mapped = base_ok.map(lambda n: n + plus_delta)
        self.check("map.success", mapped.unwrap_or(-1))
        self.check(
            "map.success_matches", mapped.unwrap_or(-1) == base_value + plus_delta
        )
        self.check("map.failure", base_fail.map(lambda n: n + plus_delta).is_failure)
        map_err = self.rand_str(12)
        self.check(
            "map.exception_to_failure",
            base_ok.map(lambda _: (_ for _ in ()).throw(ValueError(map_err))).error,
        )
        self.check(
            "map.exception_to_failure_matches",
            base_ok.map(lambda _: (_ for _ in ()).throw(ValueError(map_err))).error
            == map_err,
        )

        mult = self.rand_int(2, 5)
        self.check(
            "flat_map.success",
            base_ok.flat_map(lambda n: r[int].ok(n * mult)).unwrap_or(-1),
        )
        self.check(
            "flat_map.success_matches",
            base_ok.flat_map(lambda n: r[int].ok(n * mult)).unwrap_or(-1)
            == base_value * mult,
        )
        flat_fail_msg = self.rand_str(10)
        self.check(
            "flat_map.failure",
            base_ok.flat_map(lambda _: r[int].fail(flat_fail_msg)).error,
        )
        self.check(
            "flat_map.failure_matches",
            base_ok.flat_map(lambda _: r[int].fail(flat_fail_msg)).error
            == flat_fail_msg,
        )

        minus_delta = self.rand_int(1, 9)
        self.check(
            "and_then.success",
            base_ok.and_then(lambda n: r[int].ok(n - minus_delta)).unwrap_or(-1),
        )
        self.check(
            "and_then.success_matches",
            base_ok.and_then(lambda n: r[int].ok(n - minus_delta)).unwrap_or(-1)
            == base_value - minus_delta,
        )
        self.check(
            "and_then.failure", base_fail.and_then(lambda n: r[int].ok(n)).is_failure
        )

        bind_delta = self.rand_int(1, 9)
        bind_ok = self.bind_probe(base_ok, bind_delta)
        bind_fail = self.bind_probe(base_fail, bind_delta)
        self.check("bind.success", self.bind_status(bind_ok))
        self.check("bind.failure", self.bind_status(bind_fail))

        self.check(
            "alt.success_unchanged", base_ok.alt(lambda e: f"alt:{e}").unwrap_or(-1)
        )
        self.check(
            "alt.success_unchanged_matches",
            base_ok.alt(lambda e: f"alt:{e}").unwrap_or(-1) == base_value,
        )
        self.check("alt.failure_changed", base_fail.alt(lambda e: f"alt:{e}").error)
        self.check(
            "alt.failure_changed_matches",
            base_fail.alt(lambda e: f"alt:{e}").error == f"alt:{fail_err}",
        )
        self.check(
            "map_error.failure_changed",
            base_fail.map_error(lambda e: f"mapped:{e}").error,
        )
        self.check(
            "map_error.failure_changed_matches",
            base_fail.map_error(lambda e: f"mapped:{e}").error == f"mapped:{fail_err}",
        )
        self.check(
            "map_error.success_unchanged",
            base_ok.map_error(lambda e: f"mapped:{e}").unwrap_or(-1),
        )
        self.check(
            "map_error.success_unchanged_matches",
            base_ok.map_error(lambda e: f"mapped:{e}").unwrap_or(-1) == base_value,
        )

        self.check(
            "lash.failure_recovered",
            base_fail.lash(lambda e: r[int].ok(len(e))).unwrap_or(-1),
        )
        self.check(
            "lash.failure_recovered_matches",
            base_fail.lash(lambda e: r[int].ok(len(e))).unwrap_or(-1) == len(fail_err),
        )
        unchanged = self.rand_int(1, 100)
        self.check(
            "lash.success_unchanged",
            base_ok.lash(lambda _: r[int].ok(unchanged)).unwrap_or(-1),
        )
        self.check(
            "lash.success_unchanged_matches",
            base_ok.lash(lambda _: r[int].ok(unchanged)).unwrap_or(-1) == base_value,
        )
        self.check(
            "recover.failure",
            base_fail.recover(lambda e: len(e)).unwrap_or(-1),
        )
        self.check(
            "recover.failure_matches",
            base_fail.recover(lambda e: len(e)).unwrap_or(-1) == len(fail_err),
        )
        recov = self.rand_int(1, 99)
        self.check("recover.success", base_ok.recover(lambda _: recov).unwrap_or(-1))
        self.check(
            "recover.success_matches",
            base_ok.recover(lambda _: recov).unwrap_or(-1) == base_value,
        )

    def demo_side_effects_and_folds(self) -> None:
        """Exercise side-effect helpers, map_or, fold, and filter."""
        self.section("side_effects_and_folds")

        side_effects: list[int] = []
        error_effects: list[str] = []

        ok_num = self.rand_int(1, 999)
        fail_text = self.rand_str(8)
        ok_value = r[int].ok(ok_num)
        fail_value = r[int].fail(fail_text)

        self.check(
            "tap.success", ok_value.tap(lambda n: side_effects.append(n)).is_success
        )
        self.check(
            "tap.failure", fail_value.tap(lambda n: side_effects.append(n)).is_failure
        )
        self.check("tap.log", side_effects)
        self.check("tap.log_matches", side_effects == [ok_num])

        self.check(
            "tap_error.failure",
            fail_value.tap_error(lambda e: error_effects.append(e)).is_failure,
        )
        self.check(
            "tap_error.success",
            ok_value.tap_error(lambda e: error_effects.append(e)).is_success,
        )
        self.check("tap_error.log", error_effects)
        self.check("tap_error.log_matches", error_effects == [fail_text])

        map_or_default = self.rand_int(1, 999)
        self.check("map_or.success_default", ok_value.map_or(map_or_default))
        self.check(
            "map_or.success_default_matches",
            ok_value.map_or(map_or_default) == ok_num,
        )
        self.check("map_or.failure_default", fail_value.map_or(map_or_default))
        self.check(
            "map_or.failure_default_matches",
            fail_value.map_or(map_or_default) == map_or_default,
        )
        prefix = self.rand_str(4)
        self.check(
            "map_or.success_func", ok_value.map_or("", lambda n: f"{prefix}:{n}")
        )
        self.check(
            "map_or.success_func_matches",
            ok_value.map_or("", lambda n: f"{prefix}:{n}") == f"{prefix}:{ok_num}",
        )
        fallback = self.rand_str(7)
        self.check(
            "map_or.failure_func",
            fail_value.map_or(fallback, lambda n: f"{prefix}:{n}"),
        )
        self.check(
            "map_or.failure_func_matches",
            fail_value.map_or(fallback, lambda n: f"{prefix}:{n}") == fallback,
        )

        ok_tag = self.rand_str(3)
        fail_tag = self.rand_str(3)
        self.check(
            "fold.success",
            ok_value.fold(
                on_failure=lambda e: f"{fail_tag}:{e}",
                on_success=lambda n: f"{ok_tag}:{n}",
            ),
        )
        self.check(
            "fold.success_matches",
            ok_value.fold(
                on_failure=lambda e: f"{fail_tag}:{e}",
                on_success=lambda n: f"{ok_tag}:{n}",
            )
            == f"{ok_tag}:{ok_num}",
        )
        self.check(
            "fold.failure",
            fail_value.fold(
                on_failure=lambda e: f"{fail_tag}:{e}",
                on_success=lambda n: f"{ok_tag}:{n}",
            ),
        )
        self.check(
            "fold.failure_matches",
            fail_value.fold(
                on_failure=lambda e: f"{fail_tag}:{e}",
                on_success=lambda n: f"{ok_tag}:{n}",
            )
            == f"{fail_tag}:{fail_text}",
        )

        self.check("filter.success_pass", ok_value.filter(lambda n: n > 0).is_success)
        self.check("filter.success_fail", ok_value.filter(lambda n: n < 0).is_failure)
        self.check(
            "filter.failure_stays", fail_value.filter(lambda n: n > 0).is_failure
        )

    def demo_conversions_and_models(self) -> None:
        """Exercise conversion APIs and Pydantic model integration."""
        self.section("conversions_and_models")

        ok_input = self.rand_int(1, 999)
        fail_msg = self.rand_str(8)
        ok_value = r[int].ok(ok_input)
        fail_value = r[int].fail(fail_msg)

        self.check("to_maybe.success", ok_value.to_maybe().unwrap())
        self.check("to_maybe.success_matches", ok_value.to_maybe().unwrap() == ok_input)
        maybe_fallback = self.rand_int(1, 999)
        self.check("to_maybe.failure", fail_value.to_maybe().value_or(maybe_fallback))
        self.check(
            "to_maybe.failure_matches",
            fail_value.to_maybe().value_or(maybe_fallback) == maybe_fallback,
        )

        maybe_str = self.rand_str(5)
        empty_err = self.rand_str(7)
        self.check(
            "from_maybe.some",
            r[str].from_maybe(Some(maybe_str), empty_err).unwrap_or(""),
        )
        self.check(
            "from_maybe.some_matches",
            r[str].from_maybe(Some(maybe_str), empty_err).unwrap_or("") == maybe_str,
        )
        self.check("from_maybe.nothing", r[str].from_maybe(Nothing, empty_err).error)
        self.check(
            "from_maybe.nothing_matches",
            r[str].from_maybe(Nothing, empty_err).error == empty_err,
        )

        io_value = ok_value.to_io()
        self.check("to_io.success.type", type(io_value).__name__)
        self.check("to_io.success.type_matches", type(io_value).__name__ == "IOSuccess")
        try:
            fail_value.to_io()
            self.check("to_io.failure.raises", False)
        except FlextExceptions.ValidationError as exc:
            self.check("to_io.failure.raises", True)
            self.check("to_io.failure.type", type(exc).__name__)

        self.check("to_io_result.success.type", type(ok_value.to_io_result()).__name__)
        self.check(
            "to_io_result.success.type_matches",
            type(ok_value.to_io_result()).__name__ == "IOSuccess",
        )
        self.check(
            "to_io_result.failure.type", type(fail_value.to_io_result()).__name__
        )
        self.check(
            "to_io_result.failure.type_matches",
            type(fail_value.to_io_result()).__name__ == "IOFailure",
        )

        io_success_input = self.rand_int(1, 999)
        io_failure_msg = self.rand_str(6)
        from_io_ok = r[int].from_io_result(IOSuccess(io_success_input))
        from_io_fail = r[int].from_io_result(IOFailure(io_failure_msg))
        from_io_bad = r[int].from_io_result(self.rand_str(12))
        self.check("from_io_result.success", from_io_ok.is_success)
        self.check(
            "from_io_result.success_matches",
            from_io_ok.unwrap_or(-1) == io_success_input,
        )
        self.check("from_io_result.failure", from_io_fail.error)
        self.check(
            "from_io_result.failure_matches", from_io_fail.error == io_failure_msg
        )
        self.check("from_io_result.invalid", from_io_bad.error)
        self.check("from_io_result.invalid_is_failure", from_io_bad.is_failure)

        person = self.rand_person()
        valid_data: dict[str, t.ContainerValue] = {
            "name": person.name,
            "age": person.age,
        }
        invalid_data: dict[str, t.ContainerValue] = {
            "name": person.name,
            "age": self.rand_str(4),
        }

        from_validation_ok = r[self.Person].from_validation(valid_data, self.Person)
        from_validation_fail = r[self.Person].from_validation(invalid_data, self.Person)
        self.check("from_validation.success", from_validation_ok.is_success)
        self.check("from_validation.failure", from_validation_fail.is_failure)
        self.check(
            "from_validation.success_matches",
            from_validation_ok.unwrap_or(self.Person(name="", age=0)).age == person.age,
        )

        self.check(
            "to_model.success",
            r[dict[str, t.ContainerValue]]
            .ok(valid_data)
            .to_model(self.Person)
            .value.age,
        )
        self.check(
            "to_model.success_matches",
            r[dict[str, t.ContainerValue]]
            .ok(valid_data)
            .to_model(self.Person)
            .value.age
            == person.age,
        )
        to_model_failure_msg = self.rand_str(10)
        self.check(
            "to_model.from_failure",
            r[dict[str, t.ContainerValue]]
            .fail(to_model_failure_msg)
            .to_model(self.Person)
            .error,
        )
        self.check(
            "to_model.from_failure_matches",
            r[dict[str, t.ContainerValue]]
            .fail(to_model_failure_msg)
            .to_model(self.Person)
            .error
            == to_model_failure_msg,
        )
        self.check(
            "to_model.validation_failure",
            r[dict[str, t.ContainerValue]]
            .ok(invalid_data)
            .to_model(self.Person)
            .is_failure,
        )

    def demo_collections_and_resource(self) -> None:
        """Exercise collection helpers and resource management wrapper."""
        self.section("collections_and_resource")

        def to_even(n: int) -> FlextResult[int]:
            if n % 2 == 0:
                return r[int].ok(n)
            return r[int].fail(f"odd:{n}")

        even_base = self.rand_int(1, 100)
        even_values = [2 * even_base, 2 * (even_base + 1), 2 * (even_base + 2)]
        mixed_values = [2 * even_base, 2 * even_base + 1, 2 * (even_base + 1)]
        odd_values = [2 * even_base + 1, 2 * even_base + 3, 2 * even_base + 5]
        self.check(
            "traverse.success",
            r[list].traverse(even_values, to_even, fail_fast=True).unwrap_or([]),
        )
        self.check(
            "traverse.success_matches",
            r[list].traverse(even_values, to_even, fail_fast=True).unwrap_or([])
            == even_values,
        )
        self.check(
            "traverse.fail_fast",
            r[list].traverse(mixed_values, to_even, fail_fast=True).error,
        )
        self.check(
            "traverse.fail_fast_matches",
            isinstance(
                r[list].traverse(mixed_values, to_even, fail_fast=True).error, str
            ),
        )
        self.check(
            "traverse.collect",
            r[list].traverse(odd_values, to_even, fail_fast=False).error,
        )
        self.check(
            "traverse.collect_is_failure",
            r[list].traverse(odd_values, to_even, fail_fast=False).is_failure,
        )

        acc_ok_a = self.rand_int(1, 50)
        acc_ok_b = self.rand_int(51, 99)
        acc_fail_a = self.rand_str(4)
        acc_fail_b = self.rand_str(4)
        acc_ok = r.accumulate_errors(r[int].ok(acc_ok_a), r[int].ok(acc_ok_b))
        acc_fail = r.accumulate_errors(
            r[int].ok(acc_ok_a),
            r[int].fail(acc_fail_a),
            r[int].fail(acc_fail_b),
        )
        self.check("accumulate_errors.success", acc_ok.unwrap_or([]))
        self.check(
            "accumulate_errors.success_matches",
            acc_ok.unwrap_or([]) == [acc_ok_a, acc_ok_b],
        )
        self.check("accumulate_errors.failure", acc_fail.error)
        self.check("accumulate_errors.failure_is_failure", acc_fail.is_failure)

        cleaned_values: list[int] = []
        handle_value = self.rand_int(1, 200)

        def make_handle() -> Ex01FlextResult.Handle:
            return self.Handle(value=handle_value)

        def clean_handle(handle: Ex01FlextResult.Handle) -> None:
            handle.cleaned = True
            cleaned_values.append(handle.value)

        success_resource = r[int].with_resource(
            make_handle,
            lambda handle: r[int].ok(handle.value * 2),
            cleanup=clean_handle,
        )
        failure_msg = self.rand_str(11)
        failure_resource = r[int].with_resource(
            make_handle,
            lambda _: r[int].fail(failure_msg),
            cleanup=clean_handle,
        )
        no_cleanup_resource = r[int].with_resource(
            make_handle,
            lambda handle: r[int].ok(handle.value + 1),
        )

        self.check("with_resource.success", success_resource.unwrap_or(-1))
        self.check(
            "with_resource.success_matches",
            success_resource.unwrap_or(-1) == handle_value * 2,
        )
        self.check("with_resource.failure", failure_resource.error)
        self.check(
            "with_resource.failure_matches",
            failure_resource.error == failure_msg,
        )
        self.check("with_resource.no_cleanup", no_cleanup_resource.unwrap_or(-1))
        self.check(
            "with_resource.no_cleanup_matches",
            no_cleanup_resource.unwrap_or(-1) == handle_value + 1,
        )
        self.check("with_resource.cleanup_calls", cleaned_values)
        self.check(
            "with_resource.cleanup_calls_matches",
            cleaned_values == [handle_value, handle_value],
        )


if __name__ == "__main__":
    Ex01FlextResult(__file__).run()
