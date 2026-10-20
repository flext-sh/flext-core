"""Behavioral integration tests for the flext-core public API surface.

Every test asserts observable public behavior: the ``r[T]`` outcome of fallible
operations, public model state, container contracts, and the logging DSL. No
private attribute, internal helper, or implementation detail is inspected.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import override

import pytest

from flext_core import FlextContainer, FlextService
from tests import m, p, r, t, tm, u
from tests.integration.migration_validation_cases import capture_output

_EXPECTED_DOUBLED_VALUE = 42
_OBSERVED_VALUE = 5


class TestsFlextCoreMigrationValidation:
    """Contract tests for the guaranteed-stable flext-core public surface."""

    # Section: Result contract

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("user_123", "user_123"), ("", ""), ("A B C", "A B C")],
    )
    @staticmethod
    def test_ok_result_exposes_wrapped_value(value: str, expected: str) -> None:
        """A successful result reports success and returns the wrapped value."""
        result: p.Result[str] = r[str].ok(value)

        tm.ok(result.success)
        tm.fail(result.failure)
        tm.that(result.error, none=True)
        tm.that(result.value, eq=expected)
        tm.that(result.unwrap(), eq=expected)

    @staticmethod
    def test_fail_result_carries_error_message() -> None:
        """A failed result reports failure and preserves the error message."""
        result: p.Result[str] = r[str].fail("Invalid email format")

        tm.ok(result.failure)
        tm.fail(result.success)
        tm.that(result.error, none=False)
        tm.that(tm.not_none(result.error), has="Invalid email format")

    @staticmethod
    def test_unwrap_raises_on_failure() -> None:
        """Unwrapping a failure raises instead of inventing a value."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError):
            result.unwrap()

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [(r[int].ok(42), 0, 42), (r[int].fail("missing"), 7, 7)],
    )
    @staticmethod
    def test_unwrap_or_returns_default_only_on_failure(
        result: p.Result[int],
        default: int,
        expected: int,
    ) -> None:
        """unwrap_or yields the value on success and the default on failure."""
        tm.that(result.unwrap_or(default), eq=expected)

    @staticmethod
    def test_map_transforms_success_and_skips_failure() -> None:
        """Map applies to a success value but leaves a failure untouched."""
        mapped_ok = r[str].ok("test_value").map(str.upper)
        tm.ok(mapped_ok.success)
        tm.that(mapped_ok.value, eq="TEST_VALUE")

        mapped_fail = r[str].fail("orig").map(str.upper)
        tm.ok(mapped_fail.failure)
        tm.that(mapped_fail.error, eq="orig")

    @staticmethod
    def test_flat_map_chains_fallible_operations() -> None:
        """flat_map sequences dependent fallible steps and short-circuits."""

        def parse(raw: str) -> p.Result[int]:
            if not raw.isdigit():
                return r[int].fail("not a number")
            return r[int].ok(int(raw))

        chained_ok = r[str].ok("21").flat_map(parse).map(lambda n: n * 2)
        tm.ok(chained_ok.success)
        tm.that(chained_ok.value, eq=_EXPECTED_DOUBLED_VALUE)

        chained_fail = r[str].ok("abc").flat_map(parse)
        tm.ok(chained_fail.failure)
        tm.that(chained_fail.error, none=False)
        tm.that(tm.not_none(chained_fail.error), has="not a number")

    @staticmethod
    def test_map_error_transforms_only_the_failure_channel() -> None:
        """map_error rewrites a failure's error and leaves success alone."""
        rewritten: p.Result[str] = r[str].fail("bad").map_error(str.upper)
        tm.ok(rewritten.failure)
        tm.that(rewritten.error, eq="BAD")

        untouched: p.Result[str] = r[str].ok("keep").map_error(str.upper)
        tm.ok(untouched.success)
        tm.that(untouched.value, eq="keep")

    @staticmethod
    def test_recover_replaces_failure_with_fallback_value() -> None:
        """Recover converts a failure into a success using the error."""
        recovered = r[str].fail("e").recover(lambda _err: "fallback")
        tm.ok(recovered.success)
        tm.that(recovered.value, eq="fallback")

        preserved = r[str].ok("orig").recover(lambda _err: "fallback")
        tm.that(preserved.value, eq="orig")

    @staticmethod
    def test_filter_demotes_success_that_fails_predicate() -> None:
        """Filter keeps a passing value and rejects a failing one."""
        tm.ok(r[int].ok(4).filter(lambda n: n > 0).success)
        tm.ok(r[int].ok(-1).filter(lambda n: n > 0).failure)

    @staticmethod
    def test_tap_and_tap_error_observe_without_changing_outcome() -> None:
        """tap/tap_error run side effects on the matching channel only."""
        seen: list[int] = []
        errors: list[str] = []

        ok_after = r[int].ok(_OBSERVED_VALUE).tap(seen.append).tap_error(errors.append)
        tm.that(seen, eq=[_OBSERVED_VALUE])
        tm.that(errors, empty=True)
        tm.that(ok_after.value, eq=_OBSERVED_VALUE)

        fail_after: p.Result[int] = (
            r[int].fail("z").tap(seen.append).tap_error(errors.append)
        )
        tm.that(seen, eq=[_OBSERVED_VALUE])
        tm.that(errors, eq=["z"])
        tm.ok(fail_after.failure)

    # ------------------------------------------------------------ container

    @staticmethod
    def test_container_is_process_singleton() -> None:
        """FlextContainer() returns the same shared instance every call."""
        tm.ok(FlextContainer() is FlextContainer())

    @staticmethod
    def test_container_binds_and_resolves_registered_service() -> None:
        """A bound service resolves to the same object via its public API."""
        container = FlextContainer()

        class RegisteredService(m.BaseModel):
            name: str = "test"

        container.bind("migration_probe_service", RegisteredService())
        resolution = container.resolve(
            "migration_probe_service",
            type_cls=RegisteredService,
        )

        tm.ok(resolution.success)
        tm.that(resolution.value.name, eq="test")

    @staticmethod
    def test_container_resolve_missing_key_fails() -> None:
        """Resolving an unregistered key yields a failure, not an exception."""
        resolution = FlextContainer().resolve("migration_absent_key", type_cls=int)

        tm.ok(resolution.failure)
        tm.that(resolution.error, none=False)
        tm.that(tm.not_none(resolution.error), has="migration_absent_key")

    # -------------------------------------------------------------- service

    @staticmethod
    def test_service_execute_returns_success() -> None:
        """A concrete FlextService.execute honors the r[bool] void contract."""

        class NoopService(FlextService[bool]):
            @override
            def execute(self, **_kwargs: t.Scalar) -> p.Result[bool]:
                return r[bool].ok(value=True)

        outcome = NoopService().execute()
        tm.ok(outcome.success)
        tm.that(outcome.error, none=True)

    @staticmethod
    def test_service_method_returns_failure_on_invalid_input() -> None:
        """Domain validation surfaces as an r failure, not a raised error."""

        class UserService(FlextService[bool]):
            @override
            def execute(self, **_kwargs: t.Scalar) -> p.Result[bool]:
                return r[bool].ok(value=True)

            @staticmethod
            def create_user(username: str, email: str) -> p.Result[t.StrMapping]:
                if not username or not email:
                    return r[t.StrMapping].fail("Username and email required")
                return r[t.StrMapping].ok({"username": username, "email": email})

        service = UserService()

        failure = service.create_user("", "alice@example.com")
        tm.ok(failure.failure)
        tm.that(failure.error, none=False)
        tm.that(tm.not_none(failure.error), has="required")

        success = service.create_user("alice", "alice@example.com")
        tm.ok(success.success)
        tm.that(success.value["username"], eq="alice")
        tm.that(success.value["email"], eq="alice@example.com")

    # --------------------------------------------------------------- logger

    @staticmethod
    def test_logger_emits_structured_message(capfd: p.Tests.OutputCapture) -> None:
        """The logging DSL produces observable output for the given message."""
        logger = u.fetch_logger(__name__)

        _ = capture_output(
            lambda: logger.info("migration probe message"),
            capfd,
            contains="migration probe message",
        )

    # --------------------------------------------------- stable API contract

    @staticmethod
    def test_factory_helpers_produce_protocol_conformant_objects() -> None:
        """Public builders return objects satisfying their published protocols."""
        tm.that(u.build_dispatcher(), is_=p.Dispatcher)
        tm.that(u.build_registry(), is_=p.Registry)
