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
from flext_tests import r, tm
from tests.models import m
from tests.protocols import p
from tests.typings import t
from tests.utilities import u

from .migration_validation_cases import capture_stdout

_EXPECTED_DOUBLED_VALUE = 42
_OBSERVED_VALUE = 5


class TestsFlextCoreMigrationValidation:
    """Contract tests for the guaranteed-stable flext-core public surface."""

    # ------------------------------------------------------------------ r[T]

    @pytest.mark.parametrize(
        ("value", "expected"), [("user_123", "user_123"), ("", ""), ("A B C", "A B C")]
    )
    def test_ok_result_exposes_wrapped_value(self, value: str, expected: str) -> None:
        """A successful result reports success and returns the wrapped value."""
        result: p.Result[str] = r[str].ok(value)

        tm.that(result.success, eq=True)
        tm.that(result.failure, eq=False)
        tm.that(result.error, none=True)
        tm.that(result.value, eq=expected)
        tm.that(result.unwrap(), eq=expected)

    def test_fail_result_carries_error_message(self) -> None:
        """A failed result reports failure and preserves the error message."""
        result: p.Result[str] = r[str].fail("Invalid email format")

        tm.that(result.failure, eq=True)
        tm.that(result.success, eq=False)
        tm.that(result.error, none=False)
        tm.that(tm.not_none(result.error), has="Invalid email format")

    def test_unwrap_raises_on_failure(self) -> None:
        """Unwrapping a failure raises instead of inventing a value."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError):
            result.unwrap()

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [(r[int].ok(42), 0, 42), (r[int].fail("missing"), 7, 7)],
    )
    def test_unwrap_or_returns_default_only_on_failure(
        self, result: p.Result[int], default: int, expected: int
    ) -> None:
        """unwrap_or yields the value on success and the default on failure."""
        tm.that(result.unwrap_or(default), eq=expected)

    def test_map_transforms_success_and_skips_failure(self) -> None:
        """Map applies to a success value but leaves a failure untouched."""
        mapped_ok = r[str].ok("test_value").map(str.upper)
        tm.that(mapped_ok.success, eq=True)
        tm.that(mapped_ok.value, eq="TEST_VALUE")

        mapped_fail = r[str].fail("orig").map(str.upper)
        tm.that(mapped_fail.failure, eq=True)
        tm.that(mapped_fail.error, eq="orig")

    def test_flat_map_chains_fallible_operations(self) -> None:
        """flat_map sequences dependent fallible steps and short-circuits."""

        def parse(raw: str) -> p.Result[int]:
            if not raw.isdigit():
                return r[int].fail("not a number")
            return r[int].ok(int(raw))

        chained_ok = r[str].ok("21").flat_map(parse).map(lambda n: n * 2)
        tm.that(chained_ok.success, eq=True)
        tm.that(chained_ok.value, eq=_EXPECTED_DOUBLED_VALUE)

        chained_fail = r[str].ok("abc").flat_map(parse)
        tm.that(chained_fail.failure, eq=True)
        tm.that(chained_fail.error, none=False)
        tm.that(tm.not_none(chained_fail.error), has="not a number")

    def test_map_error_transforms_only_the_failure_channel(self) -> None:
        """map_error rewrites a failure's error and leaves success alone."""
        rewritten = r[str].fail("bad").map_error(str.upper)
        tm.that(rewritten.failure, eq=True)
        tm.that(rewritten.error, eq="BAD")

        untouched = r[str].ok("keep").map_error(str.upper)
        tm.that(untouched.success, eq=True)
        tm.that(untouched.value, eq="keep")

    def test_recover_replaces_failure_with_fallback_value(self) -> None:
        """Recover converts a failure into a success using the error."""
        recovered = r[str].fail("e").recover(lambda _err: "fallback")
        tm.that(recovered.success, eq=True)
        tm.that(recovered.value, eq="fallback")

        preserved = r[str].ok("orig").recover(lambda _err: "fallback")
        tm.that(preserved.value, eq="orig")

    def test_filter_demotes_success_that_fails_predicate(self) -> None:
        """Filter keeps a passing value and rejects a failing one."""
        tm.that(r[int].ok(4).filter(lambda n: n > 0).success, eq=True)
        tm.that(r[int].ok(-1).filter(lambda n: n > 0).failure, eq=True)

    def test_tap_and_tap_error_observe_without_changing_outcome(self) -> None:
        """tap/tap_error run side effects on the matching channel only."""
        seen: list[int] = []
        errors: list[str] = []

        ok_after = r[int].ok(_OBSERVED_VALUE).tap(seen.append).tap_error(errors.append)
        tm.that(seen, eq=[_OBSERVED_VALUE])
        tm.that(errors, empty=True)
        tm.that(ok_after.value, eq=_OBSERVED_VALUE)

        fail_after = r[int].fail("z").tap(seen.append).tap_error(errors.append)
        tm.that(seen, eq=[_OBSERVED_VALUE])
        tm.that(errors, eq=["z"])
        tm.that(fail_after.failure, eq=True)

    # ------------------------------------------------------------ container

    def test_container_is_process_singleton(self) -> None:
        """FlextContainer() returns the same shared instance every call."""
        tm.that(FlextContainer() is FlextContainer(), eq=True)

    def test_container_binds_and_resolves_registered_service(self) -> None:
        """A bound service resolves to the same object via its public API."""
        container = FlextContainer()

        class RegisteredService(m.BaseModel):
            name: str = "test"

        container.bind("migration_probe_service", RegisteredService())
        resolution = container.resolve(
            "migration_probe_service", type_cls=RegisteredService
        )

        tm.that(resolution.success, eq=True)
        tm.that(resolution.value.name, eq="test")

    def test_container_resolve_missing_key_fails(self) -> None:
        """Resolving an unregistered key yields a failure, not an exception."""
        resolution = FlextContainer().resolve("migration_absent_key", type_cls=int)

        tm.that(resolution.failure, eq=True)
        tm.that(resolution.error, none=False)
        tm.that(tm.not_none(resolution.error), has="migration_absent_key")

    # -------------------------------------------------------------- service

    def test_service_execute_returns_success(self) -> None:
        """A concrete FlextService.execute honors the r[None] contract."""

        class NoopService(FlextService[None]):
            @override
            def execute(self, **_kwargs: t.Scalar) -> p.Result[None]:
                return r[None].ok(None)

        outcome = NoopService().execute()
        tm.that(outcome.success, eq=True)
        tm.that(outcome.error, none=True)

    def test_service_method_returns_failure_on_invalid_input(self) -> None:
        """Domain validation surfaces as an r failure, not a raised error."""

        class UserService(FlextService[None]):
            @override
            def execute(self, **_kwargs: t.Scalar) -> p.Result[None]:
                return r[None].ok(None)

            def create_user(self, username: str, email: str) -> p.Result[t.StrMapping]:
                if not username or not email:
                    return r[t.StrMapping].fail("Username and email required")
                return r[t.StrMapping].ok({"username": username, "email": email})

        service = UserService()

        failure = service.create_user("", "alice@example.com")
        tm.that(failure.failure, eq=True)
        tm.that(failure.error, none=False)
        tm.that(tm.not_none(failure.error), has="required")

        success = service.create_user("alice", "alice@example.com")
        tm.that(success.success, eq=True)
        tm.that(success.value["username"], eq="alice")
        tm.that(success.value["email"], eq="alice@example.com")

    # --------------------------------------------------------------- logger

    def test_logger_emits_structured_message(self) -> None:
        """The logging DSL produces observable output for the given message."""
        logger = u.fetch_logger(__name__)

        _ = capture_stdout(
            lambda: logger.info("migration probe message"),
            contains="migration probe message",
        )

    # --------------------------------------------------- stable API contract

    def test_factory_helpers_produce_protocol_conformant_objects(self) -> None:
        """Public builders return objects satisfying their published protocols."""
        tm.that(u.build_dispatcher(), is_=p.Dispatcher)
        tm.that(u.build_registry(), is_=p.Registry)
