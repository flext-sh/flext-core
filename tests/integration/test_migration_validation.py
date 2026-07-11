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
from flext_tests import r

from flext_core import FlextContainer, FlextService
from tests.models import m
from tests.protocols import p
from tests.typings import t
from tests.utilities import u

from .migration_validation_cases import capture_stdout


class TestsFlextCoreMigrationValidation:
    """Contract tests for the guaranteed-stable flext-core public surface."""

    # ------------------------------------------------------------------ r[T]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("user_123", "user_123"),
            ("", ""),
            ("A B C", "A B C"),
        ],
    )
    def test_ok_result_exposes_wrapped_value(
        self,
        value: str,
        expected: str,
    ) -> None:
        """A successful result reports success and returns the wrapped value."""
        result: p.Result[str] = r[str].ok(value)

        assert result.success is True
        assert result.failure is False
        assert result.error is None
        assert result.value == expected
        assert result.unwrap() == expected

    def test_fail_result_carries_error_message(self) -> None:
        """A failed result reports failure and preserves the error message."""
        result: p.Result[str] = r[str].fail("Invalid email format")

        assert result.failure is True
        assert result.success is False
        assert result.error is not None
        assert "Invalid email format" in result.error

    def test_unwrap_raises_on_failure(self) -> None:
        """Unwrapping a failure raises instead of inventing a value."""
        result: p.Result[int] = r[int].fail("boom")

        with pytest.raises(RuntimeError):
            result.unwrap()

    @pytest.mark.parametrize(
        ("result", "default", "expected"),
        [
            (r[int].ok(42), 0, 42),
            (r[int].fail("missing"), 7, 7),
        ],
    )
    def test_unwrap_or_returns_default_only_on_failure(
        self,
        result: p.Result[int],
        default: int,
        expected: int,
    ) -> None:
        """unwrap_or yields the value on success and the default on failure."""
        assert result.unwrap_or(default) == expected

    def test_map_transforms_success_and_skips_failure(self) -> None:
        """Map applies to a success value but leaves a failure untouched."""
        mapped_ok = r[str].ok("test_value").map(str.upper)
        assert mapped_ok.success is True
        assert mapped_ok.value == "TEST_VALUE"

        mapped_fail = r[str].fail("orig").map(str.upper)
        assert mapped_fail.failure is True
        assert mapped_fail.error == "orig"

    def test_flat_map_chains_fallible_operations(self) -> None:
        """flat_map sequences dependent fallible steps and short-circuits."""

        def parse(raw: str) -> p.Result[int]:
            if not raw.isdigit():
                return r[int].fail("not a number")
            return r[int].ok(int(raw))

        chained_ok = r[str].ok("21").flat_map(parse).map(lambda n: n * 2)
        assert chained_ok.success is True
        assert chained_ok.value == 42

        chained_fail = r[str].ok("abc").flat_map(parse)
        assert chained_fail.failure is True
        assert chained_fail.error is not None
        assert "not a number" in chained_fail.error

    def test_map_error_transforms_only_the_failure_channel(self) -> None:
        """map_error rewrites a failure's error and leaves success alone."""
        rewritten = r[str].fail("bad").map_error(str.upper)
        assert rewritten.failure is True
        assert rewritten.error == "BAD"

        untouched = r[str].ok("keep").map_error(str.upper)
        assert untouched.success is True
        assert untouched.value == "keep"

    def test_recover_replaces_failure_with_fallback_value(self) -> None:
        """Recover converts a failure into a success using the error."""
        recovered = r[str].fail("e").recover(lambda _err: "fallback")
        assert recovered.success is True
        assert recovered.value == "fallback"

        preserved = r[str].ok("orig").recover(lambda _err: "fallback")
        assert preserved.value == "orig"

    def test_filter_demotes_success_that_fails_predicate(self) -> None:
        """Filter keeps a passing value and rejects a failing one."""
        assert r[int].ok(4).filter(lambda n: n > 0).success is True
        assert r[int].ok(-1).filter(lambda n: n > 0).failure is True

    def test_tap_and_tap_error_observe_without_changing_outcome(self) -> None:
        """tap/tap_error run side effects on the matching channel only."""
        seen: list[int] = []
        errors: list[str] = []

        ok_after = r[int].ok(5).tap(seen.append).tap_error(errors.append)
        assert seen == [5]
        assert errors == []
        assert ok_after.value == 5

        fail_after = r[int].fail("z").tap(seen.append).tap_error(errors.append)
        assert seen == [5]
        assert errors == ["z"]
        assert fail_after.failure is True

    # ------------------------------------------------------------ container

    def test_container_is_process_singleton(self) -> None:
        """FlextContainer() returns the same shared instance every call."""
        assert FlextContainer() is FlextContainer()

    def test_container_binds_and_resolves_registered_service(self) -> None:
        """A bound service resolves to the same object via its public API."""
        container = FlextContainer()

        class RegisteredService(m.BaseModel):
            name: str = "test"

        container.bind("migration_probe_service", RegisteredService())
        resolution = container.resolve(
            "migration_probe_service",
            type_cls=RegisteredService,
        )

        assert resolution.success is True
        assert resolution.value.name == "test"

    def test_container_resolve_missing_key_fails(self) -> None:
        """Resolving an unregistered key yields a failure, not an exception."""
        resolution = FlextContainer().resolve(
            "migration_absent_key",
            type_cls=int,
        )

        assert resolution.failure is True
        assert resolution.error is not None
        assert "migration_absent_key" in resolution.error

    # -------------------------------------------------------------- service

    def test_service_execute_returns_success(self) -> None:
        """A concrete FlextService.execute honors the r[None] contract."""

        class NoopService(FlextService[None]):
            @override
            def execute(self, **_kwargs: t.Scalar) -> p.Result[None]:
                return r[None].ok(None)

        outcome = NoopService().execute()
        assert outcome.success is True
        assert outcome.error is None

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
        assert failure.failure is True
        assert failure.error is not None
        assert "required" in failure.error

        success = service.create_user("alice", "alice@example.com")
        assert success.success is True
        assert success.value["username"] == "alice"
        assert success.value["email"] == "alice@example.com"

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
        assert isinstance(u.build_dispatcher(), p.Dispatcher)
        assert isinstance(u.build_registry(), p.Registry)
