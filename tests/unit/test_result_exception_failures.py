"""Behavioral tests for r[T] failure results that carry exception context.

Every assertion targets the public FlextResult contract a caller depends on:
failure state, error/error_code/error_data payload, the carried exception, and
how all of that propagates through the result combinators. No private
attributes, no internal patching, no mock spying on the unit under test.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Mapping
from pathlib import Path
from traceback import format_exception, walk_tb
from typing import Annotated

import pytest

from tests import c, m, p, r, t, tm


class TestsFlextCoreResultExceptionFailures:
    """Tests for ``FlextCoreResultExceptionFailures``."""

    @staticmethod
    @pytest.mark.parametrize(
        ("error_msg", "expected_error"),
        [
            ("Operation failed", "Operation failed"),
            ("Division by zero", "Division by zero"),
            (None, ""),
        ],
    )
    def test_fail_without_exception_exposes_error_and_no_exception(
        error_msg: str | None,
        expected_error: str,
    ) -> None:
        """Test fail without exception exposes error and no exception."""
        result: p.Result[int] = r[int].fail(error_msg)

        tm.ok(result.failure)
        tm.fail(result.success)
        tm.that(result.error, eq=expected_error)
        tm.that(result.exception, none=True)

    @staticmethod
    def test_fail_preserves_carried_exception_identity_and_type() -> None:
        """Test fail preserves carried exception identity and type."""
        exc = ZeroDivisionError("cannot divide by zero")

        result: p.Result[float] = r[float].fail("Division by zero", exception=exc)

        tm.ok(result.failure)
        tm.that(result.error, eq="Division by zero")
        tm.ok(result.exception is exc)
        tm.that(result.exception, is_=ZeroDivisionError)

    @staticmethod
    def test_fail_with_none_error_and_exception_normalizes_error_to_empty() -> None:
        """Test fail with none error and exception normalizes error to empty."""
        exc = RuntimeError("something went wrong")

        result: p.Result[int] = r[int].fail(None, exception=exc)

        tm.ok(result.failure)
        tm.that(result.error, eq="")
        tm.ok(result.exception is exc)

    @staticmethod
    def test_fail_exposes_error_data_mapping_alongside_exception() -> None:
        """Test fail exposes error data mapping alongside exception."""
        error_data: Mapping[str, str] = {"field": "email", "reason": "invalid format"}
        exc = ValueError("invalid email")

        result: p.Result[Mapping[str, str]] = r[Mapping[str, str]].fail(
            "Validation failed",
            error_data=error_data,
            exception=exc,
        )

        tm.ok(result.failure)
        tm.that(result.error, eq="Validation failed")
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("reason"), eq="invalid format")
        tm.ok(result.exception is exc)

    @staticmethod
    def test_fail_preserves_correlation_id_without_metadata_attributes() -> None:
        """Test fail preserves correlation id without metadata attributes."""

        class CorrelationOnlyError(ValueError):
            correlation_id: str

            def __init__(self) -> None:
                super().__init__("no metadata attributes")
                self.correlation_id = "corr-only-456"

        result: p.Result[str] = r[str].fail(None, exception=CorrelationOnlyError())

        tm.ok(result.failure)
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(
                result.error_data.get(c.ContextKey.CORRELATION_ID),
                eq="corr-only-456",
            )

    @staticmethod
    def test_fail_serializes_publication_issues_and_preserves_exception() -> None:
        """Plain nested issue payloads retain data and the original failure."""
        original = ValueError("source changed during publication")
        issues: list[t.JsonValue] = [
            {
                "file": "src/package/module.py",
                "line": 1,
                "column": 0,
                "code": "source-drift",
                "message": str(original),
                "locations": [],
                "related_locations": [],
            }
        ]
        result: p.Result[tuple[Path, ...]] = r[tuple[Path, ...]].fail(
            str(original),
            exception=original,
            error_data={
                "phase": "lazy-init",
                "completed": True,
                "issues": issues,
                "written_files": ["src/independent/__init__.py"],
                "pending_files": ["src/package/__init__.py"],
            },
        )
        tm.fail(result)
        tm.ok(result.exception is original)
        expected: t.JsonDict = {
            "phase": "lazy-init",
            "completed": True,
            "issues": [
                {
                    "file": "src/package/module.py",
                    "line": 1,
                    "column": 0,
                    "code": "source-drift",
                    "message": str(original),
                    "locations": [],
                    "related_locations": [],
                }
            ],
            "written_files": ["src/independent/__init__.py"],
            "pending_files": ["src/package/__init__.py"],
        }
        tm.that(result.error_data, eq=expected)
        with pytest.raises(
            ValueError, match="source changed during publication"
        ) as raised:
            result.unwrap()
        tm.ok(raised.value is original)

    @staticmethod
    def test_fail_enriches_error_data_from_exception_metadata() -> None:
        """Test fail enriches error data from exception metadata."""

        class MetadataError(ValueError):
            metadata: m.Metadata
            correlation_id: str

            def __init__(self) -> None:
                super().__init__("invalid email")
                self.metadata = m.Metadata(
                    attributes={"field": "email", "details": {"retryable": False}},
                )
                self.correlation_id = "corr-123"

        result: p.Result[str] = r[str].fail(
            "Validation failed",
            exception=MetadataError(),
        )

        tm.ok(result.failure)
        tm.that(result.error_data, none=False)
        if result.error_data is not None:
            tm.that(result.error_data.get("field"), eq="email")
            tm.that(result.error_data.get("details"), eq={"retryable": False})
            tm.that(result.error_data.get(c.ContextKey.CORRELATION_ID), eq="corr-123")

    @staticmethod
    def test_map_on_failure_short_circuits_and_keeps_exception() -> None:
        """Test map on failure short circuits and keeps exception."""
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        mapped: p.Result[int] = failure.map(lambda value: value + 1)

        tm.ok(mapped.failure)
        tm.that(mapped.error, eq="bad")
        tm.ok(mapped.exception is exc)

    @staticmethod
    def test_flat_map_on_failure_short_circuits_and_keeps_exception() -> None:
        """Test flat map on failure short circuits and keeps exception."""
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        chained: p.Result[int] = failure.flat_map(r[int].ok)

        tm.ok(chained.failure)
        tm.ok(chained.exception is exc)

    @staticmethod
    def test_map_error_transforms_message_and_preserves_exception() -> None:
        """Test map error transforms message and preserves exception."""
        exc = ValueError("boom")
        failure: p.Result[int] = r[int].fail("bad", exception=exc)

        remapped: p.Result[int] = failure.map_error(lambda msg: msg.upper())

        tm.ok(remapped.failure)
        tm.that(remapped.error, eq="BAD")
        tm.ok(remapped.exception is exc)

    @staticmethod
    def test_unwrap_or_returns_default_for_carried_failure() -> None:
        """Test unwrap or returns default for carried failure."""
        failure: p.Result[int] = r[int].fail("bad", exception=ValueError("boom"))

        tm.that(failure.unwrap_or(99), eq=99)

    @staticmethod
    @pytest.mark.parametrize("use_property", [False, True])
    @pytest.mark.parametrize("kind", [ValueError, OSError, KeyboardInterrupt])
    def test_failure_terminals_preserve_native_identity_and_result_context(
        *,
        use_property: bool,
        kind: type[BaseException],
    ) -> None:
        """Extraction retains the actual cause, producer traceback and metadata."""
        cause = LookupError("original cause")

        def fail_producer() -> None:
            msg = "capture failed"
            raise kind(msg) from cause

        with pytest.raises(kind) as caught:
            fail_producer()
        original = caught.value
        producer = tm.not_none(original.__traceback__)
        failure: p.Result[int] = r[int].fail(
            "selected observer failed",
            error_code="CAPTURE_FAILURE",
            error_data={"target": "selected-observer"},
            exception=original,
        )

        with pytest.raises(kind) as raised:
            _ = failure.value if use_property else failure.unwrap()

        tm.ok(raised.value is original)
        tm.ok(raised.value.__cause__ is cause)
        tm.that(
            any(
                frame is producer.tb_frame and line == producer.tb_lineno
                for frame, line in walk_tb(raised.value.__traceback__)
            ),
            eq=True,
        )
        note = raised.value.__notes__[-1]
        details = m.ConfigMap.model_validate_json(note)
        tm.that(details["error"], eq=failure.error)
        tm.that(details["error_code"], eq=failure.error_code)
        tm.that(details["error_data"], eq=failure.error_data)

        with pytest.raises(kind):
            _ = failure.value if use_property else failure.unwrap()
        tm.that(raised.value.__notes__.count(note), eq=1)

    @staticmethod
    @pytest.mark.parametrize("use_property", [False, True])
    def test_logical_failure_terminal_retains_structured_context(
        *,
        use_property: bool,
    ) -> None:
        """A failure without a native cause still exposes its full result context."""
        failure: p.Result[int] = r[int].fail(
            "capture rejected",
            error_code="CAPTURE_REJECTED",
            error_data={"target": "selected-observer"},
        )
        with pytest.raises(RuntimeError) as raised:
            _ = failure.value if use_property else failure.unwrap()
        details = m.ConfigMap.model_validate_json(raised.value.__notes__[-1])
        tm.that(details["error"], eq=failure.error)
        tm.that(details["error_code"], eq=failure.error_code)
        tm.that(details["error_data"], eq=failure.error_data)

    @staticmethod
    @pytest.mark.parametrize("use_property", [False, True])
    @pytest.mark.parametrize("explicit_data", [False, True])
    def test_failure_metadata_is_private_before_native_traceback_rendering(
        *,
        use_property: bool,
        explicit_data: bool,
    ) -> None:
        """Nested and JSON-text metadata use one privacy boundary before encoding."""
        key = next(iter(c.SENSITIVE_ERROR_DATA_KEYS))
        encoded = m.ConfigMap(
            root={
                key.swapcase(): "encoded-secret",
                "PrivateTenant": "encoded-private",
                "visible": "retained",
            }
        ).model_dump_json()
        payload: t.JsonDict = {
            c.ContextScope.REQUEST: [
                {
                    key.swapcase(): "nested-secret",
                    "PrivateTenant": "nested-private",
                    "visible": "retained",
                }
            ],
            "encoded": encoded,
        }

        class PrivateError(ValueError):
            excluded_context_keys = frozenset({"privatetenant"})

            def __init__(self) -> None:
                super().__init__("capture failed")
                self.metadata = m.Metadata(attributes=payload)

        original = PrivateError()
        failure: p.Result[int] = r[int].fail(
            "selected observer failed",
            exception=original,
            error_data=payload if explicit_data else None,
        )
        with pytest.raises(PrivateError) as raised:
            _ = failure.value if use_property else failure.unwrap()
        rendered = "".join(format_exception(raised.value))
        tm.ok(raised.value is original)
        tm.that(rendered, lacks="nested-secret")
        tm.that(rendered, lacks="encoded-secret")
        tm.that(rendered, lacks="nested-private")
        tm.that(rendered, lacks="encoded-private")
        tm.that(rendered, has="retained")

    @staticmethod
    def test_recover_produces_success_from_carried_failure() -> None:
        """Test recover produces success from carried failure."""
        failure: p.Result[int] = r[int].fail("bad", exception=ValueError("boom"))

        recovered: p.Result[int] = failure.recover(lambda _error: 7)

        tm.ok(recovered.success)
        tm.that(recovered.unwrap(), eq=7)

    @staticmethod
    def test_typed_value_model_survives_recover_from_failure() -> None:
        """Test typed value model survives recover from failure."""

        class UserModel(m.Value):
            name: Annotated[str, m.Field(description="User name")]
            age: Annotated[int, m.Field(description="User age")]

        fallback = UserModel(name="anon", age=0)
        failure: p.Result[UserModel] = r[UserModel].fail(
            "lookup failed",
            exception=KeyError("missing"),
        )

        recovered: p.Result[UserModel] = failure.recover(lambda _error: fallback)

        tm.ok(recovered.success)
        tm.that(recovered.unwrap().name, eq="anon")
        tm.that(recovered.unwrap().age, eq=0)

    @staticmethod
    def test_empty_fail_map_stays_failed_without_raising() -> None:
        """Test empty fail map stays failed without raising."""
        failure: p.Result[int] = r[int].fail(None, exception=ValueError("boom"))
        mapped: p.Result[int] = failure.map(lambda value: value + 1)
        tm.ok(mapped.failure)
        tm.that(mapped.error, eq="")
        tm.ok(mapped.exception is failure.exception)
