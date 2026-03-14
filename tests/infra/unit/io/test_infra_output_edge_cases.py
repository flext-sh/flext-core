"""Tests for flext_infra output edge cases, behavior, and MRO facade methods.

Tests edge cases, boundary conditions, color/unicode behavior, and MRO facade
method accessibility through u.Infra.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io

from flext_infra import u
from flext_infra._utilities.output import OutputBackend
from flext_tests import tm


def _make_backend(
    *,
    use_color: bool = False,
    use_unicode: bool = False,
    stream: io.StringIO | None = None,
) -> OutputBackend:
    """Create a backend with test-friendly settings."""
    buf = stream or io.StringIO()
    return OutputBackend(use_color=use_color, use_unicode=use_unicode, stream=buf)


class TestInfraOutputNoColor:
    """Tests for behavior when color is disabled."""

    def test_no_ansi_codes_when_color_disabled(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.info("test")
        backend.warning("test")
        backend.error("test")
        backend.status("check", "proj", True, 0.1)
        backend.header("Title")
        backend.progress(1, 1, "proj", "test")
        backend.summary("check", 1, 1, 0, 0, 0.1)
        tm.that("\x1b[" not in buf.getvalue(), eq=True)


class TestMroFacadeMethods:
    """Tests for u.Infra MRO facade methods."""

    def test_output_methods_accessible_via_mro(self) -> None:
        tm.that(callable(u.Infra.info), eq=True)
        tm.that(callable(u.Infra.error), eq=True)
        tm.that(callable(u.Infra.warning), eq=True)
        tm.that(callable(u.Infra.status), eq=True)
        tm.that(callable(u.Infra.summary), eq=True)
        tm.that(callable(u.Infra.header), eq=True)
        tm.that(callable(u.Infra.progress), eq=True)
        tm.that(callable(u.Infra.debug), eq=True)
        tm.that(callable(u.Infra.gate_result), eq=True)


class TestInfraOutputEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_status_with_zero_elapsed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=0.0)
        tm.that(buf.getvalue(), contains="0.00s")

    def test_status_with_large_elapsed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=999.99)
        tm.that(buf.getvalue(), contains="999.99s")

    def test_summary_with_all_zeros(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary("test", total=0, success=0, failed=0, skipped=0, elapsed=0.0)
        text = buf.getvalue()
        tm.that(text, contains="Total: 0")
        tm.that(text, contains="Success: 0")

    def test_summary_with_large_numbers(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary(
            "check", total=1000, success=950, failed=40, skipped=10, elapsed=123.45
        )
        text = buf.getvalue()
        tm.that(text, contains="Total: 1000")
        tm.that(text, contains="Success: 950")
        tm.that(text, contains="Failed: 40")

    def test_error_with_multiline_detail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        detail = "line 1\nline 2\nline 3"
        backend.error("multi", detail=detail)
        text = buf.getvalue()
        tm.that(text, contains="ERROR: multi")
        tm.that(text, contains="line 1")
        tm.that(text, contains="line 3")

    def test_header_with_long_title(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        long_title = "A" * 100
        backend.header(long_title)
        tm.that(buf.getvalue(), contains=long_title)

    def test_progress_with_same_current_and_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(5, 5, "proj", "test")
        tm.that(buf.getvalue(), contains="[5/5]")

    def test_multiple_messages_in_sequence(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.info("msg1")
        backend.warning("msg2")
        backend.error("msg3")
        text = buf.getvalue()
        tm.that(text, contains="INFO: msg1")
        tm.that(text, contains="WARN: msg2")
        tm.that(text, contains="ERROR: msg3")

    def test_color_and_unicode_together(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=True, stream=buf)
        backend.status("test", "proj", result=True, elapsed=0.1)
        text = buf.getvalue()
        tm.that("✓" in text or "\x1b[" in text, eq=True)

    def test_gate_result_passed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.gate_result("ruff", count=0, passed=True, elapsed=0.5)
        text = buf.getvalue()
        tm.that(text, contains="[OK]")
        tm.that(text, contains="ruff")

    def test_gate_result_failed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.gate_result("mypy", count=3, passed=False, elapsed=1.2)
        text = buf.getvalue()
        tm.that(text, contains="[FAIL]")
        tm.that(text, contains="3 errors")

    def test_debug_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.debug("trace info")
        tm.that(buf.getvalue(), contains="DEBUG: trace info")
