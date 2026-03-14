"""Tests for flext_infra output formatting — status, summary, messages, headers, progress.

Tests OutputBackend formatting methods for status, summary, error/warning/info messages,
headers, and progress indicators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import re

from flext_infra._utilities.output import OutputBackend
from flext_tests import tm

ANSI_RE = re.compile(r"\033\[\d+m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_backend(
    *,
    use_color: bool = False,
    use_unicode: bool = False,
    stream: io.StringIO | None = None,
) -> OutputBackend:
    """Create a backend with test-friendly settings."""
    buf = stream or io.StringIO()
    return OutputBackend(use_color=use_color, use_unicode=use_unicode, stream=buf)


class TestInfraOutputStatus:
    """Tests for output status formatting using OutputBackend directly."""

    def test_success_status_contains_ok(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "flext-core", result=True, elapsed=1.23)
        text = buf.getvalue()
        tm.that(text, contains="[OK]")
        tm.that(text, contains="check")
        tm.that(text, contains="flext-core")
        tm.that(text, contains="1.23s")

    def test_failure_status_contains_fail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("lint", "flext-api", result=False, elapsed=0.45)
        text = buf.getvalue()
        tm.that(text, contains="[FAIL]")
        tm.that(text, contains="flext-api")

    def test_unicode_symbols(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=True, stream=buf)
        backend.status("test", "proj", result=True, elapsed=0.1)
        tm.that(buf.getvalue(), contains="✓")

    def test_color_codes_present_when_enabled(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=0.5)
        tm.that(buf.getvalue(), contains="\x1b[")


class TestInfraOutputSummary:
    """Tests for output summary formatting."""

    def test_summary_format(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary(
            "check", total=33, success=30, failed=2, skipped=1, elapsed=12.34
        )
        text = buf.getvalue()
        tm.that(text, contains="check summary")
        tm.that(text, contains="Total: 33")
        tm.that(text, contains="Success: 30")
        tm.that(text, contains="Failed: 2")
        tm.that(text, contains="Skipped: 1")
        tm.that(text, contains="12.34s")

    def test_summary_no_color_for_zero_counts(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=False, stream=buf)
        backend.summary("test", total=5, success=5, failed=0, skipped=0, elapsed=1.0)
        text = buf.getvalue()
        plain = _strip_ansi(text)
        tm.that(plain, contains="Failed: 0")
        tm.that(plain, contains="Skipped: 0")


class TestInfraOutputMessages:
    """Tests for error/warning/info message formatting."""

    def test_error_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.error("something broke")
        tm.that(buf.getvalue(), contains="ERROR: something broke")

    def test_error_with_detail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.error("fail", detail="see logs")
        text = buf.getvalue()
        tm.that(text, contains="ERROR: fail")
        tm.that(text, contains="see logs")

    def test_warning_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.warning("deprecated feature")
        tm.that(buf.getvalue(), contains="WARN: deprecated feature")

    def test_info_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.info("starting check")
        tm.that(buf.getvalue(), contains="INFO: starting check")


class TestInfraOutputHeader:
    """Tests for section header formatting."""

    def test_header_ascii(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.header("Quality Gates")
        text = buf.getvalue()
        tm.that(text, contains="=" * 60)
        tm.that(text, contains="Quality Gates")

    def test_header_unicode(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=True, stream=buf)
        backend.header("Quality Gates")
        tm.that(buf.getvalue(), contains="═" * 60)


class TestInfraOutputProgress:
    """Tests for progress indicator formatting."""

    def test_progress_format(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(1, 33, "flext-core", "check")
        text = buf.getvalue()
        tm.that(text, contains="[01/33]")
        tm.that(text, contains="flext-core")
        tm.that(text, contains="check ...")

    def test_progress_single_digit_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(3, 5, "proj", "test")
        tm.that(buf.getvalue(), contains="[3/5]")

    def test_progress_large_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(7, 100, "proj", "lint")
        tm.that(buf.getvalue(), contains="[007/100]")
