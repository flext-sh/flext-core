"""Tests for flext_infra output — terminal output utility.

Tests the _OutputBackend directly with custom config.
Uses u.Infra MRO for facade method verification.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import os
import re

from _pytest.monkeypatch import MonkeyPatch

from flext_infra import u
from flext_infra._utilities.output import (
    _OutputBackend,
)
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_tests import tm

ANSI_RE = re.compile(r"\033\[\d+m")

_should_use_color = FlextInfraUtilitiesTerminal.terminal_should_use_color
_should_use_unicode = FlextInfraUtilitiesTerminal.terminal_should_use_unicode


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _make_backend(
    *,
    use_color: bool = False,
    use_unicode: bool = False,
    stream: io.StringIO | None = None,
) -> _OutputBackend:
    """Create a backend with test-friendly settings."""
    buf = stream or io.StringIO()
    return _OutputBackend(use_color=use_color, use_unicode=use_unicode, stream=buf)


class TestShouldUseColor:
    """Tests for terminal_should_use_color detection."""

    def test_no_color_env_disables(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        tm.that(_should_use_color(), eq=False)

    def test_no_color_empty_string_disables(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "")
        tm.that(_should_use_color(), eq=False)

    def test_force_color_enables(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        tm.that(_should_use_color(), eq=True)

    def test_no_color_beats_force_color(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        tm.that(_should_use_color(), eq=False)

    def test_ci_env_disables(self, monkeypatch: MonkeyPatch) -> None:
        for var in ("CI", "GITHUB_ACTIONS", "GITLAB_CI"):
            for key in list(os.environ):
                monkeypatch.delenv(key, raising=False)
            monkeypatch.setenv(var, "true")
            tm.that(
                _should_use_color(),
                eq=False,
                msg=f"{var} should disable color",
            )

    def test_tty_with_xterm_enables(self, monkeypatch: MonkeyPatch) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        tm.that(_should_use_color(stream), eq=True)

    def test_tty_with_dumb_term_disables(self, monkeypatch: MonkeyPatch) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TERM", "dumb")
        tm.that(_should_use_color(stream), eq=False)

    def test_tty_with_empty_term_disables(self, monkeypatch: MonkeyPatch) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TERM", "")
        tm.that(_should_use_color(stream), eq=False)

    def test_non_tty_disables(self, monkeypatch: MonkeyPatch) -> None:
        stream = io.StringIO()
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        tm.that(_should_use_color(stream), eq=False)


class TestShouldUseUnicode:
    """Tests for terminal_should_use_unicode detection."""

    def test_utf8_lang_enables(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        tm.that(_should_use_unicode(), eq=True)

    def test_lc_all_utf8_enables(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LC_ALL", "en_US.utf8")
        tm.that(_should_use_unicode(), eq=True)

    def test_c_locale_disables(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LANG", "C")
        tm.that(_should_use_unicode(), eq=False)

    def test_empty_env_disables(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        tm.that(_should_use_unicode(), eq=False)

    def test_lc_all_takes_priority(self, monkeypatch: MonkeyPatch) -> None:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
        monkeypatch.setenv("LANG", "C")
        tm.that(_should_use_unicode(), eq=True)


class TestInfraOutputStatus:
    """Tests for output status formatting using _OutputBackend directly."""

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
