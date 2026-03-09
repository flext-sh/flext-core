"""Tests for flext_infra output — terminal output utility.

Tests the _OutputBackend directly with custom config.
Uses u.Infra MRO for facade method verification.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import re
from unittest.mock import patch

from flext_infra import u
from flext_infra._utilities.output import (
    _OutputBackend,
)
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal

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

    def test_no_color_env_disables(self) -> None:
        with patch.dict("os.environ", {"NO_COLOR": "1"}, clear=False):
            assert _should_use_color() is False

    def test_no_color_empty_string_disables(self) -> None:
        with patch.dict("os.environ", {"NO_COLOR": ""}, clear=False):
            assert _should_use_color() is False

    def test_force_color_enables(self) -> None:
        env = {"FORCE_COLOR": "1"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color() is True

    def test_no_color_beats_force_color(self) -> None:
        env = {"NO_COLOR": "1", "FORCE_COLOR": "1"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color() is False

    def test_ci_env_disables(self) -> None:
        for var in ("CI", "GITHUB_ACTIONS", "GITLAB_CI"):
            env = {var: "true"}
            with patch.dict("os.environ", env, clear=True):
                assert _should_use_color() is False, f"{var} should disable color"

    def test_tty_with_xterm_enables(self) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        env = {"TERM": "xterm-256color"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color(stream) is True

    def test_tty_with_dumb_term_disables(self) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        env = {"TERM": "dumb"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color(stream) is False

    def test_tty_with_empty_term_disables(self) -> None:
        stream = io.StringIO()
        object.__setattr__(stream, "isatty", lambda: True)
        with patch.dict("os.environ", {"TERM": ""}, clear=True):
            assert _should_use_color(stream) is False

    def test_non_tty_disables(self) -> None:
        stream = io.StringIO()
        with patch.dict("os.environ", {}, clear=True):
            assert _should_use_color(stream) is False


class TestShouldUseUnicode:
    """Tests for terminal_should_use_unicode detection."""

    def test_utf8_lang_enables(self) -> None:
        with patch.dict("os.environ", {"LANG": "en_US.UTF-8"}, clear=True):
            assert _should_use_unicode() is True

    def test_lc_all_utf8_enables(self) -> None:
        with patch.dict("os.environ", {"LC_ALL": "en_US.utf8"}, clear=True):
            assert _should_use_unicode() is True

    def test_c_locale_disables(self) -> None:
        with patch.dict("os.environ", {"LANG": "C"}, clear=True):
            assert _should_use_unicode() is False

    def test_empty_env_disables(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _should_use_unicode() is False

    def test_lc_all_takes_priority(self) -> None:
        env = {"LC_ALL": "en_US.UTF-8", "LANG": "C"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_unicode() is True


class TestInfraOutputStatus:
    """Tests for output status formatting using _OutputBackend directly."""

    def test_success_status_contains_ok(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "flext-core", result=True, elapsed=1.23)
        text = buf.getvalue()
        assert "[OK]" in text
        assert "check" in text
        assert "flext-core" in text
        assert "1.23s" in text

    def test_failure_status_contains_fail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("lint", "flext-api", result=False, elapsed=0.45)
        text = buf.getvalue()
        assert "[FAIL]" in text
        assert "flext-api" in text

    def test_unicode_symbols(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=True, stream=buf)
        backend.status("test", "proj", result=True, elapsed=0.1)
        assert "✓" in buf.getvalue()

    def test_color_codes_present_when_enabled(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=0.5)
        assert "\x1b[" in buf.getvalue()


class TestInfraOutputSummary:
    """Tests for output summary formatting."""

    def test_summary_format(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary(
            "check", total=33, success=30, failed=2, skipped=1, elapsed=12.34
        )
        text = buf.getvalue()
        assert "check summary" in text
        assert "Total: 33" in text
        assert "Success: 30" in text
        assert "Failed: 2" in text
        assert "Skipped: 1" in text
        assert "12.34s" in text

    def test_summary_no_color_for_zero_counts(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=False, stream=buf)
        backend.summary("test", total=5, success=5, failed=0, skipped=0, elapsed=1.0)
        text = buf.getvalue()
        plain = _strip_ansi(text)
        assert "Failed: 0" in plain
        assert "Skipped: 0" in plain


class TestInfraOutputMessages:
    """Tests for error/warning/info message formatting."""

    def test_error_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.error("something broke")
        assert "ERROR: something broke" in buf.getvalue()

    def test_error_with_detail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.error("fail", detail="see logs")
        text = buf.getvalue()
        assert "ERROR: fail" in text
        assert "see logs" in text

    def test_warning_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.warning("deprecated feature")
        assert "WARN: deprecated feature" in buf.getvalue()

    def test_info_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.info("starting check")
        assert "INFO: starting check" in buf.getvalue()


class TestInfraOutputHeader:
    """Tests for section header formatting."""

    def test_header_ascii(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.header("Quality Gates")
        text = buf.getvalue()
        assert "=" * 60 in text
        assert "Quality Gates" in text

    def test_header_unicode(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=True, stream=buf)
        backend.header("Quality Gates")
        assert "═" * 60 in buf.getvalue()


class TestInfraOutputProgress:
    """Tests for progress indicator formatting."""

    def test_progress_format(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(1, 33, "flext-core", "check")
        text = buf.getvalue()
        assert "[01/33]" in text
        assert "flext-core" in text
        assert "check ..." in text

    def test_progress_single_digit_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(3, 5, "proj", "test")
        assert "[3/5]" in buf.getvalue()

    def test_progress_large_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(7, 100, "proj", "lint")
        assert "[007/100]" in buf.getvalue()


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
        assert "\x1b[" not in buf.getvalue()


class TestMroFacadeMethods:
    """Tests for u.Infra MRO facade methods."""

    def test_output_methods_accessible_via_mro(self) -> None:
        assert callable(u.Infra.info)
        assert callable(u.Infra.error)
        assert callable(u.Infra.warning)
        assert callable(u.Infra.status)
        assert callable(u.Infra.summary)
        assert callable(u.Infra.header)
        assert callable(u.Infra.progress)
        assert callable(u.Infra.debug)
        assert callable(u.Infra.gate_result)


class TestInfraOutputEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_status_with_zero_elapsed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=0.0)
        assert "0.00s" in buf.getvalue()

    def test_status_with_large_elapsed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.status("check", "proj", result=True, elapsed=999.99)
        assert "999.99s" in buf.getvalue()

    def test_summary_with_all_zeros(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary("test", total=0, success=0, failed=0, skipped=0, elapsed=0.0)
        text = buf.getvalue()
        assert "Total: 0" in text
        assert "Success: 0" in text

    def test_summary_with_large_numbers(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.summary(
            "check", total=1000, success=950, failed=40, skipped=10, elapsed=123.45
        )
        text = buf.getvalue()
        assert "Total: 1000" in text
        assert "Success: 950" in text
        assert "Failed: 40" in text

    def test_error_with_multiline_detail(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        detail = "line 1\nline 2\nline 3"
        backend.error("multi", detail=detail)
        text = buf.getvalue()
        assert "ERROR: multi" in text
        assert "line 1" in text
        assert "line 3" in text

    def test_header_with_long_title(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        long_title = "A" * 100
        backend.header(long_title)
        text = buf.getvalue()
        assert long_title in text

    def test_progress_with_same_current_and_total(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.progress(5, 5, "proj", "test")
        assert "[5/5]" in buf.getvalue()

    def test_multiple_messages_in_sequence(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.info("msg1")
        backend.warning("msg2")
        backend.error("msg3")
        text = buf.getvalue()
        assert "INFO: msg1" in text
        assert "WARN: msg2" in text
        assert "ERROR: msg3" in text

    def test_color_and_unicode_together(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_color=True, use_unicode=True, stream=buf)
        backend.status("test", "proj", result=True, elapsed=0.1)
        text = buf.getvalue()
        assert "✓" in text or "\x1b[" in text

    def test_gate_result_passed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.gate_result("ruff", count=0, passed=True, elapsed=0.5)
        text = buf.getvalue()
        assert "[OK]" in text
        assert "ruff" in text

    def test_gate_result_failed(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(use_unicode=False, stream=buf)
        backend.gate_result("mypy", count=3, passed=False, elapsed=1.2)
        text = buf.getvalue()
        assert "[FAIL]" in text
        assert "3 errors" in text

    def test_debug_message(self) -> None:
        buf = io.StringIO()
        backend = _make_backend(stream=buf)
        backend.debug("trace info")
        assert "DEBUG: trace info" in buf.getvalue()
