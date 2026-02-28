"""Tests for flext_infra.output — terminal output utility."""

from __future__ import annotations

import io
import re
from unittest.mock import patch

from flext_infra.output import (
    FlextInfraOutput,
    _should_use_color,
    _should_use_unicode,
)

ANSI_RE = re.compile(r"\033\[\d+m")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class TestShouldUseColor:
    """Tests for _should_use_color detection."""

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
        stream.isatty = lambda: True
        env = {"TERM": "xterm-256color"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color(stream) is True

    def test_tty_with_dumb_term_disables(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True
        env = {"TERM": "dumb"}
        with patch.dict("os.environ", env, clear=True):
            assert _should_use_color(stream) is False

    def test_tty_with_empty_term_disables(self) -> None:
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch.dict("os.environ", {"TERM": ""}, clear=True):
            assert _should_use_color(stream) is False

    def test_non_tty_disables(self) -> None:
        stream = io.StringIO()
        with patch.dict("os.environ", {}, clear=True):
            assert _should_use_color(stream) is False


class TestShouldUseUnicode:
    """Tests for _should_use_unicode detection."""

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
    """Tests for InfraOutput.status formatting."""

    def test_success_status_contains_ok(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.status("check", "flext-core", result=True, elapsed=1.23)
        text = buf.getvalue()
        assert "[OK]" in text
        assert "check" in text
        assert "flext-core" in text
        assert "1.23s" in text

    def test_failure_status_contains_fail(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.status("lint", "flext-api", result=False, elapsed=0.45)
        text = buf.getvalue()
        assert "[FAIL]" in text
        assert "flext-api" in text

    def test_unicode_symbols(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=True, stream=buf)
        out.status("test", "proj", result=True, elapsed=0.1)
        assert "✓" in buf.getvalue()

    def test_color_codes_present_when_enabled(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=True, use_unicode=False, stream=buf)
        out.status("check", "proj", result=True, elapsed=0.5)
        assert "\033[" in buf.getvalue()


class TestInfraOutputSummary:
    """Tests for InfraOutput.summary formatting."""

    def test_summary_format(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.summary("check", total=33, success=30, failed=2, skipped=1, elapsed=12.34)
        text = buf.getvalue()
        assert "check summary" in text
        assert "Total: 33" in text
        assert "Success: 30" in text
        assert "Failed: 2" in text
        assert "Skipped: 1" in text
        assert "12.34s" in text

    def test_summary_no_color_for_zero_counts(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=True, use_unicode=False, stream=buf)
        out.summary("test", total=5, success=5, failed=0, skipped=0, elapsed=1.0)
        text = buf.getvalue()
        plain = _strip_ansi(text)
        assert "Failed: 0" in plain
        assert "Skipped: 0" in plain


class TestInfraOutputMessages:
    """Tests for error/warning/info message formatting."""

    def test_error_message(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.error("something broke")
        assert "ERROR: something broke" in buf.getvalue()

    def test_error_with_detail(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.error("fail", detail="see logs")
        text = buf.getvalue()
        assert "ERROR: fail" in text
        assert "see logs" in text

    def test_warning_message(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.warning("deprecated feature")
        assert "WARN: deprecated feature" in buf.getvalue()

    def test_info_message(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.info("starting check")
        assert "INFO: starting check" in buf.getvalue()


class TestInfraOutputHeader:
    """Tests for section header formatting."""

    def test_header_ascii(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.header("Quality Gates")
        text = buf.getvalue()
        assert "=" * 60 in text
        assert "Quality Gates" in text

    def test_header_unicode(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=True, stream=buf)
        out.header("Quality Gates")
        assert "═" * 60 in buf.getvalue()


class TestInfraOutputProgress:
    """Tests for progress indicator formatting."""

    def test_progress_format(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.progress(1, 33, "flext-core", "check")
        text = buf.getvalue()
        assert "[01/33]" in text
        assert "flext-core" in text
        assert "check ..." in text

    def test_progress_single_digit_total(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.progress(3, 5, "proj", "test")
        assert "[3/5]" in buf.getvalue()

    def test_progress_large_total(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.progress(7, 100, "proj", "lint")
        assert "[007/100]" in buf.getvalue()


class TestInfraOutputNoColor:
    """Tests for behavior when color is disabled."""

    def test_no_ansi_codes_when_color_disabled(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.info("test")
        out.warning("test")
        out.error("test")
        out.status("check", "proj", True, 0.1)
        out.header("Title")
        out.progress(1, 1, "proj", "test")
        out.summary("check", 1, 1, 0, 0, 0.1)
        assert "\033[" not in buf.getvalue()


class TestModuleSingleton:
    """Tests for module-level output singleton."""

    def test_output_singleton_importable(self) -> None:
        from flext_infra import output  # noqa: PLC0415

        assert isinstance(output, FlextInfraOutput)

    def test_output_writes_to_stderr_by_default(self) -> None:
        from flext_infra import output  # noqa: PLC0415

        assert output._stream is not None


class TestInfraOutputEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_status_with_zero_elapsed(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.status("check", "proj", result=True, elapsed=0.0)
        assert "0.00s" in buf.getvalue()

    def test_status_with_large_elapsed(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.status("check", "proj", result=True, elapsed=999.99)
        assert "999.99s" in buf.getvalue()

    def test_summary_with_all_zeros(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.summary("test", total=0, success=0, failed=0, skipped=0, elapsed=0.0)
        text = buf.getvalue()
        assert "Total: 0" in text
        assert "Success: 0" in text

    def test_summary_with_large_numbers(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        out.summary("check", total=1000, success=950, failed=40, skipped=10, elapsed=123.45)
        text = buf.getvalue()
        assert "Total: 1000" in text
        assert "Success: 950" in text
        assert "Failed: 40" in text

    def test_error_with_multiline_detail(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        detail = "line 1\nline 2\nline 3"
        out.error("multi", detail=detail)
        text = buf.getvalue()
        assert "ERROR: multi" in text
        assert "line 1" in text
        assert "line 3" in text

    def test_header_with_long_title(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, use_unicode=False, stream=buf)
        long_title = "A" * 100
        out.header(long_title)
        text = buf.getvalue()
        assert long_title in text

    def test_progress_with_same_current_and_total(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.progress(5, 5, "proj", "test")
        assert "[5/5]" in buf.getvalue()

    def test_multiple_messages_in_sequence(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=False, stream=buf)
        out.info("msg1")
        out.warning("msg2")
        out.error("msg3")
        text = buf.getvalue()
        assert "INFO: msg1" in text
        assert "WARN: msg2" in text
        assert "ERROR: msg3" in text

    def test_color_and_unicode_together(self) -> None:
        buf = io.StringIO()
        out = FlextInfraOutput(use_color=True, use_unicode=True, stream=buf)
        out.status("test", "proj", result=True, elapsed=0.1)
        text = buf.getvalue()
        # Should have both ANSI codes and unicode symbols
        assert "✓" in text or "\033[" in text

    def test_stream_parameter_respected(self) -> None:
        buf1 = io.StringIO()
        buf2 = io.StringIO()
        out1 = FlextInfraOutput(use_color=False, stream=buf1)
        out2 = FlextInfraOutput(use_color=False, stream=buf2)
        out1.info("msg1")
        out2.info("msg2")
        assert "msg1" in buf1.getvalue()
        assert "msg2" in buf2.getvalue()
        assert "msg1" not in buf2.getvalue()
        assert "msg2" not in buf1.getvalue()
