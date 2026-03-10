"""Tests for flext_infra terminal detection — color and unicode support.

Tests terminal_should_use_color and terminal_should_use_unicode detection
based on environment variables and terminal capabilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import os

from _pytest.monkeypatch import MonkeyPatch

from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_tests import tm

_should_use_color = FlextInfraUtilitiesTerminal.terminal_should_use_color
_should_use_unicode = FlextInfraUtilitiesTerminal.terminal_should_use_unicode


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
