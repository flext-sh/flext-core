"""Tests for release version and tag resolution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

import flext_infra.release.__main__ as _main_mod
from flext_core import r
from flext_infra.release.__main__ import _resolve_tag, _resolve_version
from flext_tests import tm


def _args(
    version: str | None, bump: str | None, interactive: int, tag: str = ""
) -> Namespace:
    return Namespace(version=version, bump=bump, interactive=interactive, tag=tag)


def _input_minor(_prompt: str) -> str:
    del _prompt
    return "minor"


def _input_invalid(_prompt: str) -> str:
    del _prompt
    return "invalid"


def _input_major(_prompt: str) -> str:
    del _prompt
    return "major"


def _stub_vs(
    *,
    parse: r[str] | None = None,
    current: r[str] | None = None,
    bump: r[str] | None = None,
) -> type:
    """Build a fake FlextInfraUtilitiesVersioning with configurable returns."""

    class _F:
        def parse_semver(self, version: str) -> r[str]:
            return parse if parse is not None else r[str].ok(version)

        def current_workspace_version(self, root: Path) -> r[str]:
            del root
            return current if current is not None else r[str].ok("1.0.0")

        def bump_version(self, cur: str, kind: str) -> r[str]:
            del cur, kind
            return bump if bump is not None else r[str].ok("1.1.0")

    return _F


class TestReleaseMainVersionResolution:
    """Test version resolution logic."""

    def test_resolve_version_explicit(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(parse=r[str].ok("1.0.0")),
        )
        args = _args(version="1.0.0", bump="", interactive=1)
        tm.that(_resolve_version(args, tmp_path), eq="1.0.0")

    def test_resolve_version_invalid_explicit(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(parse=r[str].fail("invalid")),
        )
        args = _args(version="invalid", bump="", interactive=1)
        with pytest.raises(RuntimeError):
            _resolve_version(args, tmp_path)

    def test_resolve_version_from_current(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("0.9.0")),
        )
        args = _args(version="", bump="", interactive=0)
        tm.that(_resolve_version(args, tmp_path), eq="0.9.0")

    def test_resolve_version_current_read_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].fail("read error")),
        )
        args = _args(version="", bump="", interactive=1)
        with pytest.raises(RuntimeError):
            _resolve_version(args, tmp_path)

    def test_resolve_version_with_bump(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0"), bump=r[str].ok("1.1.0")),
        )
        args = _args(version="", bump="minor", interactive=1)
        tm.that(_resolve_version(args, tmp_path), eq="1.1.0")

    def test_resolve_version_bump_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0"), bump=r[str].fail("invalid bump")),
        )
        args = _args(version="", bump="invalid", interactive=1)
        with pytest.raises(RuntimeError):
            _resolve_version(args, tmp_path)

    def test_resolve_version_interactive_input(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0"), bump=r[str].ok("1.1.0")),
        )
        monkeypatch.setattr("builtins.input", _input_minor)
        args = _args(version="", bump="", interactive=1)
        tm.that(_resolve_version(args, tmp_path), eq="1.1.0")

    def test_resolve_version_interactive_invalid_input(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0")),
        )
        monkeypatch.setattr("builtins.input", _input_invalid)
        args = _args(version="", bump="", interactive=1)
        with pytest.raises(RuntimeError):
            _resolve_version(args, tmp_path)

    def test_resolve_version_non_interactive(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0")),
        )
        args = _args(version="", bump="", interactive=0)
        tm.that(_resolve_version(args, tmp_path), eq="1.0.0")


class TestResolveVersionInteractive:
    """Test _resolve_version with interactive mode edge cases."""

    def test_resolve_version_interactive_invalid_bump(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0")),
        )
        monkeypatch.setattr("builtins.input", _input_invalid)
        args = _args(version=None, bump=None, interactive=1)
        with pytest.raises(RuntimeError, match="invalid bump type"):
            _resolve_version(args, tmp_path)

    def test_resolve_version_interactive_bump_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _main_mod,
            "FlextInfraUtilitiesVersioning",
            _stub_vs(current=r[str].ok("1.0.0"), bump=r[str].fail("bump failed")),
        )
        monkeypatch.setattr("builtins.input", _input_major)
        args = _args(version=None, bump=None, interactive=1)
        with pytest.raises(RuntimeError, match="bump failed"):
            _resolve_version(args, tmp_path)


class TestReleaseMainTagResolution:
    """Test tag resolution logic."""

    def test_resolve_tag_explicit(self) -> None:
        tm.that(
            _resolve_tag(
                _args(version="", bump="", interactive=0, tag="v1.0.0"), "1.0.0"
            ),
            eq="v1.0.0",
        )

    def test_resolve_tag_invalid_prefix(self) -> None:
        with pytest.raises(RuntimeError):
            _resolve_tag(
                _args(version="", bump="", interactive=0, tag="1.0.0"), "1.0.0"
            )

    def test_resolve_tag_auto_generated(self) -> None:
        tm.that(
            _resolve_tag(_args(version="", bump="", interactive=0, tag=""), "1.0.0"),
            eq="v1.0.0",
        )
