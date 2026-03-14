"""Tests for release main() orchestration flow.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from _pytest.monkeypatch import MonkeyPatch

import flext_infra.release.__main__ as _main_mod
from flext_core import r, t
from flext_infra.release.__main__ import main
from flext_tests import tm


def _patch_main_deps(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    *,
    root_result: r[Path] | None = None,
    release_result: r[bool] | None = None,
    capture: list[SimpleNamespace] | None = None,
    error_calls: list[str] | None = None,
) -> None:
    """Patch all main() dependencies via monkeypatch."""

    class _Rt:
        @staticmethod
        def ensure_structlog_configured() -> None:
            pass

    class _Ps:
        def workspace_root(self, hint: Path) -> r[Path]:
            del hint
            return root_result if root_result is not None else r[Path].ok(tmp_path)

    class _Vs:
        def current_workspace_version(self, root: Path) -> r[str]:
            del root
            return r[str].ok("1.0.0")

        def parse_semver(self, version: str) -> r[str]:
            return r[str].ok(version)

        def bump_version(self, cur: str, kind: str) -> r[str]:
            del cur, kind
            return r[str].ok("1.1.0")

    class _Or:
        def run_release(self, **kwargs: t.Scalar) -> r[bool]:
            if capture is not None:
                capture.append(SimpleNamespace(**kwargs))
            return release_result if release_result is not None else r[bool].ok(True)

    class _Out:
        @staticmethod
        def error(msg: str) -> None:
            if error_calls is not None:
                error_calls.append(msg)

    monkeypatch.setattr(_main_mod, "FlextRuntime", _Rt)
    monkeypatch.setattr(_main_mod, "FlextInfraUtilitiesPaths", _Ps)
    monkeypatch.setattr(_main_mod, "FlextInfraUtilitiesVersioning", _Vs)
    monkeypatch.setattr(_main_mod, "FlextInfraReleaseOrchestrator", _Or)
    monkeypatch.setattr(_main_mod, "output", _Out)


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["prog", "--root", str(tmp_path), *extra]


class TestReleaseMainFlow:
    """Test main() orchestration."""

    def test_main_success(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", _argv(tmp_path, "--phase", "validate", "--interactive", "0")
        )
        _patch_main_deps(monkeypatch, tmp_path)
        tm.that(main(), eq=0)

    def test_main_workspace_root_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", _argv(tmp_path, "--phase", "validate", "--interactive", "0")
        )
        errors: list[str] = []
        _patch_main_deps(
            monkeypatch,
            tmp_path,
            root_result=r[Path].fail("not found"),
            error_calls=errors,
        )
        tm.that(main(), eq=1)
        tm.that(len(errors), eq=1)

    def test_main_version_resolution_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", _argv(tmp_path, "--phase", "version", "--version", "invalid")
        )

        class _FailVs:
            def parse_semver(self, version: str) -> r[str]:
                del version
                return r[str].fail("invalid")

            def current_workspace_version(self, root: Path) -> r[str]:
                del root
                return r[str].ok("1.0.0")

            def bump_version(self, cur: str, kind: str) -> r[str]:
                del cur, kind
                return r[str].ok("1.1.0")

        errors: list[str] = []
        _patch_main_deps(monkeypatch, tmp_path, error_calls=errors)
        monkeypatch.setattr(_main_mod, "FlextInfraUtilitiesVersioning", _FailVs)
        tm.that(main(), eq=1)
        tm.that(len(errors), eq=1)

    def test_main_release_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys, "argv", _argv(tmp_path, "--phase", "validate", "--interactive", "0")
        )
        errors: list[str] = []
        _patch_main_deps(
            monkeypatch,
            tmp_path,
            release_result=r[bool].fail("release failed"),
            error_calls=errors,
        )
        tm.that(main(), eq=1)
        tm.that(len(errors), eq=1)

    def test_main_all_phases(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", _argv(tmp_path, "--phase", "all", "--interactive", "0")
        )
        calls: list[SimpleNamespace] = []
        _patch_main_deps(monkeypatch, tmp_path, capture=calls)
        tm.that(main(), eq=0)
        tm.that(calls[0].phases, eq=["validate", "version", "build", "publish"])

    def test_main_with_push(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(tmp_path, "--phase", "validate", "--push", "--interactive", "0"),
        )
        calls: list[SimpleNamespace] = []
        _patch_main_deps(monkeypatch, tmp_path, capture=calls)
        tm.that(main(), eq=0)
        tm.that(calls[0].push, eq=True)

    def test_main_with_dry_run(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(tmp_path, "--phase", "validate", "--dry-run", "--interactive", "0"),
        )
        calls: list[SimpleNamespace] = []
        _patch_main_deps(monkeypatch, tmp_path, capture=calls)
        tm.that(main(), eq=0)
        tm.that(calls[0].dry_run, eq=True)

    def test_main_with_projects(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(tmp_path, "--phase", "validate", "--projects", "proj1", "proj2"),
        )
        calls: list[SimpleNamespace] = []
        _patch_main_deps(monkeypatch, tmp_path, capture=calls)
        tm.that(main(), eq=0)
        tm.that(calls[0].project_names, eq=["proj1", "proj2"])
