"""Tests for flext_infra.deps.__main__ — dispatch, structlog, argv, imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

from flext_infra.deps import __main__ as main_mod
from flext_infra.deps.__main__ import _SUBCOMMAND_MODULES, _main_impl, main
from flext_tests import tm
from tests.infra import t


def _fake_module(return_value: t.Infra.TomlValue = 0) -> ModuleType:
    mod = ModuleType("fake_subcommand")
    setattr(mod, "main", lambda: return_value)
    return mod


def _stub_import(mod: ModuleType) -> Callable[[str], ModuleType]:
    def _import(name: str) -> ModuleType:
        _ = name
        return mod

    return _import


def _patch_dispatch(
    mp: pytest.MonkeyPatch, argv: list[str], ret: t.Infra.TomlValue = 0
) -> None:
    mp.setattr(sys, "argv", argv)
    mp.setattr(
        main_mod,
        "importlib",
        SimpleNamespace(
            import_module=_stub_import(_fake_module(ret)),
        ),
    )


class TestMainSubcommandDispatch:
    @pytest.mark.parametrize("name", list(_SUBCOMMAND_MODULES.keys()))
    def test_dispatch_each_subcommand(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
    ) -> None:
        _patch_dispatch(monkeypatch, ["prog", name, "--workspace", "."])
        tm.that(_main_impl(), eq=0)


class TestMainModuleImport:
    @pytest.mark.parametrize(
        ("subcommand", "expected_module"),
        [
            ("detect", "flext_infra.deps.detector"),
            ("modernize", "flext_infra.deps.modernizer"),
            ("path-sync", "flext_infra.deps.path_sync"),
        ],
    )
    def test_imports_correct_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        subcommand: str,
        expected_module: str,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", subcommand, "--workspace", "."])
        imported: list[str] = []
        fake = _fake_module(0)

        def tracking_import(name: str) -> ModuleType:
            imported.append(name)
            return fake

        monkeypatch.setattr(
            main_mod,
            "importlib",
            SimpleNamespace(
                import_module=tracking_import,
            ),
        )
        _main_impl()
        tm.that(imported[0], eq=expected_module)


class TestMainSysArgvModification:
    def test_modifies_sys_argv_for_subcommand(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_dispatch(monkeypatch, ["prog", "detect", "--arg1", "value1"])
        _main_impl()
        tm.that("detect" in sys.argv[0], eq=True)

    def test_passes_remaining_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dispatch(monkeypatch, ["prog", "detect", "-q", "--no-fail"])
        _main_impl()
        tm.that("-q" in sys.argv, eq=True)
        tm.that("--no-fail" in sys.argv, eq=True)


class TestMainDelegation:
    def test_main_delegates_to_run_cli(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        received: list[Callable[[list[str] | None], int]] = []

        def fake_run_cli(fn: Callable[[list[str] | None], int]) -> int:
            received.append(fn)
            return 0

        monkeypatch.setattr(
            main_mod,
            "u",
            SimpleNamespace(Infra=SimpleNamespace(run_cli=fake_run_cli)),
        )
        tm.that(main(), eq=0)
        tm.that(len(received), eq=1)
        tm.that(received[0] is main_mod._main_impl, eq=True)


class TestMainExceptionHandling:
    def test_subcommand_exception_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "detect", "--workspace", "."])
        error_mod = ModuleType("error_mod")

        def raise_error() -> int:
            msg = "Test error"
            raise RuntimeError(msg)

        setattr(error_mod, "main", raise_error)
        monkeypatch.setattr(
            main_mod,
            "importlib",
            SimpleNamespace(
                import_module=_stub_import(error_mod),
            ),
        )
        tm.that(main(), eq=1)


def test_string_zero_return_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test string '0' return value normalization (edge case)."""
    _patch_dispatch(monkeypatch, ["prog", "detect", "--workspace", "."], "0")
    tm.that(_main_impl(), eq=0)
