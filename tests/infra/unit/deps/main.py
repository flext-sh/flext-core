"""Tests for flext_infra.deps.__main__ — dispatch, structlog, argv, imports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from flext_infra.deps import __main__ as main_mod
from flext_infra.deps.__main__ import _SUBCOMMANDS, main
from flext_tests import tm

_NO_STRUCTLOG = SimpleNamespace(ensure_structlog_configured=lambda: None)


def _fake_module(return_value: object = 0) -> ModuleType:
    mod = ModuleType("fake_subcommand")
    setattr(mod, "main", lambda: return_value)  # noqa: B010
    return mod


def _stub_import(mod: ModuleType) -> object:
    def _import(name: str) -> ModuleType:
        return mod

    return _import


def _patch_dispatch(mp: pytest.MonkeyPatch, argv: list[str], ret: object = 0) -> None:
    mp.setattr(sys, "argv", argv)
    mp.setattr(main_mod, "FlextRuntime", _NO_STRUCTLOG)
    mp.setattr(
        main_mod,
        "importlib",
        SimpleNamespace(
            import_module=_stub_import(_fake_module(ret)),
        ),
    )


class TestMainSubcommandDispatch:
    @pytest.mark.parametrize("name", list(_SUBCOMMANDS.keys()))
    def test_dispatch_each_subcommand(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
    ) -> None:
        _patch_dispatch(monkeypatch, ["prog", name])
        tm.that(main(), eq=0)


class TestMainReturnValues:
    @pytest.mark.parametrize(
        ("return_val", "expected"),
        [(0, 0), (None, 0), (False, 0), (42, 42), (True, 1), ("0", 0)],
        ids=["zero", "none", "false", "nonzero", "true", "string-zero"],
    )
    def test_return_value_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        return_val: object,
        expected: int,
    ) -> None:
        _patch_dispatch(monkeypatch, ["prog", "detect"], return_val)
        tm.that(main(), eq=expected)


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
        monkeypatch.setattr(sys, "argv", ["prog", subcommand])
        monkeypatch.setattr(main_mod, "FlextRuntime", _NO_STRUCTLOG)
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
        main()
        tm.that(imported[0], eq=expected_module)


class TestMainSysArgvModification:
    def test_modifies_sys_argv_for_subcommand(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_dispatch(monkeypatch, ["prog", "detect", "--arg1", "value1"])
        main()
        tm.that("detect" in sys.argv[0], eq=True)

    def test_passes_remaining_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_dispatch(monkeypatch, ["prog", "detect", "-q", "--no-fail"])
        main()
        tm.that("-q" in sys.argv, eq=True)
        tm.that("--no-fail" in sys.argv, eq=True)


class TestMainStructlogConfiguration:
    def test_ensures_structlog_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called: list[bool] = []
        monkeypatch.setattr(sys, "argv", ["prog", "detect"])
        monkeypatch.setattr(
            main_mod,
            "FlextRuntime",
            SimpleNamespace(
                ensure_structlog_configured=lambda: called.append(True),
            ),
        )
        monkeypatch.setattr(
            main_mod,
            "importlib",
            SimpleNamespace(
                import_module=_stub_import(_fake_module(0)),
            ),
        )
        main()
        tm.that(len(called), eq=1)

    def test_ensures_structlog_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        order: list[str] = []
        monkeypatch.setattr(sys, "argv", ["prog", "detect"])
        monkeypatch.setattr(
            main_mod,
            "FlextRuntime",
            SimpleNamespace(
                ensure_structlog_configured=lambda: order.append("ensure"),
            ),
        )

        def tracking_import(name: str) -> ModuleType:
            order.append("import")
            return _fake_module(0)

        monkeypatch.setattr(
            main_mod,
            "importlib",
            SimpleNamespace(
                import_module=tracking_import,
            ),
        )
        main()
        tm.that(order[0], eq="ensure")
        tm.that(order[1], eq="import")


class TestMainExceptionHandling:
    def test_subcommand_exception_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "detect"])
        monkeypatch.setattr(main_mod, "FlextRuntime", _NO_STRUCTLOG)
        error_mod = ModuleType("error_mod")

        def raise_error() -> int:
            msg = "Test error"
            raise RuntimeError(msg)

        setattr(error_mod, "main", raise_error)  # noqa: B010
        monkeypatch.setattr(
            main_mod,
            "importlib",
            SimpleNamespace(
                import_module=_stub_import(error_mod),
            ),
        )
        with pytest.raises(RuntimeError, match="Test error"):
            main()
