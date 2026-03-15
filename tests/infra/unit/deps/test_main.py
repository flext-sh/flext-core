"""Tests for flext_infra.deps.__main__ subcommand dispatch.

Validates subcommand mapping, help/error paths, and return-value
normalization using real imports and pytest monkeypatch.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest

from flext_infra.deps import __main__ as deps_main
from flext_infra.deps.__main__ import _SUBCOMMAND_MODULES, _main_impl, main
from flext_tests import tm
from tests.infra import t


def _fake_module(return_value: t.Infra.TomlValue = 0) -> ModuleType:
    """Create a real ModuleType with a main() returning *return_value*."""
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
    """Patch sys.argv and importlib for dispatch tests."""
    mp.setattr(sys, "argv", argv)
    mp.setattr(
        deps_main,
        "importlib",
        SimpleNamespace(import_module=_stub_import(_fake_module(ret))),
    )


class TestSubcommandMapping:
    """Test subcommand mapping completeness."""

    EXPECTED_SUBCOMMAND_MODULES: dict[str, str] = {
        "detect": "flext_infra.deps.detector",
        "extra-paths": "flext_infra.deps.extra_paths",
        "internal-sync": "flext_infra.deps.internal_sync",
        "modernize": "flext_infra.deps.modernizer",
        "path-sync": "flext_infra.deps.path_sync",
    }

    def test_subcommands_count(self) -> None:
        """Test correct number of subcommands."""
        tm.that(len(_SUBCOMMAND_MODULES), eq=5)

    @pytest.mark.parametrize(
        ("name", "module"),
        list(EXPECTED_SUBCOMMAND_MODULES.items()),
        ids=list(EXPECTED_SUBCOMMAND_MODULES.keys()),
    )
    def test_subcommand_mapping(self, name: str, module: str) -> None:
        """Test each subcommand maps to correct module."""
        tm.that(name in _SUBCOMMAND_MODULES, eq=True, msg=f"Missing subcommand: {name}")
        tm.that(_SUBCOMMAND_MODULES[name], eq=module)

    @pytest.mark.parametrize("name", list(EXPECTED_SUBCOMMAND_MODULES.keys()))
    def test_subcommand_module_importable(self, name: str) -> None:
        """Test each subcommand module can be imported."""
        module = importlib.import_module(_SUBCOMMAND_MODULES[name])
        tm.that(hasattr(module, "main"), eq=True, msg=f"{name} module has no main()")


class TestMainHelpAndErrors:
    """Test main function help and error handling."""

    def test_main_with_help_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with -h flag returns 0."""
        monkeypatch.setattr(sys, "argv", ["prog", "-h"])
        tm.that(main(), eq=0)

    def test_main_with_no_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with no arguments returns 0 after help."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        result = main()
        tm.that(result, eq=0)

    def test_main_with_unknown_subcommand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main with unknown subcommand returns parser exit code."""
        monkeypatch.setattr(sys, "argv", ["prog", "unknown"])
        tm.that(main(), eq=2)


class TestMainReturnValues:
    """Test main function return value normalization."""

    @pytest.mark.parametrize(
        ("return_val", "expected"),
        [
            (0, 0),
            (None, 0),
            (False, 0),
            (42, 42),
            (True, 1),
        ],
        ids=["zero", "none", "false", "nonzero", "true"],
    )
    def test_return_value_normalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        return_val: t.Infra.TomlValue,
        expected: int,
    ) -> None:
        """Test _main_impl normalizes subcommand return values."""
        _patch_dispatch(monkeypatch, ["prog", "detect", "--workspace", "."], return_val)
        result = _main_impl()
        tm.that(result, eq=expected)
