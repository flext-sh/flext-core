"""Tests for flext_infra.deps.__main__ subcommand dispatch.

Validates subcommand mapping, help/error paths, and return-value
normalization using real imports and pytest monkeypatch.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flext_infra.deps import __main__ as deps_main
from flext_infra.deps.__main__ import _SUBCOMMANDS, main
from flext_tests import tm

_NO_STRUCTLOG = SimpleNamespace(ensure_structlog_configured=lambda: None)


def _fake_module(return_value: object = 0) -> ModuleType:
    """Create a real ModuleType with a main() returning *return_value*."""
    mod = ModuleType("fake_subcommand")
    setattr(mod, "main", lambda: return_value)
    return mod


def _stub_import(mod: ModuleType) -> object:
    def _import(name: str) -> ModuleType:
        return mod

    return _import


def _patch_dispatch(mp: pytest.MonkeyPatch, argv: list[str], ret: object = 0) -> None:
    """Patch sys.argv, FlextRuntime, and importlib for dispatch tests."""
    mp.setattr(sys, "argv", argv)
    mp.setattr(deps_main, "FlextRuntime", _NO_STRUCTLOG)
    mp.setattr(
        deps_main,
        "importlib",
        SimpleNamespace(import_module=_stub_import(_fake_module(ret))),
    )


class TestSubcommandMapping:
    """Test subcommand mapping completeness."""

    EXPECTED_SUBCOMMANDS: dict[str, str] = {
        "detect": "flext_infra.deps.detector",
        "extra-paths": "flext_infra.deps.extra_paths",
        "internal-sync": "flext_infra.deps.internal_sync",
        "modernize": "flext_infra.deps.modernizer",
        "path-sync": "flext_infra.deps.path_sync",
    }

    def test_subcommands_count(self) -> None:
        """Test correct number of subcommands."""
        tm.that(len(_SUBCOMMANDS), eq=5)

    @pytest.mark.parametrize(
        ("name", "module"),
        list(EXPECTED_SUBCOMMANDS.items()),
        ids=list(EXPECTED_SUBCOMMANDS.keys()),
    )
    def test_subcommand_mapping(self, name: str, module: str) -> None:
        """Test each subcommand maps to correct module."""
        tm.that(name in _SUBCOMMANDS, eq=True, msg=f"Missing subcommand: {name}")
        tm.that(_SUBCOMMANDS[name], eq=module)

    @pytest.mark.parametrize("name", list(EXPECTED_SUBCOMMANDS.keys()))
    def test_subcommand_module_importable(self, name: str) -> None:
        """Test each subcommand module can be imported."""
        module = importlib.import_module(_SUBCOMMANDS[name])
        tm.that(hasattr(module, "main"), eq=True, msg=f"{name} module has no main()")


class TestMainHelpAndErrors:
    """Test main function help and error handling."""

    def test_main_with_help_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with -h flag returns 0."""
        monkeypatch.setattr(sys, "argv", ["prog", "-h"])
        result = main()
        tm.that(result, eq=0)

    def test_main_with_no_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test main with no arguments returns 1."""
        monkeypatch.setattr(sys, "argv", ["prog"])
        result = main()
        tm.that(result, eq=1)

    def test_main_with_unknown_subcommand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test main with unknown subcommand returns 1."""
        monkeypatch.setattr(sys, "argv", ["prog", "unknown"])
        result = main()
        tm.that(result, eq=1)


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
        return_val: object,
        expected: int,
    ) -> None:
        """Test main normalizes subcommand return values."""
        _patch_dispatch(monkeypatch, ["prog", "detect"], return_val)
        result = main()
        tm.that(result, eq=expected)
