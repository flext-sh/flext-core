"""Tests for flext_infra.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from flext_infra import __main__ as main_mod
from flext_infra.__main__ import FlextInfraMainCLI


def test_main_returns_error_when_no_args() -> None:
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["flext-infra"]
        result = main_mod.main_inner()
    finally:
        sys.argv = original_argv
    assert result == 1


def test_main_help_flag_returns_zero() -> None:
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["flext-infra", "--help"]
        result = main_mod.main_inner()
    finally:
        sys.argv = original_argv
    assert result == 0


def test_main_unknown_group_returns_error() -> None:
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["flext-infra", "unknown"]
        result = main_mod.main_inner()
    finally:
        sys.argv = original_argv
    assert result == 1


def test_main_all_groups_defined() -> None:
    expected_groups = {
        "basemk",
        "check",
        "codegen",
        "core",
        "deps",
        "docs",
        "github",
        "maintenance",
        "refactor",
        "release",
        "workspace",
    }
    assert set(FlextInfraMainCLI.GROUPS.keys()) == expected_groups


def test_main_group_modules_are_valid() -> None:
    for group, module_path in FlextInfraMainCLI.GROUPS.items():
        assert isinstance(module_path, str)
        assert module_path.startswith("flext_infra.")
        assert (
            module_path.endswith(".__main__") or module_path == "flext_infra.refactor"
        )
        assert group in module_path
