"""Tests for flext_infra.__main__ CLI entry point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from flext_infra.__main__ import FlextInfraMainCLI


def test_main_returns_error_when_no_args() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "Usage: python -m flext_infra" in completed.stderr


def test_main_help_flag_returns_zero() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Usage: python -m flext_infra" in completed.stderr


def test_main_unknown_group_returns_error() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra", "unknown"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "unknown group 'unknown'" in completed.stderr


@pytest.mark.parametrize("group", sorted(FlextInfraMainCLI.GROUPS.keys()))
def test_main_dispatches_group_help(group: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "flext_infra", group, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout.lower() or "usage:" in completed.stderr.lower()


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
