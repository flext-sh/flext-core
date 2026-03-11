"""Tests for codegen CLI entry point (__main__.py).

Validates CLI argument parsing, command dispatch, and exit codes
using real service instances with temporary workspaces.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flext_infra.codegen import __main__ as codegen_main
from flext_tests import tm


class TestHandleLazyInit:
    """Tests for _handle_lazy_init with real service instances."""

    def test_success(self, tmp_path: Path) -> None:
        """_handle_lazy_init returns 0 on empty workspace."""
        result = codegen_main.main(["lazy-init", "--root", str(tmp_path)])
        tm.that(result, eq=0)

    def test_check_mode(self, tmp_path: Path) -> None:
        """_handle_lazy_init respects --check flag."""
        result = codegen_main.main(
            ["lazy-init", "--check", "--root", str(tmp_path)],
        )
        tm.that(result, eq=0)

    def test_enforce_mode(self, tmp_path: Path) -> None:
        """_handle_lazy_init in enforce mode (not check)."""
        result = codegen_main.main(["lazy-init", "--root", str(tmp_path)])
        tm.that(result, eq=0)


class TestMainCommandDispatch:
    """Tests for main() command routing."""

    def test_lazy_init_command(self, tmp_path: Path) -> None:
        """main() with lazy-init command returns 0."""
        result = codegen_main.main(["lazy-init", "--root", str(tmp_path)])
        tm.that(result, eq=0)

    def test_lazy_init_with_check_flag(self, tmp_path: Path) -> None:
        """main() lazy-init with --check flag parses correctly."""
        result = codegen_main.main(
            ["lazy-init", "--check", "--root", str(tmp_path)],
        )
        tm.that(result, eq=0)

    def test_lazy_init_default_root(self) -> None:
        """main() lazy-init uses cwd as default root."""
        result = codegen_main.main(["lazy-init"])
        tm.that(result, eq=0)

    def test_unknown_command(self) -> None:
        """main() with unknown command exits with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            codegen_main.main(["unknown-command"])
        tm.that(exc_info.value.code, eq=2)

    def test_no_command(self) -> None:
        """main() with no command exits with code 2."""
        argv: list[str] = []
        with pytest.raises(SystemExit) as exc_info:
            codegen_main.main(argv)
        tm.that(exc_info.value.code, eq=2)

    def test_lazy_init_with_custom_root(self, tmp_path: Path) -> None:
        """main() lazy-init with custom root directory."""
        custom_root = tmp_path / "custom"
        result = codegen_main.main(["lazy-init", "--root", str(custom_root)])
        tm.that(result, eq=0)


class TestMainEntryPoint:
    """Tests for __main__ module entry point."""

    def test_entry_point_returns_int(self) -> None:
        """main() returns an integer exit code."""
        result = codegen_main.main(["lazy-init"])
        tm.that(type(result).__name__, eq="int")

    def test_entry_point_via_sys_exit(self) -> None:
        """__main__ entry point via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "flext_infra.codegen", "lazy-init", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/marlonsc/flext/flext-core",
            check=False,
        )
        tm.that(result.returncode, eq=0)
        tm.that(result.stdout, contains="lazy-init")


__all__: list[str] = []
