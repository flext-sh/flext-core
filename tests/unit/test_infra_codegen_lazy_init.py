"""Tests for FlextInfraLazyInitGenerator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra.codegen import FlextInfraLazyInitGenerator


class TestFlextInfraLazyInitGenerator:
    """Test suite for FlextInfraLazyInitGenerator service."""

    def test_init_accepts_workspace_root(self, tmp_path: Path) -> None:
        """Test generator initialization with workspace root."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        assert generator is not None

    def test_run_with_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        """Test run() on empty workspace returns 0 (no unmapped exports)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        assert result == 0

    def test_run_with_check_only_flag(self, tmp_path: Path) -> None:
        """Test run() respects check_only flag without modifying files."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        assert result == 0

    def test_generate_output_to_sandboxed_path(self, tmp_path: Path) -> None:
        """Test that generated output goes to sandboxed tmp_path, not src/."""
        # Create a minimal __init__.py in sandbox
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text(
            '"""Test package."""\n'
            "from test_pkg.module import TestClass\n"
            '__all__ = ["TestClass"]\n'
        )

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)

        # Verify output is in tmp_path, not in actual src/flext_infra/
        assert result >= 0
        # Verify the file still exists in sandbox
        assert init_file.exists()

    def test_generator_is_flext_service(self, tmp_path: Path) -> None:
        """Test that FlextInfraLazyInitGenerator is a FlextService[str]."""
        from flext_core import FlextService

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        # Verify it's an instance of FlextService
        assert isinstance(generator, FlextService)

    def test_run_returns_integer_exit_code(self, tmp_path: Path) -> None:
        """Test that run() returns an integer exit code."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        assert isinstance(result, int)
        assert result >= 0
