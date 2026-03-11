"""Tests for FlextInfraCodegenLazyInit service-level behavior.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import FlextService
from flext_infra.codegen import FlextInfraCodegenLazyInit
from flext_tests import tm


class TestFlextInfraCodegenLazyInit:
    """Test suite for FlextInfraCodegenLazyInit service."""

    def test_init_accepts_workspace_root(self, tmp_path: Path) -> None:
        """Test generator initialization with workspace root."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        tm.that(generator, none=False)

    def test_run_with_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        """Test run() on empty workspace returns 0 (no errors)."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)

    def test_run_with_check_only_flag(self, tmp_path: Path) -> None:
        """Test run() respects check_only flag without modifying files."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(result, eq=0)

    def test_generator_is_flext_service(self, tmp_path: Path) -> None:
        """Test that FlextInfraCodegenLazyInit is a FlextService."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        tm.that(generator, is_=FlextService)

    def test_run_returns_integer_exit_code(self, tmp_path: Path) -> None:
        """Test that run() returns an integer exit code."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, is_=int)
        tm.that(result, gte=0)

    def test_execute_method_returns_flext_result(self, tmp_path: Path) -> None:
        """Test execute() method returns r[int]."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.execute()
        tm.ok(result)
        tm.that(result.value, is_=int)

    def test_generate_from_sibling_files(self, tmp_path: Path) -> None:
        """Test that generator discovers exports from sibling .py files."""
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        # Create sibling .py file with __all__
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)
        init_file = src_dir / "__init__.py"
        tm.that(init_file.exists(), eq=True)
        content = init_file.read_text()
        tm.that(content, contains="TestModel")
        tm.that(content, contains="test_pkg.models")

    def test_generate_bottom_up(self, tmp_path: Path) -> None:
        """Test that subdirectory exports bubble up to parent."""
        src_dir = tmp_path / "src" / "pkg"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir(parents=True)
        # Subdirectory has a module
        (sub_dir / "service.py").write_text(
            '"""Service."""\n\n__all__ = ["SubService"]\n\nclass SubService:\n    pass\n',
        )
        # Parent has its own module
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["PkgModel"]\n\nclass PkgModel:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)
        # Child __init__.py should have SubService
        child_init = sub_dir / "__init__.py"
        tm.that(child_init.exists(), eq=True)
        tm.that(child_init.read_text(), contains="SubService")
        # Parent __init__.py should have both
        parent_init = src_dir / "__init__.py"
        tm.that(parent_init.exists(), eq=True)
        parent_content = parent_init.read_text()
        tm.that(parent_content, contains="PkgModel")
        tm.that(parent_content, contains="SubService")

    def test_generate_preserves_existing_docstring(self, tmp_path: Path) -> None:
        """Test that existing docstring is preserved in regenerated file."""
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        # Create existing __init__.py with docstring
        (src_dir / "__init__.py").write_text(
            '"""My custom package docstring."""\n\n__all__ = []\n',
        )
        # Create sibling .py file
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["Foo"]\n\nclass Foo:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        generator.run(check_only=False)
        content = (src_dir / "__init__.py").read_text()
        tm.that(content, contains="My custom package docstring")
