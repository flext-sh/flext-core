"""Tests for FlextInfraCodegenLazyInit._process_directory integration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.codegen import FlextInfraCodegenLazyInit
from flext_tests import tm


class TestProcessDirectory:
    """Test the _process_directory method (integration-level)."""

    def test_generates_init_from_sibling_files(self, tmp_path: Path) -> None:
        """Test _process_directory generates __init__.py from siblings."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="TestModel")
        init_content = (src_dir / "__init__.py").read_text()
        tm.that(init_content, contains="TestModel")

    def test_check_only_does_not_write(self, tmp_path: Path) -> None:
        """Test _process_directory in check_only mode doesn't write files."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            src_dir,
            check_only=True,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="TestModel")
        # __init__.py should NOT have been created
        tm.that((src_dir / "__init__.py").exists(), eq=False)

    def test_skips_directory_without_package(self, tmp_path: Path) -> None:
        """Test _process_directory skips dirs that can't infer package."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        random_dir = tmp_path / "random"
        random_dir.mkdir()
        (random_dir / "models.py").write_text("class Model: pass\n")
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            random_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=None)
        tm.that(exports, eq={})

    def test_includes_child_exports(self, tmp_path: Path) -> None:
        """Test _process_directory includes child subdirectory exports."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "pkg"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["ParentModel"]\n\nclass ParentModel:\n    pass\n',
        )
        dir_exports = {
            str(sub_dir): {
                "ChildService": ("pkg.sub.service", "ChildService"),
            },
        }
        result, exports = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="ParentModel")
        tm.that(exports, contains="ChildService")

    def test_handles_version_file(self, tmp_path: Path) -> None:
        """Test _process_directory handles __version__.py correctly."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["Model"]\n\nclass Model:\n    pass\n',
        )
        (src_dir / "__version__.py").write_text(
            '__version__ = "1.0.0"\n__version_info__ = (1, 0, 0)\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, _ = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        content = (src_dir / "__init__.py").read_text()
        tm.that(content, contains='__version__ = "1.0.0"')
        tm.that(content, contains="__version_info__")
