"""Tests for FlextInfraCodegenFixer workspace-level operations.

Validates project exclusion, empty src handling, and files_modified tracking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.codegen.fixer import FlextInfraCodegenFixer
from flext_tests import tm


def _to_pascal(snake: str) -> str:
    return "".join(part.title() for part in snake.split("_"))


def _create_project(
    tmp_path: Path,
    name: str,
    pkg_name: str,
    files: dict[str, str],
) -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text(f"[project]\nname='{name}'\n")
    (project / ".git").mkdir()
    pkg = project / "src" / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (pkg / "typings.py").write_text(
        f"from flext_core import FlextTypes\n"
        f"class {_to_pascal(pkg_name)}Types(FlextTypes):\n    pass\n",
    )
    (pkg / "constants.py").write_text(
        f"from flext_core import FlextConstants\n"
        f"class {_to_pascal(pkg_name)}Constants(FlextConstants):\n    pass\n",
    )
    for filename, content in files.items():
        (pkg / filename).write_text(content)
    return project


def test_flexcore_excluded_from_run(tmp_path: Path) -> None:
    flexcore = tmp_path / "flexcore"
    flexcore.mkdir()
    (flexcore / "Makefile").touch()
    (flexcore / "pyproject.toml").write_text("[project]\nname='flexcore'\n")
    (flexcore / ".git").mkdir()
    pkg = flexcore / "src" / "flexcore"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (pkg / "typings.py").write_text("pass\n")
    (pkg / "constants.py").write_text("pass\n")
    (pkg / "base.py").write_text("import typing\nT = typing.TypeVar('T')\n")
    _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": "import typing\nT = typing.TypeVar('T')\n"
            "class TestProjBase:\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    results = fixer.run()
    project_names = [res.project for res in results]
    tm.that("flexcore" not in project_names, eq=True)
    tm.that("test-proj" in project_names, eq=True)


def test_project_without_src_returns_empty(tmp_path: Path) -> None:
    project = tmp_path / "no-src-proj"
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text("[project]\nname='no-src-proj'\n")
    (project / ".git").mkdir()
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    tm.that(result.project, eq="no-src-proj")
    tm.that(result.violations_fixed, eq=[])
    tm.that(result.violations_skipped, eq=[])
    tm.that(result.files_modified, eq=[])


def test_files_modified_tracks_affected_files(tmp_path: Path) -> None:
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": "from typing import Final\nMAX_RETRIES: Final = 3\n"
            "class TestProjBase:\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    tm.that(len(result.files_modified), eq=2)
    modified_str = " ".join(result.files_modified)
    tm.that(modified_str, contains="base.py")
    tm.that(modified_str, contains="constants.py")


__all__: list[str] = []
