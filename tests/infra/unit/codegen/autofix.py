"""Tests for FlextInfraCodegenFixer service.

Validates AST-based auto-fix detection for namespace violations:
- Standalone TypeVar/TypeAlias/Final detection
- In-context usage exclusion (used by Generic classes)
- SyntaxError resilience

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.codegen.fixer import FlextInfraCodegenFixer
from flext_tests import tm


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


def _to_pascal(snake: str) -> str:
    return "".join(part.title() for part in snake.split("_"))


@pytest.fixture
def fixer(tmp_path: Path) -> FlextInfraCodegenFixer:
    return FlextInfraCodegenFixer(tmp_path)


def test_standalone_typevar_detected_as_fixable(tmp_path: Path) -> None:
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": "import typing\nT = typing.TypeVar('T')\n"
            "class TestProjBase:\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    tm.that(len(result.violations_fixed), gte=1)
    typevar_violations = [v for v in result.violations_fixed if "TypeVar" in v.message]
    tm.that(len(typevar_violations), eq=1)
    tm.that(typevar_violations[0].fixable, eq=True)
    tm.that(typevar_violations[0].rule, eq="NS-002")


def test_in_context_typevar_not_flagged(tmp_path: Path) -> None:
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": "from typing import TypeVar, Generic\n"
            "T = TypeVar('T')\nclass TestProjBase(Generic[T]):\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    typevar_fixed = [v for v in result.violations_fixed if "TypeVar" in v.message]
    tm.that(len(typevar_fixed), eq=0)


def test_standalone_final_detected_as_fixable(tmp_path: Path) -> None:
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
    final_violations = [v for v in result.violations_fixed if "Final" in v.message]
    tm.that(len(final_violations), eq=1)
    tm.that(final_violations[0].fixable, eq=True)
    tm.that(final_violations[0].rule, eq="NS-001")
    tm.that(final_violations[0].message, contains="constants.py")


def test_standalone_typealias_detected_as_fixable(tmp_path: Path) -> None:
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": "from typing import TypeAlias\n"
            "MyType: TypeAlias = str\nclass TestProjBase:\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    alias_violations = [v for v in result.violations_fixed if "TypeAlias" in v.message]
    tm.that(len(alias_violations), eq=1)
    tm.that(alias_violations[0].fixable, eq=True)
    tm.that(alias_violations[0].rule, eq="NS-002")
    tm.that(alias_violations[0].message, contains="typings.py")


def test_syntax_error_files_skipped(tmp_path: Path) -> None:
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "broken.py": "def foo(\n    # missing closing paren\n",
            "base.py": "import typing\nT = typing.TypeVar('T')\n"
            "class TestProjBase:\n    pass\n",
        },
    )
    fixer = FlextInfraCodegenFixer(tmp_path)
    result = fixer.fix_project(project)
    tm.that(result.project, eq="test-proj")
    typevar_violations = [v for v in result.violations_fixed if "TypeVar" in v.message]
    tm.that(len(typevar_violations), eq=1)


__all__: list[str] = []
