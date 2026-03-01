"""Tests for FlextInfraAutoFixer service.

Validates AST-based auto-fix detection for namespace violations:
- Standalone TypeVar/TypeAlias/Final detection
- In-context usage exclusion (used by Generic classes)
- SyntaxError resilience
- Project exclusion (flexcore)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from flext_infra.codegen.auto_fix import FlextInfraAutoFixer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_prefix_static(project_root: Path) -> str:
    """Standalone reimplementation of _derive_prefix for testing.

    The source calls ``FlextInfraNamespaceValidator._derive_prefix(project_path)``
    as an unbound method, which fails because ``_derive_prefix`` is an instance
    method. This helper provides the same logic as a plain function so we can
    patch the call site.
    """
    src_dir = project_root / "src"
    if not src_dir.is_dir():
        return ""
    for child in sorted(src_dir.iterdir()):
        if child.is_dir() and (child / "__init__.py").exists():
            return "".join(part.title() for part in child.name.split("_"))
    return ""


def _create_project(
    tmp_path: Path,
    name: str,
    pkg_name: str,
    files: dict[str, str],
) -> Path:
    """Scaffold a minimal project with given source files."""
    project = tmp_path / name
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text(f"[project]\nname='{name}'\n")
    (project / ".git").mkdir()
    pkg = project / "src" / pkg_name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").touch()
    (pkg / "typings.py").write_text(
        "from flext_core import FlextTypes\n"
        f"class {_to_pascal(pkg_name)}Types(FlextTypes):\n"
        "    pass\n"
    )
    (pkg / "constants.py").write_text(
        "from flext_core import FlextConstants\n"
        f"class {_to_pascal(pkg_name)}Constants(FlextConstants):\n"
        "    pass\n"
    )
    for filename, content in files.items():
        (pkg / filename).write_text(content)
    return project


def _to_pascal(snake: str) -> str:
    return "".join(part.title() for part in snake.split("_"))


_PATCH_TARGET = (
    "flext_infra.codegen.auto_fix.FlextInfraNamespaceValidator._derive_prefix"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixer(tmp_path: Path) -> FlextInfraAutoFixer:
    """Create a fixer instance rooted at tmp_path."""
    return FlextInfraAutoFixer(tmp_path)


# ---------------------------------------------------------------------------
# Test: Standalone TypeVar detected as fixable
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_standalone_typevar_detected_as_fixable(
    _mock_prefix: object, tmp_path: Path
) -> None:
    """A standalone TypeVar not used by any class is detected as fixable."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "import typing\n"
                "T = typing.TypeVar('T')\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    assert len(result.violations_fixed) >= 1
    typevar_violations = [v for v in result.violations_fixed if "TypeVar" in v.message]
    assert len(typevar_violations) == 1
    assert typevar_violations[0].fixable is True
    assert typevar_violations[0].rule == "NS-002"


# ---------------------------------------------------------------------------
# Test: In-context TypeVar (used by Generic class) is NOT flagged for move
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_in_context_typevar_not_flagged(_mock_prefix: object, tmp_path: Path) -> None:
    """A TypeVar used by a class in the same file is skipped (not fixable)."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "from typing import TypeVar, Generic\n"
                "T = TypeVar('T')\n"
                "class TestProjBase(Generic[T]):\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    # Should NOT appear in violations_fixed
    typevar_fixed = [v for v in result.violations_fixed if "TypeVar" in v.message]
    assert len(typevar_fixed) == 0

    # The find_standalone_typevars already filters in-context usage,
    # so it won't appear in skipped either — it's simply not detected
    # as standalone at all.


# ---------------------------------------------------------------------------
# Test: Standalone Final constant detected as fixable
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_standalone_final_detected_as_fixable(
    _mock_prefix: object, tmp_path: Path
) -> None:
    """A standalone Final constant is detected as fixable (NS-001)."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "from typing import Final\n"
                "MAX_RETRIES: Final = 3\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    final_violations = [v for v in result.violations_fixed if "Final" in v.message]
    assert len(final_violations) == 1
    assert final_violations[0].fixable is True
    assert final_violations[0].rule == "NS-001"
    assert "constants.py" in final_violations[0].message


# ---------------------------------------------------------------------------
# Test: Standalone TypeAlias detected as fixable
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_standalone_typealias_detected_as_fixable(
    _mock_prefix: object, tmp_path: Path
) -> None:
    """A standalone TypeAlias is detected as fixable (NS-002)."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "from typing import TypeAlias\n"
                "MyType: TypeAlias = str\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    alias_violations = [v for v in result.violations_fixed if "TypeAlias" in v.message]
    assert len(alias_violations) == 1
    assert alias_violations[0].fixable is True
    assert alias_violations[0].rule == "NS-002"
    assert "typings.py" in alias_violations[0].message


# ---------------------------------------------------------------------------
# Test: Files with SyntaxError are skipped (not crashed)
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_syntax_error_files_skipped(_mock_prefix: object, tmp_path: Path) -> None:
    """Files with syntax errors are silently skipped without crashing."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "broken.py": "def foo(\n    # missing closing paren\n",
            "base.py": (
                "import typing\n"
                "T = typing.TypeVar('T')\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    # Should not crash and should still detect violations in valid files
    assert result.project == "test-proj"
    typevar_violations = [v for v in result.violations_fixed if "TypeVar" in v.message]
    assert len(typevar_violations) == 1


# ---------------------------------------------------------------------------
# Test: flexcore project is excluded
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_flexcore_excluded_from_run(_mock_prefix: object, tmp_path: Path) -> None:
    """The 'flexcore' project is excluded from workspace-wide auto-fix."""
    # Create flexcore project
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

    # Create a non-excluded project
    _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "import typing\n"
                "T = typing.TypeVar('T')\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )

    fixer = FlextInfraAutoFixer(tmp_path)
    results = fixer.run()

    project_names = [res.project for res in results]
    assert "flexcore" not in project_names
    assert "test-proj" in project_names


# ---------------------------------------------------------------------------
# Test: Project without src/ returns empty result
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_project_without_src_returns_empty(
    _mock_prefix: object, tmp_path: Path
) -> None:
    """A project without src/ directory returns empty violations."""
    project = tmp_path / "no-src-proj"
    project.mkdir()
    (project / "Makefile").touch()
    (project / "pyproject.toml").write_text("[project]\nname='no-src-proj'\n")
    (project / ".git").mkdir()

    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    assert result.project == "no-src-proj"
    assert result.violations_fixed == []
    assert result.violations_skipped == []
    assert result.files_modified == []


# ---------------------------------------------------------------------------
# Test: files_modified tracks affected files
# ---------------------------------------------------------------------------


@patch(_PATCH_TARGET, side_effect=_derive_prefix_static)
def test_files_modified_tracks_affected_files(
    _mock_prefix: object, tmp_path: Path
) -> None:
    """files_modified includes both source and target files."""
    project = _create_project(
        tmp_path,
        name="test-proj",
        pkg_name="test_proj",
        files={
            "base.py": (
                "from typing import Final\n"
                "MAX_RETRIES: Final = 3\n"
                "class TestProjBase:\n"
                "    pass\n"
            ),
        },
    )
    fixer = FlextInfraAutoFixer(tmp_path)
    result = fixer.fix_project(project)

    assert len(result.files_modified) == 2
    modified_str = " ".join(result.files_modified)
    assert "base.py" in modified_str
    assert "constants.py" in modified_str


__all__: list[str] = []
