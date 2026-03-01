"""Integration test for the end-to-end codegen pipeline.

Validates workspace-level behavior for census, scaffold, auto-fix, lazy-init,
and post-census checks using isolated temporary projects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from flext_infra.codegen.census import FlextInfraCodegenCensus
from flext_infra.codegen.fixer import FlextInfraCodegenFixer
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder

_SRC_MODULES = (
    "constants.py",
    "typings.py",
    "protocols.py",
    "models.py",
    "utilities.py",
)


class _CompatCoreConstants:
    EXEMPT_FILENAMES: ClassVar[frozenset[str]] = frozenset({
        "__init__.py",
        "conftest.py",
        "__main__.py",
    })
    EXEMPT_PREFIXES: ClassVar[tuple[str, ...]] = ("test_", "_")
    DUNDER_ALLOWED: ClassVar[frozenset[str]] = frozenset({
        "__all__",
        "__version__",
        "__version_info__",
    })
    ALIAS_NAMES: ClassVar[frozenset[str]] = frozenset({
        "c",
        "m",
        "p",
        "t",
        "u",
        "r",
        "d",
        "e",
        "h",
        "s",
        "x",
    })
    TYPEVAR_CALLABLES: ClassVar[frozenset[str]] = frozenset({
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
    })
    COLLECTION_CALLS: ClassVar[frozenset[str]] = frozenset({
        "dict",
        "list",
        "set",
        "tuple",
        "frozenset",
    })
    ENUM_BASES: ClassVar[frozenset[str]] = frozenset({"Enum", "StrEnum", "IntEnum"})


def _project_prefix(package_name: str) -> str:
    return "".join(part.title() for part in package_name.split("_"))


def _write_complete_modules(pkg_dir: Path, package_name: str) -> None:
    prefix = _project_prefix(package_name)
    for module_name in _SRC_MODULES:
        suffix = module_name.split(".")[0].title()
        _ = (pkg_dir / module_name).write_text(
            f"class {prefix}{suffix}:\n    pass\n",
            encoding="utf-8",
        )


def _write_package_init(pkg_dir: Path, package_name: str) -> None:
    prefix = _project_prefix(package_name)
    _ = (pkg_dir / "__init__.py").write_text(
        "\n".join([
            '"""Package init for pipeline test."""',
            "",
            f"from {package_name}.constants import {prefix}Constants",
            "",
            "__all__ = [",
            f'    "{prefix}Constants",',
            "]",
            "",
        ]),
        encoding="utf-8",
    )


def _make_project(
    tmp_path: Path,
    name: str,
    *,
    with_all_modules: bool,
    with_tests_dir: bool,
) -> Path:
    project = tmp_path / name
    project.mkdir()
    _ = (project / "Makefile").touch()
    _ = (project / "pyproject.toml").write_text(
        f"[project]\nname='{name}'\n", encoding="utf-8"
    )
    (project / ".git").mkdir()

    package_name = name.replace("-", "_")
    pkg_dir = project / "src" / package_name
    pkg_dir.mkdir(parents=True)
    _write_package_init(pkg_dir, package_name)

    if with_all_modules:
        _write_complete_modules(pkg_dir, package_name)

    if with_tests_dir:
        tests_dir = project / "tests"
        tests_dir.mkdir()
        prefix = _project_prefix(package_name)
        _ = (tests_dir / "__init__.py").write_text(
            "\n".join([
                '"""Tests init for pipeline test."""',
                "",
                f"from tests.constants import Tests{prefix}Constants",
                "",
                "__all__ = [",
                f'    "Tests{prefix}Constants",',
                "]",
                "",
            ]),
            encoding="utf-8",
        )

    return project


def test_codegen_pipeline_end_to_end(tmp_path: Path) -> None:
    """Pipeline flow remains isolated, idempotent, and syntactically valid."""
    _ = _make_project(
        tmp_path,
        "project-a",
        with_all_modules=True,
        with_tests_dir=False,
    )
    project_b = _make_project(
        tmp_path,
        "project-b",
        with_all_modules=True,
        with_tests_dir=False,
    )
    _ = _make_project(
        tmp_path,
        "project-c",
        with_all_modules=True,
        with_tests_dir=True,
    )
    flexcore = _make_project(
        tmp_path,
        "flexcore",
        with_all_modules=False,
        with_tests_dir=True,
    )

    package_b = project_b / "src" / "project_b"
    (package_b / "models.py").unlink()
    _ = (package_b / "base.py").write_text(
        'from typing import TypeVar\n\nTBase = TypeVar("TBase")\n\nclass ProjectBBase:\n    pass\n',
        encoding="utf-8",
    )

    flexcore_package = flexcore / "src" / "flexcore"
    assert not (flexcore_package / "constants.py").exists()

    census_service = FlextInfraCodegenCensus(workspace_root=tmp_path)
    scaffolder = FlextInfraCodegenScaffolder(workspace_root=tmp_path)
    fixer = FlextInfraCodegenFixer(workspace_root=tmp_path)
    lazy_init = FlextInfraCodegenLazyInit(workspace_root=tmp_path)

    with (
        patch(
            "flext_infra.codegen.transforms.FlextInfraCodegenTransforms.run_ruff_fix"
        ),
        patch(
            "flext_infra.core.namespace_validator.c.Infra.Core",
            _CompatCoreConstants,
            create=True,
        ),
    ):
        census_before = census_service.run()

        scaffold_results_first = scaffolder.run()
        scaffold_by_project_first = {
            result.project: result for result in scaffold_results_first
        }

        assert set(scaffold_by_project_first) == {"project-a", "project-b", "project-c"}
        assert len(scaffold_by_project_first["project-a"].files_created) == 0
        assert len(scaffold_by_project_first["project-b"].files_created) == 1
        assert len(scaffold_by_project_first["project-c"].files_created) == 5

        scaffold_results_second = scaffolder.run()
        scaffold_by_project_second = {
            result.project: result for result in scaffold_results_second
        }
        assert len(scaffold_by_project_second["project-a"].files_created) == 0
        assert len(scaffold_by_project_second["project-b"].files_created) == 0
        assert len(scaffold_by_project_second["project-c"].files_created) == 0

        fix_results = fixer.run()
        fix_by_project = {result.project: result for result in fix_results}

        assert set(fix_by_project) == {"project-a", "project-b", "project-c"}
        project_b_fixed = fix_by_project["project-b"]
        assert len(project_b_fixed.violations_fixed) > 0
        assert any(v.rule == "NS-002" for v in project_b_fixed.violations_fixed)

        unmapped_count = lazy_init.run(scan_tests=True)
        assert unmapped_count >= 0

        census_after = census_service.run()

    before_total = sum(report.total for report in census_before)
    after_total = sum(report.total for report in census_after)
    assert after_total <= before_total

    for py_file in tmp_path.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)

    assert not (flexcore_package / "constants.py").exists()


__all__: list[str] = []
