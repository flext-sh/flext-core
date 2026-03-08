"""Refactor/CST helper utilities for infrastructure code analysis.

Centralizes CST (Concrete Syntax Tree) helpers previously defined as
module-level functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, t
from flext_infra.subprocess import FlextInfraCommandRunner


class FlextInfraUtilitiesRefactor:
    """CST/refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        name = u.Infra.Refactor.dotted_name(cst_expr)
    """

    @staticmethod
    def capture_output(cmd: Sequence[str]) -> r[str]:
        """Run *cmd* and return stripped stdout as ``r[str]``."""
        return FlextInfraCommandRunner().capture(cmd)

    @staticmethod
    def dotted_name(expr: cst.BaseExpression) -> str:
        """Extract dotted name string from a CST expression.

        Recursively traverses ``Name`` and ``Attribute`` nodes to build
        a fully qualified dotted name (e.g., ``"foo.bar.baz"``).

        Args:
            expr: CST expression node.

        Returns:
            Dotted name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            root = FlextInfraUtilitiesRefactor.dotted_name(expr.value)
            if not root:
                return ""
            return f"{root}.{expr.attr.value}"
        return ""

    @staticmethod
    def root_name(expr: cst.BaseExpression) -> str:
        """Extract the root (leftmost) name from a CST expression.

        Handles ``Name``, ``Attribute``, and ``Call`` nodes to find the
        base identifier.

        Args:
            expr: CST expression node.

        Returns:
            Root name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return FlextInfraUtilitiesRefactor.root_name(expr.value)
        if isinstance(expr, cst.Call):
            return FlextInfraUtilitiesRefactor.root_name(expr.func)
        return ""

    @staticmethod
    def asname_to_local(asname: cst.AsName | None) -> str | None:
        """Extract the local alias name from a CST AsName node.

        Args:
            asname: CST AsName node, or None.

        Returns:
            Local alias string, or None if no alias or unsupported node.

        """
        if asname is None:
            return None
        if isinstance(asname.name, cst.Name):
            return asname.name.value
        return None

    @staticmethod
    def discover_project_roots(*, workspace_root: Path) -> list[Path]:
        """Discover project roots under a workspace."""
        roots: list[Path] = []

        def _looks_like_project(path: Path) -> bool:
            if not path.is_dir():
                return False
            if not (path / c.Infra.Files.MAKEFILE_FILENAME).exists():
                return False
            has_pyproject = (path / c.Infra.Files.PYPROJECT_FILENAME).exists()
            has_gomod = (path / c.Infra.Files.GO_MOD).exists()
            if not has_pyproject and (not has_gomod):
                return False
            has_scan_dir = any(
                (path / dir_name).is_dir()
                for dir_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES
            )
            return has_scan_dir

        if _looks_like_project(workspace_root):
            roots.append(workspace_root)

        for entry in sorted(workspace_root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not _looks_like_project(entry):
                continue
            roots.append(entry)
        if (
            len(roots) == 0
            and (workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir()
        ):
            roots = [workspace_root]
        return roots

    @staticmethod
    def iter_python_files(
        *,
        workspace_root: Path,
        include_tests: bool = True,
        include_examples: bool = True,
        include_scripts: bool = True,
    ) -> list[Path]:
        """Iterate Python files across all projects in a workspace.

        Args:
            workspace_root: Top-level workspace directory.
            include_tests: Whether to include files under ``tests/``.

        Returns:
            Sorted list of ``.py`` file paths.

        """
        roots = FlextInfraUtilitiesRefactor.discover_project_roots(
            workspace_root=workspace_root
        )
        files: list[Path] = []
        for project_root in roots:
            src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
            if src_dir.is_dir():
                files.extend(sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
            if include_examples:
                examples_dir = project_root / c.Infra.Directories.EXAMPLES
                if examples_dir.is_dir():
                    files.extend(
                        sorted(examples_dir.rglob(c.Infra.Extensions.PYTHON_GLOB))
                    )
            if include_scripts:
                scripts_dir = project_root / c.Infra.Directories.SCRIPTS
                if scripts_dir.is_dir():
                    files.extend(
                        sorted(scripts_dir.rglob(c.Infra.Extensions.PYTHON_GLOB))
                    )
            if include_tests:
                tests_dir = project_root / c.Infra.Directories.TESTS
                if tests_dir.is_dir():
                    files.extend(
                        sorted(tests_dir.rglob(c.Infra.Extensions.PYTHON_GLOB))
                    )
        return files

    @staticmethod
    def module_path(*, file_path: Path, project_root: Path) -> str:
        """Compute dotted module path relative to a project root.

        Strips the ``src/`` directory component and file extension.

        Args:
            file_path: Absolute path to a Python file.
            project_root: Root directory of the project.

        Returns:
            Dotted module path (e.g., ``"flext_infra.refactor.engine"``).

        """
        rel = file_path.relative_to(project_root)
        parts = [part for part in rel.with_suffix("").parts if part != "src"]
        return ".".join(parts)

    @staticmethod
    def module_family_from_path(path: str) -> str:
        """Resolve module family key from a source file path."""
        normalized = path.replace("\\", "/")
        if "_models" in normalized:
            return "_models"
        if "_utilities" in normalized:
            return "_utilities"
        if "_dispatcher" in normalized:
            return "_dispatcher"
        if "_decorators" in normalized:
            return "_decorators"
        if "_runtime" in normalized:
            return "_runtime"
        return "other_private"

    @staticmethod
    def entry_list(value: t.Infra.InfraValue | None) -> list[dict[str, str]]:
        """Normalize class-nesting config entries to a strict list."""
        if value is None:
            return []
        try:
            return TypeAdapter(list[dict[str, str]]).validate_python(value)
        except ValidationError:
            msg = "class nesting entries must be a list"
            raise ValueError(msg) from None

    @staticmethod
    def string_list(value: t.ContainerValue | None) -> list[str]:
        """Normalize policy fields that should contain string collections."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    msg = "expected list[str] value"
                    raise TypeError(msg)
                items.append(item)
            return items
        msg = "expected list[str] value"
        raise ValueError(msg)

    @staticmethod
    def mapping_list(
        value: t.ContainerValue | None,
    ) -> list[dict[str, t.ContainerValue]]:
        """Normalize policy fields that should contain mapping collections."""
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        msg = "expected list[dict[str, ContainerValue]] value"
        raise ValueError(msg)


__all__ = ["FlextInfraUtilitiesRefactor"]
