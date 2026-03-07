"""Refactor/CST helper utilities for infrastructure code analysis.

Centralizes CST (Concrete Syntax Tree) helpers previously defined as
module-level functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import c, m, t
from flext_infra.discovery import FlextInfraDiscoveryService


class FlextInfraRefactorUtilities:
    """CST/refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        name = u.Infra.Refactor.dotted_name(cst_expr)
    """

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
            root = FlextInfraRefactorUtilities.dotted_name(expr.value)
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
            return FlextInfraRefactorUtilities.root_name(expr.value)
        if isinstance(expr, cst.Call):
            return FlextInfraRefactorUtilities.root_name(expr.func)
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
        """Discover project roots under a workspace.

        Uses ``FlextInfraDiscoveryService`` and falls back to the
        workspace root itself when it contains a ``src/`` directory.

        Args:
            workspace_root: Top-level workspace directory.

        Returns:
            List of resolved project root paths.

        """
        discovery = FlextInfraDiscoveryService()
        projects = discovery.discover_projects(workspace_root)
        roots: list[Path] = []
        if projects.is_success:
            roots = [project.path for project in projects.unwrap()]
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
    ) -> list[Path]:
        """Iterate Python files across all projects in a workspace.

        Args:
            workspace_root: Top-level workspace directory.
            include_tests: Whether to include files under ``tests/``.

        Returns:
            Sorted list of ``.py`` file paths.

        """
        roots = FlextInfraRefactorUtilities.discover_project_roots(
            workspace_root=workspace_root,
        )
        files: list[Path] = []
        for project_root in roots:
            src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
            if src_dir.is_dir():
                files.extend(sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
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
        """Infer refactor policy family from a file path."""
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
    def policy_for_symbol(
        *,
        symbol_name: str,
        policy_context: t.Infra.PolicyContext,
        class_families: t.Infra.ClassFamilyMap,
    ) -> m.Infra.Refactor.ClassNestingPolicy:
        """Resolve strict family policy for a symbol."""
        family = class_families.get(symbol_name)
        if family is None:
            msg = f"missing class family mapping for symbol: {symbol_name}"
            raise ValueError(msg)
        policy = policy_context.get(family)
        if policy is None:
            msg = f"missing policy for family: {family}"
            raise ValueError(msg)
        try:
            return TypeAdapter(m.Infra.Refactor.ClassNestingPolicy).validate_python(
                policy
            )
        except ValidationError:
            msg = f"invalid policy type for family: {family}"
            raise ValueError(msg) from None

    @staticmethod
    def policy_bool(*, policy: m.Infra.Refactor.ClassNestingPolicy, key: str) -> bool:
        """Fetch strict boolean policy value."""
        raw = getattr(policy, key, None)
        if isinstance(raw, bool):
            return raw
        msg = f"policy key {key!r} must be bool"
        raise ValueError(msg)

    @staticmethod
    def policy_string_collection(
        *, policy: m.Infra.Refactor.ClassNestingPolicy, key: str
    ) -> tuple[str, ...]:
        """Fetch a strict string collection policy value."""
        raw = getattr(policy, key, None)
        if isinstance(raw, tuple):
            try:
                return TypeAdapter(tuple[str, ...]).validate_python(raw)
            except ValidationError:
                msg = f"policy key {key!r} must contain only strings"
                raise ValueError(msg) from None
        if isinstance(raw, list):
            try:
                items = TypeAdapter(list[str]).validate_python(raw)
            except ValidationError:
                msg = f"policy key {key!r} must contain only strings"
                raise ValueError(msg) from None
            return tuple(items)
        msg = f"policy key {key!r} must be list[str] or tuple[str, ...]"
        raise ValueError(msg)

    @staticmethod
    def target_allowed(
        *, policy: m.Infra.Refactor.ClassNestingPolicy, target_namespace: str
    ) -> bool:
        """Validate target namespace against allowed/forbidden policy lists."""
        allowed = FlextInfraRefactorUtilities.policy_string_collection(
            policy=policy,
            key="allowed_targets",
        )
        if allowed and target_namespace not in allowed:
            return False
        forbidden = FlextInfraRefactorUtilities.policy_string_collection(
            policy=policy,
            key="forbidden_targets",
        )
        return target_namespace not in forbidden


__all__ = [
    "FlextInfraRefactorUtilities",
]
