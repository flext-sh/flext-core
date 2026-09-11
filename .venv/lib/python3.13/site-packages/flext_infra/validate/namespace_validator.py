"""Namespace validation service.

Rope-backed validator enforcing namespace rules 0-3 for flext projects.
Detection-only — does not auto-fix any files. AST nodes are obtained via
``rope.get_pymodule(...).get_ast()`` per the flext-infra detector mandate
(no raw ``ast.parse`` source reads).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra.base import s
from flext_infra.validate.namespace_rules import FlextInfraNamespaceRules

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraNamespaceValidator(s[bool], FlextInfraNamespaceRules):
    """Rope-backed namespace validator for flext projects (Rules 0-3).

    Validates that each module follows the one-namespace-class-per-file
    convention, constants are centralized in ``constants.py``, and type
    definitions are centralized in ``typings.py``.
    """

    @override
    def execute(self) -> p.Result[bool]:
        """Execute namespace validation for the configured repository root."""
        report_result = self.validate_project(self.repository_root)
        if report_result.failure:
            return r[bool].fail(report_result.error or "namespace validation failed")
        report = report_result.unwrap()
        return r[bool].ok(report.passed)

    def validate_project(
        self, project_root: Path
    ) -> p.Result[m.Infra.ValidationReport]:
        """Validate namespace rules inside one project."""
        files_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(project_roots=(project_root,))
        )
        if files_result.failure:
            return r[m.Infra.ValidationReport].fail(
                files_result.error or "Namespace validation failed: discovery failed"
            )
        files = [
            py_file
            for py_file in files_result.value
            if not self._is_exempt_file(py_file)
        ]
        layout = u.Infra.layout(project_root)
        prefix = layout.class_stem if layout is not None else ""
        package_name = (
            layout.package_dir.name
            if layout is not None
            else project_root.name.replace("-", "_")
        )
        violations: t.MutableSequenceOf[str] = []
        with u.Infra.open_project(project_root) as rope_project:
            for filepath in files:
                tree_result = self._parse_file(rope_project, filepath)
                if tree_result.failure:
                    continue
                tree = tree_result.value
                rel = filepath.relative_to(project_root)
                is_test_file = self._is_test_file(rel)
                if self._is_namespace_governed_file(rel):
                    is_facade = self._is_facade_module(rel)
                    violations.extend(
                        self.check_rule_0(
                            tree,
                            rel,
                            prefix,
                            is_test_file=is_test_file,
                            strict_top_level=is_facade,
                            strict_single_class=is_facade,
                            require_public_class=is_facade,
                        )
                    )
                if not is_test_file:
                    violations.extend(self.check_rule_1(tree, rel))
                    violations.extend(self.check_rule_2(tree, rel))
                violations.extend(
                    self.check_rule_3(
                        tree, rel, class_stem=prefix, package_name=package_name
                    )
                )
        return self._validation_report(files=files, violations=violations)

    @staticmethod
    def _validation_report(
        *, files: t.SequenceOf[Path], violations: t.SequenceOf[str]
    ) -> p.Result[m.Infra.ValidationReport]:
        """Build the namespace validation report."""
        passed = not violations
        summary = (
            f"namespace validation passed ({len(files)} files checked)"
            if passed
            else f"{len(violations)} namespace violation(s) found ({len(files)} files checked)"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=tuple(violations), summary=summary
            )
        )

    def _is_exempt_file(self, filepath: Path) -> bool:
        """Check whether a file should be skipped from validation."""
        name = filepath.name
        if name in c.Infra.EXEMPT_FILENAMES:
            return True
        return any(name.startswith(prefix) for prefix in c.Infra.EXEMPT_PREFIXES)

    def _parse_file(
        self, rope_project: t.Infra.RopeProject, path: Path
    ) -> p.Result[ast.AST]:
        """Return the AST module for ``path`` via rope.

        ``r.ok(module)`` on success. ``r.fail(reason)`` when the resource
        cannot be fetched, the module fails to parse, or rope returns no
        ``PyModule``. Callers that want "skip silently" can collapse with
        ``unwrap_or(None)`` or ``.failure``.
        """
        try:
            resource = u.Infra.fetch_python_resource(rope_project, path)
        except c.EXC_OS_SYNTAX as exc:
            return r[ast.AST].fail(f"fetch_python_resource raised: {exc!s}")
        if resource is None:
            return r[ast.AST].fail(f"no rope resource for {path}")
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        except c.EXC_OS_SYNTAX as exc:
            return r[ast.AST].fail(f"get_pymodule raised: {exc!s}")
        ast_module = pymodule.get_ast()
        return r[ast.AST].ok(ast_module)

    @staticmethod
    def _is_namespace_governed_file(rel_path: Path) -> bool:
        """Return whether NS-000/001/002 structural rules apply to this file.

        Governed files are the five canonical facade modules and any module
        inside the corresponding private namespace directories.
        """
        if FlextInfraNamespaceValidator._is_facade_module(rel_path):
            return True
        return any(
            part in {"_constants", "_models", "_protocols", "_typings", "_utilities"}
            for part in rel_path.parts
        )

    @staticmethod
    def _is_facade_module(rel_path: Path) -> bool:
        """Return True for the five root facade modules under ``src/<pkg>/``."""
        if (
            len(rel_path.parts) != c.Infra.FACADE_MODULE_DEPTH
            or rel_path.parts[0] != c.Infra.DEFAULT_SRC_DIR
        ):
            return False
        return rel_path.name in {
            c.Infra.CONSTANTS_PY,
            c.Infra.MODELS_PY,
            c.Infra.PROTOCOLS_PY,
            c.Infra.TYPINGS_PY,
            c.Infra.UTILITIES_PY,
        }

    @staticmethod
    def _is_test_file(rel_path: Path) -> bool:
        """Return True when the file lives under the project's ``tests/`` tree."""
        return any(part == c.Infra.DIR_TESTS for part in rel_path.parts)


__all__: list[str] = ["FlextInfraNamespaceValidator"]
