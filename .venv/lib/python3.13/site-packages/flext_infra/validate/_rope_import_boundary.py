"""FLEXT base for rope-driven module-import boundary validators.

Consolidates the shared skeleton between tier-whitelist and metadata-discipline
validators (and any future rope-import boundary guard): build_report,
_collect_violations, _violations_for_module, _top_module, execute. Subclasses
declare only the banned set, summary strings, and (optionally) override
``_is_allowlisted``/``_is_in_scope``/``_format_violation``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_core import r
from flext_infra import m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraRopeImportBoundaryBase(s[bool]):
    """Base service for rope-driven module-import boundary validators.

    Subclasses set the four ``ClassVar`` declarations below. Optional hooks:
    ``_is_in_scope`` (default: all files), ``_is_allowlisted`` (default: none),
    ``_format_violation`` (default: ``"{path}: {module}"``).
    """

    _BANNED: ClassVar[frozenset[str]] = frozenset()
    _OK_SUMMARY: ClassVar[str] = ""
    _VIOLATION_KIND: ClassVar[str] = ""
    _SCAN_KIND: ClassVar[str] = ""

    def build_report(self, repository_root: Path) -> p.Result[m.Infra.ValidationReport]:
        """Scan ``repository_root`` and return a ``ValidationReport``."""
        try:
            violations = self._collect_violations(repository_root)
        except OSError as exc:
            return r[m.Infra.ValidationReport].fail(
                f"{self._SCAN_KIND} scan failed: {exc}"
            )
        passed = not violations
        summary = (
            self._OK_SUMMARY
            if passed
            else f"{len(violations)} {self._VIOLATION_KIND} violation(s)"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=list(violations), summary=summary
            )
        )

    def _collect_violations(self, repository_root: Path) -> t.StrSequence:
        """Traverse the rope project and accumulate boundary violations."""
        violations: t.MutableSequenceOf[str] = []
        with u.Infra.open_project(repository_root) as project:
            for resource in u.Infra.python_resources(project):
                file_path = u.Infra.resource_file_path(project, resource)
                if file_path is None or not self._is_in_scope(file_path):
                    continue
                module_imports = u.Infra.get_module_imports(project, resource)
                if module_imports is None:
                    continue
                violations.extend(
                    self._violations_for_module(file_path, module_imports)
                )
        return tuple(violations)

    def _is_in_scope(self, _file_path: Path) -> bool:
        """Default: every traversed module is in scope. Override to narrow."""
        return True

    def _is_allowlisted(self, _file_path: Path, _module_name: str) -> bool:
        """Per (file, module) allowlist check. Override to exempt canonical owners."""
        return False

    def _violations_for_module(
        self, file_path: Path, module_imports: t.Infra.RopeModuleImports
    ) -> t.StrSequence:
        """Return banned-import violation strings for one module."""
        out: t.MutableSequenceOf[str] = []
        for stmt in u.Infra.import_statements(module_imports):
            module_name = u.Infra.import_statement_module_name(stmt)
            if module_name is not None:
                if self._top_module(
                    module_name
                ) in self._BANNED and not self._is_allowlisted(file_path, module_name):
                    out.append(self._format_violation(file_path, module_name))
                continue
            for imported, _alias in u.Infra.import_statement_names_and_aliases(stmt):
                if self._top_module(
                    imported
                ) in self._BANNED and not self._is_allowlisted(file_path, imported):
                    out.append(self._format_violation(file_path, imported))
        return tuple(out)

    def _format_violation(self, file_path: Path, module_name: str) -> str:
        """Default: minimal ``path: module`` line. Override for richer messaging."""
        return f"{file_path}: {module_name}"

    @staticmethod
    def _top_module(module_name: str | None) -> str:
        """Return the top-level package name (``a.b.c`` → ``a``)."""
        if not module_name:
            return ""
        return module_name.split(".", maxsplit=1)[0]

    @override
    def execute(self) -> p.Result[bool]:
        """Run the validation against ``self.repository_root``."""
        report_result = self.build_report(self.repository_root)
        if report_result.failure:
            return r[bool].fail(
                report_result.error or f"{self._VIOLATION_KIND} validation failed"
            )
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)


__all__: t.StrSequence = ("FlextInfraRopeImportBoundaryBase",)
