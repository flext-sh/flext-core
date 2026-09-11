"""Runtime Beartype census validator.

Imports every ``flext_*`` module in selected projects and runs
``u.check()`` against every locally-defined class.
Aggregates violations by rule/project into the standard validation report.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, config, m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraRuntimeCensusValidator(s[bool]):
    """Post-import runtime enforcement census across workspace projects."""

    project_filter: Annotated[
        str | None, m.Field(description="Project filter (comma-separated)")
    ] = None

    def _selected_projects(
        self, projects: t.SequenceOf[p.Infra.ProjectInfo]
    ) -> t.SequenceOf[p.Infra.ProjectInfo]:
        """Apply comma-separated project filter when provided."""
        if self.project_filter is None:
            return projects
        selected = {
            item.strip() for item in self.project_filter.split(",") if item.strip()
        }
        return [project for project in projects if project.name in selected]

    @staticmethod
    def _package_name_for_project(project: p.Infra.ProjectInfo) -> str | None:
        """Resolve the importable package name for a project root."""
        layout = u.Infra.layout(project.path, project=project)
        if layout is not None:
            package_name: str = layout.package_name
            return package_name
        src_dir = project.path / c.Infra.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return None
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / c.Infra.INIT_PY).is_file():
                child_name: str = child.name
                return child_name
        return None

    @staticmethod
    def _is_local_class(klass: type, module_name: str) -> bool:
        """Return True when ``klass`` is defined in ``module_name`` (not imported)."""
        return getattr(klass, "__module__", "") == module_name

    @classmethod
    def _walk_modules(cls, package_name: str) -> t.SequenceOf[str]:
        """Return all importable module names under ``package_name``."""
        try:
            package = importlib.import_module(package_name)
        except c.EXC_BROAD_IO_TYPE as exc:
            return [f"{package_name}: import failed: {exc}"]
        prefix = package.__name__ + "."
        modules: list[str] = [package.__name__]
        try:
            for _, modname, _ in pkgutil.walk_packages(
                package.__path__, prefix=prefix, onerror=lambda _name: None
            ):
                modules.append(modname)
        except c.EXC_BROAD_IO_TYPE as exc:
            modules.append(f"{package_name}: walk_packages failed: {exc}")
        return modules

    def _check_module(self, module_name: str) -> t.SequenceOf[m.Infra.ValidationReport]:
        """Import one module and run runtime enforcement on its local classes."""
        try:
            module = importlib.import_module(module_name)
        except c.EXC_BROAD_IO_TYPE as exc:
            return [
                m.Infra.ValidationReport(
                    passed=False,
                    violations=(f"{module_name}: import failed: {exc}",),
                    summary=f"{module_name}: import failed",
                )
            ]
        violations: list[str] = []
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not self._is_local_class(obj, module.__name__):
                continue
            try:
                report = u.check(obj)
            except c.EXC_BROAD_IO_TYPE as exc:
                violations.append(
                    f"{module_name}:{obj.__qualname__}: check raised: {exc}"
                )
                continue
            for violation in report.violations:
                file_part = f"{violation.file_path}:" if violation.file_path else ""
                line_part = f"{violation.line_number}:" if violation.line_number else ""
                rule_part = f" [{violation.rule_id}]" if violation.rule_id else ""
                violations.append(
                    f"{file_part}{line_part}{obj.__qualname__}{rule_part}: "
                    f"{violation.message}"
                )
        return [
            m.Infra.ValidationReport(
                passed=not violations,
                violations=tuple(violations),
                summary=(
                    f"{module_name}: {len(violations)} runtime violation(s)"
                    if violations
                    else f"{module_name}: clean"
                ),
            )
        ]

    def _project_report(self, project: p.Infra.ProjectInfo) -> m.Infra.ValidationReport:
        """Run the runtime census for one project and return a merged report."""
        package_name = self._package_name_for_project(project)
        if package_name is None:
            return m.Infra.ValidationReport(
                passed=True,
                violations=(),
                summary=f"{project.name}: no importable package found",
            )
        module_names = self._walk_modules(package_name)
        import_failures: list[str] = []
        real_modules: list[str] = []
        for name in module_names:
            if ": " in name:
                import_failures.append(name)
            else:
                real_modules.append(name)
        if self.target_module is not None:
            real_modules = [
                name
                for name in real_modules
                if name == self.target_module
                or name.startswith(self.target_module + ".")
            ]
        real_modules = [
            name
            for name in real_modules
            if not frozenset(config.Infra.codegen.source_scan_ignored).intersection(
                name.split(".")
            )
        ]
        all_reports: list[m.Infra.ValidationReport] = [
            m.Infra.ValidationReport(
                passed=False,
                violations=tuple(import_failures),
                summary=(f"{project.name}: {len(import_failures)} import failure(s)"),
            )
        ]
        for module_name in real_modules:
            all_reports.extend(self._check_module(module_name))
        merged_violations = tuple(
            violation for report in all_reports for violation in report.violations
        )
        passed = not merged_violations
        summary = (
            f"{project.name}: {len(merged_violations)} runtime violation(s)"
            if not passed
            else f"{project.name}: runtime census passed ({len(real_modules)} module(s))"
        )
        return m.Infra.ValidationReport(
            passed=passed, violations=merged_violations, summary=summary
        )

    def build_report(self) -> p.Result[m.Infra.ValidationReport]:
        """Build one validation report for the selected workspace projects."""
        projects_result = u.Infra.projects(self.repository_root)
        if projects_result.failure:
            return r[m.Infra.ValidationReport].fail(
                projects_result.error or "project discovery failed"
            )
        projects = self._selected_projects(projects_result.unwrap())
        if not projects:
            return r[m.Infra.ValidationReport].ok(
                m.Infra.ValidationReport(
                    passed=True,
                    violations=(),
                    summary="runtime census: no projects selected",
                )
            )
        merged_violations: list[str] = []
        for project in projects:
            report = self._project_report(project)
            merged_violations.extend(report.violations)
        passed = not merged_violations
        summary = (
            "runtime census passed"
            if passed
            else f"runtime census found {len(merged_violations)} violation(s)"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=tuple(merged_violations), summary=summary
            )
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute runtime census and collapse the report to ``r[bool]``."""
        report_result = self.build_report()
        if report_result.failure:
            return r[bool].fail(report_result.error or "runtime census failed")
        report = report_result.value
        if report.passed:
            return r[bool].ok(True)
        if self.output_format == c.Cli.OutputFormats.JSON:
            return r[bool].fail(report.model_dump_json())
        return r[bool].fail(self._render_text_summary(report))

    @staticmethod
    def _render_text_summary(report: m.Infra.ValidationReport) -> str:
        """Render violations grouped by rule with file:line context.

        The flat list the census used to emit made it impossible to see which
        rule or file owned the bulk of the debt. Grouping by rule_id (falling
        back to 'UNKNOWN' when a violation string carries no bracket) gives the
        operator a histogram and a per-rule file list in one read.
        """
        import re
        from collections import defaultdict

        rule_buckets: dict[str, list[str]] = defaultdict(list)
        for violation in report.violations:
            match = re.search(r"\[(ENFORCE-\d+)\]", violation)
            rule_id = match.group(1) if match else "UNKNOWN"
            rule_buckets[rule_id].append(violation)
        header = f"{report.summary}\n"
        sections: list[str] = [header]
        for rule_id in sorted(rule_buckets):
            entries = rule_buckets[rule_id]
            sections.append(f"  {rule_id}: {len(entries)} violation(s)")
            seen: set[str] = set()
            for entry in entries:
                short = entry.split(": ", 1)[-1] if ": " in entry else entry
                if short not in seen:
                    seen.add(short)
                    sections.append(f"    {short}")
        return "\n".join(sections)


__all__: list[str] = ["FlextInfraRuntimeCensusValidator"]
