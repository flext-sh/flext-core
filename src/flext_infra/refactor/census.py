"""Utilities usage census — AST/CST-based method usage accounting.

Scans all workspace projects to count usages of public methods from
``flext_core._utilities`` classes, distinguishing alias (``u``) access
from direct class references.

Uses:
- ``libcst`` for precise CST-based import and attribute resolution
- ``ast`` (stdlib) for public method extraction from class definitions
- ``u.Infra.dotted_name`` / ``u.Infra.root_name`` from refactor._utilities
- ``u.Infra.iter_python_files`` / ``u.Infra.discover_project_roots``

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import override

import libcst as cst

from flext_core import r
from flext_infra import c, m, u

__all__ = ["FlextInfraRefactorCensus"]

# ── Result type aliases ──────────────────────────────────────────────────────

type RCensusReport = r[m.Infra.Refactor.CensusReport]


# ── LibCST Visitors ──────────────────────────────────────────────────────────


class _ImportDiscoveryVisitor(cst.CSTVisitor):
    """Discover ``u`` aliases and direct ``FlextUtilities*`` imports via LibCST.

    Follows the same pattern as ``ImportCollector`` from
    ``flext_infra.refactor.dependency_analyzer``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.u_aliases: set[str] = set()
        self.direct_imports: dict[str, str] = {}

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        module_str = u.Infra.dotted_name(node.module)
        if not module_str:
            return
        if isinstance(node.names, cst.ImportStar):
            return
        for alias in node.names:
            imported_name = alias.name.value if isinstance(alias.name, cst.Name) else ""
            local_name = (
                u.Infra.asname_to_local(alias.asname) if alias.asname else imported_name
            )
            if not local_name:
                local_name = imported_name

            # Check for `u` import from flext_core or subproject
            if imported_name in {"u", "FlextUtilities"} and (
                "flext_core" in module_str or "flext_" in module_str
            ):
                self.u_aliases.add(local_name)

            if (
                imported_name.startswith("FlextUtilities")
                and "flext_core" in module_str
            ):
                self.direct_imports[local_name] = imported_name


class _UsageCollector(cst.CSTVisitor):
    """Detect ``_utilities`` method accesses via LibCST attribute resolution.

    Uses ``u.Infra.root_name()`` for CST expression root extraction.

    Detects three access modes:
    1. ``u.method_name(...)`` — flat alias via FlextUtilities facade
    2. ``u.ClassName.method_name()`` — namespaced via inner class
    3. ``FlextUtilitiesXxx.method_name()`` — direct class reference
    """

    def __init__(
        self,
        *,
        method_index: dict[str, set[str]],
        flat_aliases: dict[str, tuple[str, str]],
        inner_class_map: dict[str, str],
        u_aliases: set[str],
        direct_imports: dict[str, str],
        file_path: Path,
        project_name: str,
    ) -> None:
        super().__init__()
        self.method_index = method_index
        self.flat_aliases = flat_aliases
        self.inner_class_map = inner_class_map
        self.u_aliases = u_aliases
        self.direct_imports = direct_imports
        self.file_path = file_path
        self.project_name = project_name
        self.records: list[m.Infra.Refactor.CensusUsageRecord] = []

    @override
    def visit_Attribute(self, node: cst.Attribute) -> None:
        method_name = node.attr.value
        value = node.value

        # Pattern 1: u.method_name (flat alias)
        if isinstance(value, cst.Name) and value.value in self.u_aliases:
            if method_name in self.flat_aliases:
                cls, orig = self.flat_aliases[method_name]
                self._record(cls, orig, c.Infra.Refactor.Census.MODE_ALIAS_FLAT)

        # Pattern 2: u.ClassName.method_name (namespaced inner class)
        if (
            isinstance(value, cst.Attribute)
            and u.Infra.root_name(value) in self.u_aliases
        ):
            inner_name = value.attr.value
            base_class = self.inner_class_map.get(inner_name, "")
            if (
                base_class in self.method_index
                and method_name in self.method_index[base_class]
            ):
                self._record(
                    base_class, method_name, c.Infra.Refactor.Census.MODE_ALIAS_NS
                )
        # Pattern 3: FlextUtilitiesXxx.method_name (direct)
        if isinstance(value, cst.Name):
            actual = self.direct_imports.get(value.value, value.value)
            if actual in self.method_index and method_name in self.method_index[actual]:
                self._record(actual, method_name, c.Infra.Refactor.Census.MODE_DIRECT)

    def _record(self, class_name: str, method_name: str, mode: str) -> None:
        self.records.append(
            m.Infra.Refactor.CensusUsageRecord(
                class_name=class_name,
                method_name=method_name,
                access_mode=mode,
                file_path=str(self.file_path),
                project=self.project_name,
            ),
        )


# ── Namespace class ──────────────────────────────────────────────────────────


class FlextInfraRefactorCensus:
    """AST/CST census of ``_utilities`` method usage across the workspace.

    Usage::

        from flext_infra.refactor.census import FlextInfraRefactorCensus

        scanner = FlextInfraRefactorCensus()
        result = scanner.run(workspace_root)
        if result.is_success:
            report = result.value
    """

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, workspace_root: Path) -> RCensusReport:
        """Execute the full census and return a structured report."""
        utilities_dir = (
            workspace_root
            / c.Infra.Refactor.Census.CORE_PROJECT
            / c.Infra.Paths.DEFAULT_SRC_DIR
            / c.Infra.Refactor.Census.UTILITIES_PACKAGE
        )
        facade_path = (
            workspace_root
            / c.Infra.Refactor.Census.CORE_PROJECT
            / c.Infra.Paths.DEFAULT_SRC_DIR
            / c.Infra.Refactor.Census.FACADE_MODULE
        )

        # Phase 1: extract public methods (ast)
        class_methods = self._extract_public_methods(utilities_dir)
        method_index: dict[str, set[str]] = {
            cls: {mi.name for mi in methods} for cls, methods in class_methods.items()
        }

        # Phase 2: build facade alias maps (ast)
        flat_aliases = self._build_flat_alias_map(facade_path)
        inner_class_map = self._build_inner_class_map(facade_path)

        # Phase 3: discover & scan (libcst + u.Infra)
        project_roots = u.Infra.discover_project_roots(workspace_root=workspace_root)
        all_files = u.Infra.iter_python_files(
            workspace_root=workspace_root,
            include_tests=True,
            include_examples=True,
            include_scripts=True,
        )
        scan_files = [
            f
            for f in all_files
            if c.Infra.Refactor.Census.UTILITIES_PACKAGE.replace("/", ".").replace(
                "\\", "."
            )
            not in str(f)
            and "__pycache__" not in str(f)
        ]

        all_records: list[m.Infra.Refactor.CensusUsageRecord] = []
        parse_errors = 0
        for fp in scan_files:
            project_name = self._identify_project(fp, project_roots)
            records, ok = self._scan_file(
                fp,
                method_index=method_index,
                flat_aliases=flat_aliases,
                inner_class_map=inner_class_map,
                project_name=project_name,
            )
            if not ok:
                parse_errors += 1
            all_records.extend(records)

        # Phase 4: aggregate
        report = self._build_report(
            class_methods=class_methods,
            records=all_records,
            files_scanned=len(scan_files),
            parse_errors=parse_errors,
        )
        out: RCensusReport = r[m.Infra.Refactor.CensusReport].ok(report)
        return out

    # ── Phase 1 internals ────────────────────────────────────────────────

    @staticmethod
    def _extract_public_methods(
        utilities_dir: Path,
    ) -> dict[str, list[m.Infra.Refactor.CensusMethodInfo]]:
        """Extract public methods from _utilities classes using stdlib ast."""
        result: dict[str, list[m.Infra.Refactor.CensusMethodInfo]] = {}
        for py_file in sorted(utilities_dir.glob(c.Infra.Extensions.PYTHON_GLOB)):
            if py_file.name == c.Infra.Files.INIT_PY:
                continue
            try:
                source = py_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue
            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                methods: list[m.Infra.Refactor.CensusMethodInfo] = []
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith(
                        "_"
                    ):
                        decs = [
                            d.id
                            if isinstance(d, ast.Name)
                            else (d.attr if isinstance(d, ast.Attribute) else "")
                            for d in item.decorator_list
                        ]
                        mtype = (
                            "static"
                            if "staticmethod" in decs
                            else "class"
                            if "classmethod" in decs
                            else "instance"
                        )
                        info = m.Infra.Refactor.CensusMethodInfo(
                            name=item.name,
                            method_type=mtype,
                            source_file=py_file.name,
                        )
                        if not any(existing.name == info.name for existing in methods):
                            methods.append(info)
                    elif isinstance(item, ast.ClassDef) and not item.name.startswith(
                        "_"
                    ):
                        for inner in ast.iter_child_nodes(item):
                            if isinstance(
                                inner, ast.FunctionDef
                            ) and not inner.name.startswith("_"):
                                info = m.Infra.Refactor.CensusMethodInfo(
                                    name=f"{item.name}.{inner.name}",
                                    method_type="static",
                                    source_file=py_file.name,
                                )
                                if not any(
                                    existing.name == info.name for existing in methods
                                ):
                                    methods.append(info)
                if methods:
                    result[node.name] = methods
        return result

    # ── Phase 2 internals ────────────────────────────────────────────────

    @staticmethod
    def _build_flat_alias_map(facade_path: Path) -> dict[str, tuple[str, str]]:
        """Parse FlextUtilities facade to build flat alias -> (class, method) map."""
        try:
            source = facade_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}

        alias_map: dict[str, tuple[str, str]] = {}
        for node in ast.iter_child_nodes(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == "FlextUtilities"):
                continue
            for item in ast.iter_child_nodes(node):
                if not isinstance(item, ast.Assign):
                    continue
                for target in item.targets:
                    if not isinstance(target, ast.Name) or not isinstance(
                        item.value, ast.Call
                    ):
                        continue
                    call = item.value
                    if not (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "staticmethod"
                        and call.args
                    ):
                        continue
                    arg = call.args[0]
                    if isinstance(arg, ast.Attribute):
                        if isinstance(arg.value, ast.Name):
                            alias_map[target.id] = (arg.value.id, arg.attr)
                        elif isinstance(arg.value, ast.Attribute) and isinstance(
                            arg.value.value, ast.Name
                        ):
                            alias_map[target.id] = (
                                arg.value.value.id,
                                f"{arg.value.attr}.{arg.attr}",
                            )
        return alias_map

    @staticmethod
    def _build_inner_class_map(facade_path: Path) -> dict[str, str]:
        """Map inner class short names to base class names."""
        try:
            source = facade_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError, OSError):
            return {}

        name_map: dict[str, str] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FlextUtilities":
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, ast.ClassDef):
                        for base in item.bases:
                            if isinstance(base, ast.Name):
                                name_map[item.name] = base.id
        return name_map

    # ── Phase 3 internals ────────────────────────────────────────────────

    @staticmethod
    def _identify_project(file_path: Path, project_roots: list[Path]) -> str:
        """Identify project name for a file path (most-specific root wins)."""
        best: Path | None = None
        for root in project_roots:
            try:
                file_path.relative_to(root)
            except ValueError:
                continue
            if best is None or len(root.parts) > len(best.parts):
                best = root
        return best.name if best else c.Infra.Defaults.UNKNOWN

    @staticmethod
    def _scan_file(
        file_path: Path,
        *,
        method_index: dict[str, set[str]],
        flat_aliases: dict[str, tuple[str, str]],
        inner_class_map: dict[str, str],
        project_name: str,
    ) -> tuple[list[m.Infra.Refactor.CensusUsageRecord], bool]:
        """Scan a single file with LibCST. Returns (records, parse_ok)."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = cst.parse_module(source)
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
            return [], False

        import_vis = _ImportDiscoveryVisitor()
        tree.visit(import_vis)

        if not import_vis.u_aliases and not import_vis.direct_imports:
            return [], True

        usage_vis = _UsageCollector(
            method_index=method_index,
            flat_aliases=flat_aliases,
            inner_class_map=inner_class_map,
            u_aliases=import_vis.u_aliases,
            direct_imports=import_vis.direct_imports,
            file_path=file_path,
            project_name=project_name,
        )
        tree.visit(usage_vis)
        return usage_vis.records, True

    # ── Phase 4: aggregation ─────────────────────────────────────────────

    @staticmethod
    def _build_report(
        *,
        class_methods: dict[str, list[m.Infra.Refactor.CensusMethodInfo]],
        records: list[m.Infra.Refactor.CensusUsageRecord],
        files_scanned: int,
        parse_errors: int,
    ) -> m.Infra.Refactor.CensusReport:
        """Aggregate raw records into a structured report."""
        # Count per (class, method, mode)
        counter: Counter[tuple[str, str, str]] = Counter()
        project_counter: Counter[tuple[str, str, str, str]] = Counter()
        for rec in records:
            counter[rec.class_name, rec.method_name, rec.access_mode] += 1
            project_counter[
                rec.project, rec.class_name, rec.method_name, rec.access_mode
            ] += 1

        # Build class summaries
        class_summaries: list[m.Infra.Refactor.CensusClassSummary] = []
        total_unused = 0
        for cls_name in sorted(class_methods.keys()):
            methods = class_methods[cls_name]
            method_summaries: list[m.Infra.Refactor.CensusMethodSummary] = []
            for mi in methods:
                af = counter.get(
                    (cls_name, mi.name, c.Infra.Refactor.Census.MODE_ALIAS_FLAT), 0
                )
                an = counter.get(
                    (cls_name, mi.name, c.Infra.Refactor.Census.MODE_ALIAS_NS), 0
                )
                dr = counter.get(
                    (cls_name, mi.name, c.Infra.Refactor.Census.MODE_DIRECT), 0
                )
                total = af + an + dr
                if total == 0:
                    total_unused += 1
                method_summaries.append(
                    m.Infra.Refactor.CensusMethodSummary(
                        name=mi.name,
                        method_type=mi.method_type,
                        alias_flat=af,
                        alias_namespaced=an,
                        direct=dr,
                        total=total,
                    ),
                )
            class_summaries.append(
                m.Infra.Refactor.CensusClassSummary(
                    class_name=cls_name,
                    source_file=methods[0].source_file if methods else "",
                    methods=method_summaries,
                ),
            )

        # Build per-project breakdown
        projects: dict[str, list[m.Infra.Refactor.CensusProjectMethodUsage]] = (
            defaultdict(list)
        )
        for (proj, cls, method, mode), count in sorted(project_counter.items()):
            projects[proj].append(
                m.Infra.Refactor.CensusProjectMethodUsage(
                    class_name=cls,
                    method_name=method,
                    access_mode=mode,
                    count=count,
                ),
            )
        project_summaries = [
            m.Infra.Refactor.CensusProjectSummary(
                project_name=proj,
                usages=usages,
                total=sum(u.count for u in usages),
            )
            for proj, usages in sorted(projects.items())
        ]

        total_methods = sum(len(methods) for methods in class_methods.values())

        return m.Infra.Refactor.CensusReport(
            classes=class_summaries,
            projects=project_summaries,
            total_classes=len(class_methods),
            total_methods=total_methods,
            total_usages=len(records),
            total_unused=total_unused,
            files_scanned=files_scanned,
            parse_errors=parse_errors,
        )

    # ── Report formatting ────────────────────────────────────────────────

    @staticmethod
    def format_report(report: m.Infra.Refactor.CensusReport) -> str:
        """Format a CensusReport into human-readable text output."""
        lines: list[str] = []
        sep = "=" * 110

        lines.append(sep)
        lines.append("FLEXT _utilities Method Usage Census")
        lines.append("Engine: libcst + stdlib ast | Infrastructure: flext_infra")
        lines.append(sep)
        lines.append(
            f"\nClasses: {report.total_classes} | Methods: {report.total_methods}"
            f" | Usages: {report.total_usages} | Unused: {report.total_unused}"
            f" | Files: {report.files_scanned} | Parse errors: {report.parse_errors}"
        )

        # Method table
        lines.append("")
        lines.append(
            f"{'CLASS':<40} {'METHOD':<30} {'u.flat':<8} {'u.NS':<8} {'Direct':<8} {'Total':<8}"
        )
        lines.append(sep)

        grand_af = grand_an = grand_dr = 0
        for cs in report.classes:
            for ms in cs.methods:
                grand_af += ms.alias_flat
                grand_an += ms.alias_namespaced
                grand_dr += ms.direct
                marker = "  " if ms.total > 0 else "⚠️"
                lines.append(
                    f"{marker} {cs.class_name:<38} {ms.name:<30}"
                    f" {ms.alias_flat:<8} {ms.alias_namespaced:<8}"
                    f" {ms.direct:<8} {ms.total:<8}"
                )
            lines.append("-" * 110)

        grand_total = grand_af + grand_an + grand_dr
        lines.append(
            f"\n{'GRAND TOTAL':<71} {grand_af:<8} {grand_an:<8} {grand_dr:<8} {grand_total:<8}"
        )

        # Per-project
        lines.append(f"\n\n{sep}")
        lines.append("PER-PROJECT BREAKDOWN")
        lines.append(sep)
        for ps in report.projects:
            alias_total = sum(
                u.count
                for u in ps.usages
                if u.access_mode != c.Infra.Refactor.Census.MODE_DIRECT
            )
            direct_total = sum(
                u.count
                for u in ps.usages
                if u.access_mode == c.Infra.Refactor.Census.MODE_DIRECT
            )
            lines.append(
                f"\n📦 {ps.project_name} (alias: {alias_total}, direct: {direct_total}, total: {ps.total})"
            )
            for pu in ps.usages:
                lines.append(
                    f"  {pu.class_name}.{pu.method_name}: {pu.access_mode}={pu.count}"
                )

        # Unused
        lines.append(f"\n\n{sep}")
        lines.append("UNUSED PUBLIC METHODS")
        lines.append(sep)
        current_cls = ""
        for cs in report.classes:
            unused = [ms for ms in cs.methods if ms.total == 0]
            if unused:
                if cs.class_name != current_cls:
                    lines.append(f"\n  {cs.class_name} ({cs.source_file}):")
                    current_cls = cs.class_name
                for ms in unused:
                    lines.append(f"    - {ms.name}")
        lines.append(f"\n  Total unused: {report.total_unused}/{report.total_methods}")

        # Top 20
        lines.append(f"\n\n{sep}")
        lines.append("TOP 20 MOST USED METHODS")
        lines.append(sep)
        all_methods = [
            (cs.class_name, ms.name, ms.total)
            for cs in report.classes
            for ms in cs.methods
        ]
        for cls, method, total in sorted(all_methods, key=lambda x: x[2], reverse=True)[
            :20
        ]:
            lines.append(f"  {total:>5}x  {cls}.{method}")

        return "\n".join(lines)

    @staticmethod
    def export_json(report: m.Infra.Refactor.CensusReport, output_path: Path) -> None:
        """Export report as JSON for tooling integration."""
        output_path.write_text(
            report.model_dump_json(indent=2),
            encoding=c.Infra.Encoding.DEFAULT,
        )
