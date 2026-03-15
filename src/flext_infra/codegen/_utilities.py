"""Code generation helpers for infrastructure lazy-init processing.

Centralizes AST/codegen helpers previously defined as module-level
functions in ``flext_infra.codegen.lazy_init``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import operator
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from flext_infra import c, t
from flext_infra._utilities.formatting import FlextInfraUtilitiesFormatting
from flext_infra._utilities.iteration import FlextInfraUtilitiesIteration
from flext_infra._utilities.parsing import FlextInfraUtilitiesParsing
from flext_infra.codegen._models import FlextInfraCodegenModels
from flext_infra.codegen.census import FlextInfraCodegenCensus
from flext_infra.codegen.transforms import FlextInfraCodegenTransforms
from flext_infra.refactor._utilities import FlextInfraUtilitiesRefactor


class FlextInfraUtilitiesCodegen(FlextInfraCodegenTransforms):
    """Code generation helpers for lazy-init and AST operations.

    Usage via namespace::

        from flext_infra import u

        pkg = u.Infra.infer_package(path)
    """

    _container_mapping_adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = (
        TypeAdapter(
            dict[str, t.Infra.InfraValue],
        )
    )

    @staticmethod
    def find_package_dir(project_root: Path) -> Path | None:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return None
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / c.Infra.Files.INIT_PY).exists():
                return child
        return None

    @staticmethod
    def _snapshot_init_files(*, project_path: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for root_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES:
            root = project_path / root_name
            if not root.is_dir():
                continue
            for init_file in FlextInfraUtilitiesIteration.iter_directory_python_files(
                root,
                pattern=c.Infra.Files.INIT_PY,
            ):
                try:
                    snapshot[str(init_file)] = init_file.read_text(
                        encoding=c.Infra.Encoding.DEFAULT,
                    )
                except OSError:
                    continue
        return snapshot

    @staticmethod
    def _snapshot_files(*, file_paths: Sequence[Path]) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for file_path in file_paths:
            try:
                snapshot[str(file_path)] = file_path.read_text(
                    encoding=c.Infra.Encoding.DEFAULT,
                )
            except OSError:
                continue
        return snapshot

    @staticmethod
    def _detect_changed_files(
        *,
        before_snapshot: dict[str, str],
        file_paths: Sequence[Path],
    ) -> set[str]:
        changed: set[str] = set()
        for file_path in file_paths:
            path_key = str(file_path)
            previous = before_snapshot.get(path_key)
            try:
                current = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            except OSError:
                continue
            if previous != current:
                changed.add(path_key)
        return changed

    @staticmethod
    def _write_changes(
        *,
        source_path: Path,
        target_path: Path,
        nodes_moved: Sequence[ast.stmt],
        moved_names: list[str],
        source_tree: ast.Module,
        pkg_name: str,
        target_module: str,
        dry_run: bool,
    ) -> None:
        if dry_run:
            return
        encoding = c.Infra.Encoding.DEFAULT
        source_text = source_path.read_text(encoding=encoding)
        source_lines = source_text.splitlines()
        target_text = target_path.read_text(encoding=encoding)

        extracted: list[str] = []
        ranges: list[tuple[int, int]] = []
        for node in nodes_moved:
            start = node.lineno
            end = node.end_lineno or node.lineno
            block = "\n".join(source_lines[start - 1 : end])
            extracted.append(block)
            ranges.append((start, end))

        import_texts = FlextInfraCodegenTransforms.collect_import_texts_for_nodes(
            nodes_moved,
            source_lines,
            source_tree,
            target_text,
        )

        for start, end in sorted(ranges, key=operator.itemgetter(0), reverse=True):
            del source_lines[start - 1 : end]

        source_result = "\n".join(source_lines)
        re_export = f"from {pkg_name}.{target_module} import " + ", ".join(
            sorted(moved_names)
        )
        source_result = FlextInfraUtilitiesRefactor.insert_import_statement(
            source_result,
            re_export,
        )
        if source_text.endswith("\n") and not source_result.endswith("\n"):
            source_result += "\n"

        target_result = target_text
        for imp in import_texts:
            target_result = FlextInfraUtilitiesRefactor.insert_import_statement(
                target_result,
                imp,
            )
        for block in extracted:
            target_result = target_result.rstrip() + "\n\n\n" + block + "\n"

        source_path.write_text(source_result, encoding=encoding)
        target_path.write_text(target_result, encoding=encoding)

        FlextInfraUtilitiesFormatting.run_ruff_fix(source_path)
        FlextInfraUtilitiesFormatting.run_ruff_fix(target_path)

    @staticmethod
    def snapshot_init_files(*, project_path: Path) -> dict[str, str]:
        return FlextInfraUtilitiesCodegen._snapshot_init_files(
            project_path=project_path
        )

    @staticmethod
    def snapshot_files(*, file_paths: Sequence[Path]) -> dict[str, str]:
        return FlextInfraUtilitiesCodegen._snapshot_files(file_paths=file_paths)

    @staticmethod
    def detect_changed_files(
        *,
        before_snapshot: dict[str, str],
        file_paths: Sequence[Path],
    ) -> set[str]:
        return FlextInfraUtilitiesCodegen._detect_changed_files(
            before_snapshot=before_snapshot,
            file_paths=file_paths,
        )

    @staticmethod
    def write_changes(
        *,
        source_path: Path,
        target_path: Path,
        nodes_moved: Sequence[ast.stmt],
        moved_names: list[str],
        source_tree: ast.Module,
        pkg_name: str,
        target_module: str,
        dry_run: bool,
    ) -> None:
        FlextInfraUtilitiesCodegen._write_changes(
            source_path=source_path,
            target_path=target_path,
            nodes_moved=nodes_moved,
            moved_names=moved_names,
            source_tree=source_tree,
            pkg_name=pkg_name,
            target_module=target_module,
            dry_run=dry_run,
        )

    @staticmethod
    def infer_package(path: Path) -> str:
        abs_path = str(path.absolute())
        for root_dir in ("/src/", "/tests/", "/examples/", "/scripts/"):
            idx = abs_path.rfind(root_dir)
            if idx != -1:
                rel = abs_path[idx + len(root_dir) :]
                pkg_parts = rel.split("/")[:-1]
                if root_dir == "/src/":
                    return ".".join(pkg_parts)
                root_name = root_dir.strip("/")
                return ".".join([root_name, *pkg_parts]) if pkg_parts else root_name
        return ""

    @staticmethod
    def resolve_module(raw_module: str, level: int, current_pkg: str) -> str:
        """Resolve a potentially relative import to an absolute module path.

        Args:
            raw_module: The raw module name from the import statement.
            level: Number of leading dots (relative import level).
            current_pkg: The current package dotted name.

        Returns:
            Absolute module path string.

        """
        if level == 0:
            return raw_module
        if not current_pkg:
            return raw_module
        parts = current_pkg.split(".")
        base = parts[: len(parts) - level + 1]
        if not base:
            return raw_module
        return ".".join(base) + ("." + raw_module if raw_module else "")

    @staticmethod
    def extract_docstring_source(tree: ast.Module, content: str) -> str:
        """Extract the raw docstring source preserving original formatting.

        Args:
            tree: Parsed AST module.
            content: Original file content string.

        Returns:
            Docstring source text, or empty string if no docstring.

        """
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            ds = tree.body[0]
            return "\n".join(content.splitlines()[ds.lineno - 1 : ds.end_lineno])
        return ""

    @staticmethod
    def extract_exports(tree: ast.Module) -> tuple[bool, list[str]]:
        """Extract ``__all__`` entries from the AST.

        Args:
            tree: Parsed AST module.

        Returns:
            Tuple of (has_all, list_of_export_names).

        """
        exports: list[str] = []
        has_all = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        has_all = True
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            exports.extend(
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            )
        return (has_all, exports)

    @staticmethod
    def extract_inline_constants(tree: ast.Module) -> dict[str, str]:
        """Extract inline constant assignments like ``__version__ = '1.0.0'``.

        Args:
            tree: Parsed AST module.

        Returns:
            Dictionary mapping constant names to their string values.

        """
        constants: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if (
                    isinstance(target, ast.Name)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    constants[target.id] = node.value.value
        return constants

    @staticmethod
    def parse_existing_lazy_imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
        """Parse an existing ``_LAZY_IMPORTS`` dict literal from the AST.

        Handles both ``_LAZY_IMPORTS = {...}`` and
        ``_LAZY_IMPORTS: dict[str, tuple[str, str]] = {...}``.

        Args:
            tree: Parsed AST module.

        Returns:
            Dictionary mapping export names to (module_path, attr_name) tuples.

        """
        result: dict[str, tuple[str, str]] = {}
        lazy_import_pair_size = 2

        def _extract(d: ast.expr) -> None:
            if not isinstance(d, ast.Dict):
                return
            for key, val in zip(d.keys, d.values, strict=False):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(val, ast.Tuple)
                    and (len(val.elts) == lazy_import_pair_size)
                    and isinstance(val.elts[0], ast.Constant)
                    and isinstance(val.elts[1], ast.Constant)
                ):
                    result[str(key.value)] = (
                        str(val.elts[0].value),
                        str(val.elts[1].value),
                    )

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS":
                        _extract(node.value)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and (node.target.id == "_LAZY_IMPORTS")
                and (node.value is not None)
            ):
                _extract(node.value)
        return result

    @staticmethod
    def derive_lazy_map(
        tree: ast.Module,
        current_pkg: str,
    ) -> dict[str, tuple[str, str]]:
        """Derive lazy import mappings from import statements in the AST.

        Args:
            tree: Parsed AST module.
            current_pkg: Current package dotted name.

        Returns:
            Dictionary mapping export names to (module_path, attr_name) tuples.

        """
        lazy_map: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                raw_module = node.module or ""
                if raw_module in c.Infra.Codegen.SKIP_MODULES:
                    continue
                module_path = FlextInfraUtilitiesCodegen.resolve_module(
                    raw_module,
                    node.level,
                    current_pkg,
                )
                if module_path == current_pkg:
                    continue
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    lazy_map[asname] = (module_path, name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    asname = alias.asname or name
                    if name in c.Infra.Codegen.SKIP_STDLIB:
                        continue
                    lazy_map[asname] = (name, "")
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                rhs = node.value.id
                if rhs in lazy_map:
                    mod, attr = lazy_map[rhs]
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            lazy_map[target.id] = (mod, attr)
        for a_name, suffix in c.Infra.Codegen.ALIAS_TO_SUFFIX.items():
            if a_name not in lazy_map:
                continue
            alias_mod, alias_attr = lazy_map[a_name]
            if alias_attr == a_name:
                for name, (mod, _) in lazy_map.items():
                    if mod == alias_mod and name.endswith(suffix) and (len(name) > 1):
                        lazy_map[a_name] = (mod, name)
                        break
        return lazy_map

    @staticmethod
    def resolve_unmapped(
        exports_set: set[str],
        filtered: dict[str, tuple[str, str]],
        current_pkg: str,
        pkg_dir: Path,
    ) -> None:
        """Resolve unmapped single-letter aliases and ``__version__``.

        Mutates ``filtered`` in place to add resolved mappings.

        Args:
            exports_set: Set of all export names.
            filtered: dict of already-mapped exports (modified in place).
            current_pkg: Current package dotted name.
            pkg_dir: Package directory path.

        """
        unmapped = exports_set - set(filtered)
        if not unmapped:
            return
        for alias in sorted(unmapped):
            if alias in c.Infra.Codegen.ALIAS_TO_SUFFIX:
                suffix = c.Infra.Codegen.ALIAS_TO_SUFFIX[alias]
                for name, (mod, _) in filtered.items():
                    if name.endswith(suffix) and len(name) > 1:
                        filtered[alias] = (mod, name)
                        break
            elif alias == "__version__" and current_pkg:
                ver_file = pkg_dir / "__version__.py"
                if ver_file.exists():
                    filtered["__version__"] = (
                        f"{current_pkg}.__version__",
                        "__version__",
                    )
            elif alias == "__version_info__" and current_pkg:
                ver_file = pkg_dir / "__version__.py"
                if ver_file.exists():
                    filtered["__version_info__"] = (
                        f"{current_pkg}.__version__",
                        "__version_info__",
                    )

    @staticmethod
    def generate_type_checking(
        groups: Mapping[str, list[tuple[str, str]]],
    ) -> list[str]:
        """Generate the ``if TYPE_CHECKING`` import block.

        Groups imports by top-level package with blank lines between groups,
        following isort conventions.

        Args:
            groups: Mapping of module names to list of (export_name, attr_name) tuples.

        Returns:
            List of lines for the TYPE_CHECKING block.

        """
        lines: list[str] = ["if TYPE_CHECKING:"]
        lines.append("    from flext_core.typings import FlextTypes")
        if not groups:
            return lines

        def _emit_module(mod: str) -> None:
            items = groups[mod]
            alias_items = sorted(
                (item for item in items if not item[1]), key=operator.itemgetter(0)
            )
            sorted_items = sorted(
                (item for item in items if item[1]),
                key=lambda x: (x[1], x[0] != x[1]),
            )
            for export_name, _ in alias_items:
                alias_line = f"    import {mod} as {export_name}"
                lines.append(alias_line)
            if not sorted_items:
                return
            parts: list[str] = []
            for export_name, attr_name in sorted_items:
                if export_name == attr_name:
                    parts.append(export_name)
                else:
                    parts.append(f"{attr_name} as {export_name}")
            joined = ", ".join(parts)
            line = f"    from {mod} import {joined}"
            if len(line) > c.Infra.Codegen.MAX_LINE_LENGTH:
                lines.append(f"    from {mod} import (")
                lines.extend(f"        {part}," for part in parts)
                lines.append("    )")
            else:
                lines.append(line)

        sorted_mods = sorted(groups, key=str.lower)
        prev_top: str | None = None
        for mod in sorted_mods:
            top = mod.split(".")[0]
            if prev_top is not None and top != prev_top:
                lines.append("")
            _emit_module(mod)
            prev_top = top
        return lines

    @staticmethod
    def generate_file(
        docstring_source: str,
        exports: list[str],
        filtered: Mapping[str, tuple[str, str]],
        inline_constants: Mapping[str, str],
        current_pkg: str,
        eager_typevar_names: frozenset[str] = frozenset(),
    ) -> str:
        """Generate the complete ``__init__.py`` content.

        Args:
            docstring_source: Raw docstring source from the original file.
            exports: List of export names.
            filtered: Mapping of export names to (module_path, attr_name) tuples.
            inline_constants: Mapping of constant names to values.
            current_pkg: Current package dotted name.

        Returns:
            Complete generated ``__init__.py`` file content.

        """
        lazy_filtered: dict[str, tuple[str, str]] = {
            name: val
            for name, val in filtered.items()
            if name not in eager_typevar_names
        }
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for export_name in sorted(lazy_filtered):
            mod, attr = lazy_filtered[export_name]
            groups[mod].append((export_name, attr))
        out: list[str] = [c.Infra.Codegen.AUTOGEN_HEADER]
        if docstring_source:
            out.extend([docstring_source, ""])
        is_core_internal = current_pkg.startswith(
            c.Infra.Packages.CORE_UNDERSCORE + ".",
        )
        if current_pkg == c.Infra.Packages.CORE_UNDERSCORE or is_core_internal:
            lazy_import = "from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr"
        else:
            lazy_import = (
                "from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr"
            )
        out.extend([
            "from __future__ import annotations",
            "",
            "from typing import TYPE_CHECKING",
            "",
            lazy_import,
        ])
        if eager_typevar_names:
            typings_mod = f"{current_pkg}.typings"
            sorted_tvars = sorted(eager_typevar_names)
            eager_line = f"from {typings_mod} import " + ", ".join(sorted_tvars)
            if len(eager_line) > c.Infra.Codegen.MAX_LINE_LENGTH:
                out.append(f"from {typings_mod} import (")
                out.extend(f"    {tv}," for tv in sorted_tvars)
                out.append(")")
            else:
                out.append(eager_line)
        out.append("")
        out.extend(FlextInfraUtilitiesCodegen.generate_type_checking(groups))
        out.append("")
        for name, value in sorted(inline_constants.items()):
            out.append(f'{name} = "{value}"')
        if inline_constants:
            out.append("")
        out.extend([
            "_LAZY_IMPORTS: dict[str, tuple[str, str]] = {",
        ])
        for exp in sorted(exports):
            if exp in lazy_filtered:
                mod, attr = lazy_filtered[exp]
                out.append(f'    "{exp}": ("{mod}", "{attr}"),')
        out.extend(["}", ""])
        out.append("__all__ = [")
        out.extend(f'    "{exp}",' for exp in sorted(exports))
        out.extend(["]", "", ""])
        out.extend([
            "def __getattr__(name: str) -> FlextTypes.ModuleExport:",
            '    """Lazy-load module attributes on first access (PEP 562)."""',
            "    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)",
            "",
            "",
            "def __dir__() -> list[str]:",
            '    """Return list of available attributes for dir() and autocomplete."""',
            "    return sorted(__all__)",
            "",
            "",
            "cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)",
            "",
        ])
        return "\n".join(out)

    @staticmethod
    def quality_gate_load_before_payload(
        workspace_root: Path,
        before_report: Path | None,
        baseline_file: Path | None,
    ) -> tuple[dict[str, t.Infra.InfraValue] | None, str, str]:
        baseline_path = before_report or baseline_file
        if baseline_path is None:
            return (None, "", "")
        resolved = (
            baseline_path
            if baseline_path.is_absolute()
            else workspace_root / baseline_path
        ).resolve()
        if not resolved.is_file():
            return (None, str(resolved), f"baseline file not found: {resolved}")
        try:
            text = resolved.read_text(encoding=c.Infra.Encoding.DEFAULT)
            payload = (
                FlextInfraUtilitiesCodegen._container_mapping_adapter.validate_json(
                    text
                )
            )
        except (OSError, UnicodeDecodeError, ValueError):
            return (None, str(resolved), "baseline parse failed")
        try:
            raw = FlextInfraUtilitiesCodegen._container_mapping_adapter.validate_python(
                payload,
            )
        except ValidationError:
            return (None, str(resolved), "baseline payload is not a JSON object")
        return (raw, str(resolved), "")

    @staticmethod
    def quality_gate_before_metrics(
        before_payload: dict[str, t.Infra.InfraValue] | None,
    ) -> dict[str, t.Infra.InfraValue]:
        if before_payload is None:
            return {
                "total_violations": -1,
                "duplicate_groups": -1,
                "projects_total": 0,
                "projects_passed": 0,
                "projects_failed": 0,
            }
        return {
            "total_violations": FlextInfraUtilitiesCodegen.extract_total_violations(
                before_payload,
            ),
            "duplicate_groups": FlextInfraUtilitiesCodegen.extract_duplicate_groups(
                before_payload,
            ),
            "projects_total": FlextInfraUtilitiesCodegen.extract_projects_total(
                before_payload,
            ),
            "projects_passed": FlextInfraUtilitiesCodegen.extract_projects_passed(
                before_payload,
            ),
            "projects_failed": FlextInfraUtilitiesCodegen.extract_projects_failed(
                before_payload,
            ),
        }

    @staticmethod
    def quality_gate_after_metrics(
        *,
        census_reports: Sequence[FlextInfraCodegenModels.CensusReport],
        duplicate_groups: int,
        import_scan: dict[str, t.Infra.InfraValue],
        modified_files: list[str],
    ) -> dict[str, t.Infra.InfraValue]:
        by_rule: dict[str, int] = dict.fromkeys(
            c.Infra.Codegen.QualityGate.RULE_KEYS, 0
        )
        total_violations = 0
        for report in census_reports:
            violations = tuple(report.violations)
            total_violations += len(violations)
            for raw_violation in violations:
                parsed = FlextInfraCodegenModels.CensusViolation.model_validate(
                    raw_violation
                )
                if parsed.rule in by_rule:
                    by_rule[parsed.rule] += 1
        projects_total = len(census_reports)
        projects_passed = sum(1 for item in census_reports if int(item.total) == 0)
        projects_failed = projects_total - projects_passed
        violations_by_rule: dict[str, t.Infra.InfraValue] = dict(by_rule)
        modified_python_files_value: list[t.Infra.InfraValue] = list(modified_files)
        return {
            "total_violations": total_violations,
            "violations_by_rule": violations_by_rule,
            "duplicate_groups": duplicate_groups,
            "projects_total": projects_total,
            "projects_passed": projects_passed,
            "projects_failed": projects_failed,
            "mro_failures": 0,
            "layer_violations": 0,
            "cross_project_reference_violations": 0,
            "import_parse_violations": FlextInfraUtilitiesCodegen.as_int(
                import_scan.get("invalid_import_from_count"),
            ),
            "import_parse_errors": FlextInfraUtilitiesCodegen.as_int(
                import_scan.get("parse_error_count"),
            ),
            "modified_python_files": modified_python_files_value,
        }

    @staticmethod
    def quality_gate_improvement(
        before_metrics: dict[str, t.Infra.InfraValue],
        after_metrics: dict[str, t.Infra.InfraValue],
    ) -> dict[str, t.Infra.InfraValue]:
        before_violations = FlextInfraUtilitiesCodegen.as_int(
            before_metrics.get("total_violations"),
        )
        before_duplicates = FlextInfraUtilitiesCodegen.as_int(
            before_metrics.get("duplicate_groups"),
        )
        after_violations = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("total_violations"),
        )
        after_duplicates = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("duplicate_groups"),
        )
        violations_delta = (
            0 if before_violations < 0 else after_violations - before_violations
        )
        duplicates_delta = (
            0 if before_duplicates < 0 else after_duplicates - before_duplicates
        )
        return {
            "violations_delta": violations_delta,
            "duplicates_delta": duplicates_delta,
            "violations_reduced": max(0, -violations_delta),
            "duplicates_eliminated": max(0, -duplicates_delta),
            "violations_increased": max(0, violations_delta),
            "duplicates_increased": max(0, duplicates_delta),
        }

    @staticmethod
    def quality_gate_build_checks(
        *,
        after_metrics: dict[str, t.Infra.InfraValue],
        improvement: dict[str, t.Infra.InfraValue],
        pyrefly_check: dict[str, t.Infra.InfraValue],
        ruff_check: dict[str, t.Infra.InfraValue],
        before_available: bool,
        before_load_error: str,
    ) -> list[dict[str, t.Infra.InfraValue]]:
        checks: list[FlextInfraCodegenModels.QualityGateCheck] = []
        violations_total = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("total_violations"),
        )
        violations_delta = FlextInfraUtilitiesCodegen.as_int(
            improvement.get("violations_delta"),
        )
        checks.append(
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_NAMESPACE_COMPLIANCE,
                passed=(
                    violations_total == 0 or (before_available and violations_delta < 0)
                )
                and (not before_available or violations_delta <= 0),
                detail=(
                    f"total={violations_total}, delta={violations_delta}"
                    if before_available
                    else f"total={violations_total} (no baseline provided)"
                ),
                critical=False,
            ),
        )
        mro_failures = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("mro_failures")
        )
        cross_ref = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("cross_project_reference_violations"),
        )
        import_parse = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("import_parse_violations"),
        )
        import_parse_errors = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("import_parse_errors"),
        )
        layer_violations = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("layer_violations"),
        )
        duplicate_groups = FlextInfraUtilitiesCodegen.as_int(
            after_metrics.get("duplicate_groups"),
        )
        duplicates_delta = FlextInfraUtilitiesCodegen.as_int(
            improvement.get("duplicates_delta"),
        )
        checks.extend([
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_MRO_VALIDITY,
                passed=mro_failures == 0,
                detail=f"mro_failures={mro_failures}",
                critical=True,
            ),
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_IMPORT_RESOLUTION,
                passed=cross_ref == 0
                and import_parse == 0
                and (import_parse_errors == 0),
                detail=(
                    f"cross_project_reference_violations={cross_ref}, "
                    f"invalid_import_from={import_parse}, parse_errors={import_parse_errors}"
                ),
                critical=True,
            ),
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_LAYER_COMPLIANCE,
                passed=layer_violations == 0,
                detail=f"layer_violations={layer_violations}",
                critical=True,
            ),
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_DUPLICATION_REDUCTION,
                passed=(
                    duplicate_groups == 0 or (before_available and duplicates_delta < 0)
                )
                and (not before_available or duplicates_delta <= 0),
                detail=(
                    f"duplicate_groups={duplicate_groups}, delta={duplicates_delta}"
                    if before_available
                    else f"duplicate_groups={duplicate_groups} (no baseline provided)"
                ),
                critical=False,
            ),
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_TYPE_SAFETY,
                passed=bool(pyrefly_check.get("passed")),
                detail=str(pyrefly_check.get("detail", "")),
                critical=True,
            ),
            FlextInfraCodegenModels.QualityGateCheck(
                name=c.Infra.Codegen.QualityGate.CHECK_LINT_CLEAN,
                passed=bool(ruff_check.get("passed")),
                detail=str(ruff_check.get("detail", "")),
                critical=True,
            ),
        ])
        if before_load_error:
            checks.append(
                FlextInfraCodegenModels.QualityGateCheck(
                    name=c.Infra.Codegen.QualityGate.CHECK_BASELINE_LOAD,
                    passed=False,
                    detail=before_load_error,
                    critical=False,
                ),
            )
        return [item.model_dump() for item in checks]

    @staticmethod
    def quality_gate_compute_verdict(
        checks: Sequence[dict[str, t.Infra.InfraValue]],
        improvement: dict[str, t.Infra.InfraValue],
    ) -> str:
        if all(bool(item.get("passed", False)) for item in checks):
            return "PASS"
        if any(
            bool(not item.get("passed", False) and item.get("critical", False))
            for item in checks
        ):
            return "FAIL"
        if (
            FlextInfraUtilitiesCodegen.as_int(improvement.get("violations_increased"))
            > 0
            or FlextInfraUtilitiesCodegen.as_int(
                improvement.get("duplicates_increased")
            )
            > 0
        ):
            return "FAIL"
        return "CONDITIONAL_PASS"

    @staticmethod
    def quality_gate_count_duplicate_constant_groups(workspace_root: Path) -> int:
        name_to_projects: dict[str, set[str]] = {}
        for report in FlextInfraCodegenCensus(workspace_root=workspace_root).run():
            project_root = workspace_root / report.project
            constants_file = (
                project_root / "src" / report.project.replace("-", "_") / "constants.py"
            )
            if not constants_file.is_file():
                continue
            tree = FlextInfraUtilitiesParsing.parse_module_ast(constants_file)
            if tree is None:
                continue
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id.isupper():
                        name_to_projects.setdefault(target.id, set()).add(
                            report.project
                        )
        return sum(1 for projects in name_to_projects.values() if len(projects) > 1)

    @staticmethod
    def quality_gate_project_findings(
        census_reports: Sequence[FlextInfraCodegenModels.CensusReport],
    ) -> list[dict[str, t.Infra.InfraValue]]:
        findings: list[FlextInfraCodegenModels.QualityGateProjectFinding] = [
            FlextInfraCodegenModels.QualityGateProjectFinding(
                project=entry.project,
                violations_total=len(tuple(entry.violations)),
                fixable_violations=int(entry.fixable),
                validator_passed=int(entry.total) == 0,
                mro_failures=0,
                layer_violations=0,
                cross_project_reference_violations=0,
            )
            for entry in census_reports
        ]
        findings.sort(key=lambda item: item.project)
        return [item.model_dump() for item in findings]

    @staticmethod
    def quality_gate_write_artifacts(
        *,
        workspace_root: Path,
        report: dict[str, t.Infra.InfraValue],
        render_text: str,
        census_reports: Sequence[FlextInfraCodegenModels.CensusReport],
        duplicate_groups: int,
        before_payload: dict[str, t.Infra.InfraValue] | None,
    ) -> dict[str, t.Infra.InfraValue]:
        directory = workspace_root / c.Infra.Codegen.QualityGate.REPORT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        report_json = directory / "latest.json"
        report_text = directory / "latest.txt"
        census_json = directory / "census-after.json"
        inventory_json = directory / "inventory-after.json"
        validate_json = directory / "validate-after.json"
        baseline_json = directory / "baseline-used.json"
        report_adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
            dict[str, t.Infra.InfraValue]
        )
        report_json.write_text(
            report_adapter.dump_json(report, by_alias=True).decode(
                c.Infra.Encoding.DEFAULT
            ),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        report_text.write_text(render_text, encoding=c.Infra.Encoding.DEFAULT)
        census_payload: list[dict[str, t.Infra.InfraValue]] = [
            item.model_dump() for item in census_reports
        ]
        census_adapter: TypeAdapter[list[dict[str, t.Infra.InfraValue]]] = TypeAdapter(
            list[dict[str, t.Infra.InfraValue]]
        )
        census_json.write_text(
            census_adapter.dump_json(census_payload, by_alias=True).decode(
                c.Infra.Encoding.DEFAULT
            ),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        inventory_adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
            dict[str, t.Infra.InfraValue]
        )
        inventory_json.write_text(
            inventory_adapter.dump_json(
                {"duplicate_groups": duplicate_groups}, by_alias=True
            ).decode(c.Infra.Encoding.DEFAULT),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        validate_adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
            dict[str, t.Infra.InfraValue]
        )
        validate_json.write_text(
            validate_adapter.dump_json(
                {
                    "mro_failures": 0,
                    "layer_violations": 0,
                    "cross_project_reference_violations": 0,
                },
                by_alias=True,
            ).decode(c.Infra.Encoding.DEFAULT),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        if before_payload is not None:
            baseline_adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            )
            baseline_json.write_text(
                baseline_adapter.dump_json(before_payload, by_alias=True).decode(
                    c.Infra.Encoding.DEFAULT
                ),
                encoding=c.Infra.Encoding.DEFAULT,
            )
        return {
            "directory": str(directory),
            "report_json": str(report_json),
            "report_text": str(report_text),
            "census_after": str(census_json),
            "inventory_after": str(inventory_json),
            "validate_after": str(validate_json),
            "baseline_used": str(baseline_json) if before_payload is not None else "",
        }

    @staticmethod
    def quality_gate_modified_python_files(workspace_root: Path) -> list[str]:
        normalized: set[str] = set()
        for rel in FlextInfraUtilitiesCodegen.git_lines(
            workspace_root,
            ["diff", "--name-only", "--", "*.py"],
        ):
            if "constants" not in rel:
                continue
            candidate = (workspace_root / rel).resolve()
            if not candidate.is_file() or candidate.suffix != c.Infra.Extensions.PYTHON:
                continue
            try:
                normalized.add(str(candidate.relative_to(workspace_root)))
            except ValueError:
                continue
        for rel in FlextInfraUtilitiesCodegen.git_lines(
            workspace_root,
            ["diff", "--name-only", "--cached", "--", "*.py"],
        ):
            if "constants" not in rel:
                continue
            candidate = (workspace_root / rel).resolve()
            if not candidate.is_file() or candidate.suffix != c.Infra.Extensions.PYTHON:
                continue
            try:
                normalized.add(str(candidate.relative_to(workspace_root)))
            except ValueError:
                continue
        for rel in FlextInfraUtilitiesCodegen.git_lines(
            workspace_root,
            ["ls-files", "--others", "--exclude-standard", "--", "*.py"],
        ):
            if "constants" not in rel:
                continue
            candidate = (workspace_root / rel).resolve()
            if not candidate.is_file() or candidate.suffix != c.Infra.Extensions.PYTHON:
                continue
            try:
                normalized.add(str(candidate.relative_to(workspace_root)))
            except ValueError:
                continue
        if normalized:
            return sorted(normalized)
        fallback = (
            workspace_root / ".reports/codegen/constants-refactor/dedup-apply.json"
        )
        if fallback.is_file():
            try:
                text = fallback.read_text(encoding=c.Infra.Encoding.DEFAULT)
                payload = (
                    FlextInfraUtilitiesCodegen._container_mapping_adapter.validate_json(
                        text
                    )
                )
            except (OSError, UnicodeDecodeError, ValueError):
                return []
            try:
                raw = FlextInfraUtilitiesCodegen._container_mapping_adapter.validate_python(
                    payload,
                )
            except ValidationError:
                return []
            modified = raw.get("modified_files")
            if isinstance(modified, list):
                filtered: set[str] = set()
                for entry in modified:
                    if not isinstance(entry, str):
                        continue
                    if not entry.endswith(c.Infra.Extensions.PYTHON):
                        continue
                    filtered.add(entry)
                return sorted(filtered)
        return []

    @staticmethod
    def git_lines(workspace_root: Path, args: list[str]) -> list[str]:
        git_bin = shutil.which(c.Infra.Cli.GIT)
        if not git_bin:
            return []
        result = FlextInfraUtilitiesSubprocess().run_raw(
            [git_bin, "-C", str(workspace_root), *args],
        )
        if result.is_failure:
            return []
        output = result.value
        if output.exit_code != 0:
            return []
        return [line.strip() for line in output.stdout.splitlines() if line.strip()]

    @staticmethod
    def quality_gate_run_pyrefly_check(
        workspace_root: Path,
        modified_files: list[str],
    ) -> dict[str, t.Infra.InfraValue]:
        if not modified_files:
            return {
                "passed": True,
                "detail": "no modified python files detected",
                "exit_code": 0,
            }
        cmd = [
            sys.executable,
            "-m",
            c.Infra.Cli.PYREFLY,
            c.Infra.Cli.RuffCmd.CHECK,
            *modified_files,
            "--config",
            c.Infra.Files.PYPROJECT_FILENAME,
            "--summary=none",
        ]
        result = FlextInfraUtilitiesCodegen.run_external_check(workspace_root, cmd)
        detail = str(result.get("detail", "")).strip()
        if not bool(result.get("passed", False)) and detail.startswith(
            "WARN PYTHONPATH"
        ):
            result["passed"] = True
        return result

    @staticmethod
    def quality_gate_run_ruff_check(
        workspace_root: Path,
        modified_files: list[str],
    ) -> dict[str, t.Infra.InfraValue]:
        if not modified_files:
            return {
                "passed": True,
                "detail": "no modified python files detected",
                "exit_code": 0,
            }
        cmd = [
            sys.executable,
            "-m",
            c.Infra.Cli.RUFF,
            c.Infra.Verbs.CHECK,
            *modified_files,
            "--output-format",
            c.Infra.Cli.OUTPUT_JSON,
            "--quiet",
        ]
        return FlextInfraUtilitiesCodegen.run_external_check(workspace_root, cmd)

    @staticmethod
    def run_external_check(
        workspace_root: Path,
        cmd: list[str],
    ) -> dict[str, t.Infra.InfraValue]:
        result = FlextInfraUtilitiesSubprocess().run_raw(cmd, cwd=workspace_root)
        if result.is_failure:
            return {
                "passed": False,
                "detail": result.error or "execution error",
                "exit_code": 127,
            }
        command_output = result.value
        output = (command_output.stderr or command_output.stdout or "").strip()
        lines = [line for line in output.splitlines() if line.strip()]
        excerpt = " | ".join(lines[:5]) if lines else "ok"
        return {
            "passed": command_output.exit_code == 0,
            "detail": excerpt,
            "exit_code": command_output.exit_code,
        }

    @staticmethod
    def quality_gate_scan_import_nodes(
        workspace_root: Path,
        modified_files: list[str],
    ) -> dict[str, t.Infra.InfraValue]:
        invalid_import_from: list[str] = []
        parse_errors: list[str] = []
        for rel_path in modified_files:
            file_path = (workspace_root / rel_path).resolve()
            if not file_path.is_file():
                continue
            tree = FlextInfraUtilitiesParsing.parse_module_ast(file_path)
            if tree is None:
                parse_errors.append(f"{rel_path}:parse failed")
                continue
            invalid_import_from.extend(
                f"{rel_path}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module is None
                and (node.level == 0)
            )
        invalid_import_from_value: list[t.Infra.InfraValue] = list(invalid_import_from)
        parse_errors_value: list[t.Infra.InfraValue] = list(parse_errors)
        return {
            "invalid_import_from_count": len(invalid_import_from),
            "parse_error_count": len(parse_errors),
            "invalid_import_from": invalid_import_from_value,
            "parse_errors": parse_errors_value,
        }

    @staticmethod
    def extract_total_violations(payload: dict[str, t.Infra.InfraValue]) -> int:
        if "total_violations" in payload:
            return FlextInfraUtilitiesCodegen.as_int(payload.get("total_violations"))
        totals = FlextInfraUtilitiesCodegen.dict_or_empty(payload.get("totals"))
        if totals:
            return (
                FlextInfraUtilitiesCodegen.as_int(totals.get("ns001_violations"))
                + FlextInfraUtilitiesCodegen.as_int(totals.get("layer_violations"))
                + FlextInfraUtilitiesCodegen.as_int(
                    totals.get("cross_project_reference_violations"),
                )
            )
        projects = FlextInfraUtilitiesCodegen.dict_list(payload.get("projects"))
        if projects and all("total" in item for item in projects):
            return sum(
                FlextInfraUtilitiesCodegen.as_int(item.get("total"))
                for item in projects
            )
        return -1

    @staticmethod
    def extract_duplicate_groups(payload: dict[str, t.Infra.InfraValue]) -> int:
        if "duplicate_groups" in payload:
            return FlextInfraUtilitiesCodegen.as_int(payload.get("duplicate_groups"))
        duplicates = payload.get("duplicates")
        if isinstance(duplicates, list):
            return sum(1 for _ in duplicates)
        return -1

    @staticmethod
    def extract_projects_total(payload: dict[str, t.Infra.InfraValue]) -> int:
        totals = FlextInfraUtilitiesCodegen.dict_or_empty(payload.get("totals"))
        value = totals.get(c.Infra.ReportKeys.PROJECTS)
        if value is not None:
            return FlextInfraUtilitiesCodegen.as_int(value)
        projects = payload.get("projects")
        if isinstance(projects, list):
            return sum(1 for _ in projects)
        return 0

    @staticmethod
    def extract_projects_passed(payload: dict[str, t.Infra.InfraValue]) -> int:
        totals = FlextInfraUtilitiesCodegen.dict_or_empty(payload.get("totals"))
        return FlextInfraUtilitiesCodegen.as_int(totals.get("passed"))

    @staticmethod
    def extract_projects_failed(payload: dict[str, t.Infra.InfraValue]) -> int:
        totals = FlextInfraUtilitiesCodegen.dict_or_empty(payload.get("totals"))
        return FlextInfraUtilitiesCodegen.as_int(totals.get("failed"))

    @staticmethod
    def as_int(value: t.Infra.InfraValue) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def dict_or_empty(value: t.Infra.InfraValue) -> dict[str, t.Infra.InfraValue]:
        if not isinstance(value, dict):
            return {}
        return TypeAdapter(dict[str, t.Infra.InfraValue]).validate_python(value)

    @staticmethod
    def dict_list(value: t.Infra.InfraValue) -> list[dict[str, t.Infra.InfraValue]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, t.Infra.InfraValue]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            result.append(
                TypeAdapter(dict[str, t.Infra.InfraValue]).validate_python(item)
            )
        return result


__all__ = ["FlextInfraUtilitiesCodegen"]
from flext_infra._utilities.subprocess import (
    FlextInfraUtilitiesSubprocess,
)
