"""Post-transformation validation gates for refactor results."""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import re
import sys
from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import orjson
from pydantic import JsonValue

from flext_infra import c, m, t, u
from flext_infra.refactor.analysis import FlextInfraRefactorViolationAnalyzer
from flext_infra.refactor.mro_resolver import FlextInfraRefactorMROMigrationScanner

if TYPE_CHECKING:
    from flext_infra.refactor.engine import FlextInfraRefactorEngine


class PostCheckGate:
    """Validate refactor results against policy expectations."""

    def __init__(self) -> None:
        """Initialize gate."""

    def validate(
        self,
        result: m.Infra.Refactor.Result,
        expected: t.Infra.ContainerDict,
    ) -> tuple[bool, list[str]]:
        """Validate a refactor result against expected post-checks and gates."""
        errors: list[str] = []
        if not result.success:
            if result.error:
                return (False, [result.error])
            return (False, ["transform_failed"])
        if not result.modified:
            return (True, [])
        file_path = result.file_path
        post_checks = u.Infra.string_list(
            expected.get(c.Infra.ReportKeys.POST_CHECKS),
        )
        quality_gates = u.Infra.string_list(expected.get("quality_gates"))
        if self._check_enabled("imports_resolve", post_checks):
            errors.extend(self._validate_imports(file_path))
        source_symbol_raw = expected.get(c.Infra.ReportKeys.SOURCE_SYMBOL, "")
        source_symbol = source_symbol_raw if isinstance(source_symbol_raw, str) else ""
        expected_chain = u.Infra.string_list(
            expected.get("expected_base_chain"),
        )
        if (
            source_symbol
            and expected_chain
            and self._check_enabled("mro_valid", post_checks)
        ):
            errors.extend(self._validate_mro(file_path, source_symbol, expected_chain))
        if self._check_enabled("lsp_diagnostics_clean", quality_gates):
            errors.extend(self._validate_types(file_path))
        return (len(errors) == 0, errors)

    def _check_enabled(self, check_name: str, checks: list[str]) -> bool:
        return check_name in checks

    def _validate_imports(self, file_path: Path) -> list[str]:
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return [f"parse_error:{file_path}:parse_failed"]
        unresolved: list[str] = [
            f"line_{node.lineno}:invalid_import_from"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is None
            and (node.level == 0)
        ]
        return unresolved

    def _validate_mro(
        self,
        file_path: Path,
        class_name: str,
        expected_bases: Sequence[str],
    ) -> list[str]:
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return [f"mro_parse_error:{file_path}:parse_failed"]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                actual = [self._base_name(base) for base in node.bases]
                actual_clean = [name for name in actual if name]
                expected_prefix = list(expected_bases)[: len(actual_clean)]
                if actual_clean != expected_prefix:
                    return [
                        f"mro_mismatch:{class_name}:expected={expected_prefix}:actual={actual_clean}",
                    ]
                return []
        return [f"class_not_found:{class_name}"]

    def _validate_types(self, file_path: Path) -> list[str]:
        """Check that the file compiles without syntax errors."""
        cmd = [sys.executable, "-m", "py_compile", str(file_path)]
        result = u.Infra.capture_output(cmd)
        return result.fold(
            on_failure=lambda e: [f"lsp_diagnostics_clean_failed:{e or ''}"],
            on_success=lambda _: [],
        )

    def _base_name(self, base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Subscript):
            return self._base_name(base.value)
        return ""


class FlextInfraRefactorRuleDefinitionValidator:
    """Validate individual refactor rule definitions for correctness."""

    def validate_rule_definition(
        self,
        rule_def: Mapping[str, JsonValue],
    ) -> str | None:
        """Validate a rule definition and return error message if invalid."""
        rule_id = str(rule_def.get(c.Infra.ReportKeys.ID, c.Infra.Defaults.UNKNOWN))
        fix_action = (
            str(rule_def.get(c.Infra.ReportKeys.FIX_ACTION, "")).strip().lower()
        )
        if not fix_action:
            return None
        if fix_action in c.Infra.Refactor.PROPAGATION_FIX_ACTIONS:
            if fix_action == "propagate_symbol_renames" and (
                not isinstance(rule_def.get("import_symbol_renames"), dict)
            ):
                return f"{rule_id}: import_symbol_renames must be a mapping"
            if fix_action == "propagate_signature_migrations":
                migrations = rule_def.get("signature_migrations")
                if not isinstance(migrations, list) or not migrations:
                    return f"{rule_id}: signature_migrations must be a non-empty list"
        if fix_action == "remove_redundant_casts":
            targets = rule_def.get("redundant_type_targets")
            if not isinstance(targets, list) or not targets:
                return f"{rule_id}: redundant_type_targets must be a non-empty list"
        return None


class FlextInfraRefactorCliSupport:
    """CLI support utilities for the refactor engine."""

    @staticmethod
    def info(message: str) -> None:
        """Write an informational message to stdout."""
        _ = sys.stdout.write(f"{message}\n")

    @staticmethod
    def error(message: str) -> None:
        """Write an error message to stderr."""
        _ = sys.stderr.write(f"ERROR: {message}\n")

    @staticmethod
    def header(message: str) -> None:
        """Write a section header to stdout."""
        _ = sys.stdout.write(f"\n{message}\n")

    @staticmethod
    def debug(message: str) -> None:
        """Write a debug message to stdout."""
        _ = sys.stdout.write(f"DEBUG: {message}\n")

    @staticmethod
    def project_name_from_path(file_path: Path) -> str:
        """Extract project name from a file path by locating pyproject.toml."""
        for parent in file_path.parents:
            if (parent / c.Infra.Files.PYPROJECT_FILENAME).exists() and (
                parent / c.Infra.Files.MAKEFILE_FILENAME
            ).exists():
                return parent.name
        return c.Infra.Defaults.UNKNOWN

    @staticmethod
    def build_impact_map(
        results: list[m.Infra.Refactor.Result],
    ) -> list[dict[str, str]]:
        """Build an impact map from refactor results for reporting."""
        impact_map: list[dict[str, str]] = []
        symbol_pattern = re.compile(r"^(.*):\s+(.+)\s+->\s+(.+?)(?:\s+\(|$)")
        added_pattern = re.compile(r"^\[(.+)\]\s+Added keyword:\s+(.+)$")
        removed_pattern = re.compile(r"^\[(.+)\]\s+Removed keyword:\s+(.+)$")
        for result in results:
            if not result.success:
                impact_map.append({
                    c.Infra.Toml.PROJECT: FlextInfraRefactorCliSupport.project_name_from_path(
                        result.file_path,
                    ),
                    c.Infra.ReportKeys.FILE: str(result.file_path),
                    "kind": "failure",
                    "old": "",
                    "new": "",
                    c.Infra.ReportKeys.STATUS: result.error or "failed",
                })
                continue
            if not result.changes:
                continue
            project_name = FlextInfraRefactorCliSupport.project_name_from_path(
                result.file_path,
            )
            for change in result.changes:
                symbol_match = symbol_pattern.match(change)
                if symbol_match is not None:
                    _, old_symbol, new_symbol = symbol_match.groups()
                    impact_map.append({
                        c.Infra.Toml.PROJECT: project_name,
                        c.Infra.ReportKeys.FILE: str(result.file_path),
                        "kind": "rename",
                        "old": old_symbol.strip(),
                        "new": new_symbol.strip(),
                        c.Infra.ReportKeys.STATUS: "changed",
                    })
                    continue
                add_match = added_pattern.match(change)
                if add_match is not None:
                    migration_id, payload = add_match.groups()
                    impact_map.append({
                        c.Infra.Toml.PROJECT: project_name,
                        c.Infra.ReportKeys.FILE: str(result.file_path),
                        "kind": "signature_add",
                        "old": "",
                        "new": payload.strip(),
                        c.Infra.ReportKeys.STATUS: migration_id,
                    })
                    continue
                remove_match = removed_pattern.match(change)
                if remove_match is not None:
                    migration_id, payload = remove_match.groups()
                    impact_map.append({
                        c.Infra.Toml.PROJECT: project_name,
                        c.Infra.ReportKeys.FILE: str(result.file_path),
                        "kind": "signature_remove",
                        "old": payload.strip(),
                        "new": "",
                        c.Infra.ReportKeys.STATUS: migration_id,
                    })
        return impact_map

    @staticmethod
    def write_impact_map(
        results: list[m.Infra.Refactor.Result],
        output_path: Path,
    ) -> bool:
        """Write impact map to a JSON file."""
        impact_map = FlextInfraRefactorCliSupport.build_impact_map(results)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                orjson.dumps(impact_map, option=orjson.OPT_INDENT_2).decode() + "\n"
            )
            _ = output_path.write_text(payload, encoding=c.Infra.Encoding.DEFAULT)
            FlextInfraRefactorCliSupport.info(f"Impact map written: {output_path}")
            FlextInfraRefactorCliSupport.info(f"Impact map entries: {len(impact_map)}")
            return True
        except OSError:
            FlextInfraRefactorCliSupport.error(
                f"Failed to write impact map {output_path}",
            )
            return False

    @staticmethod
    def print_diff(original: str, refactored: str, file_path: Path) -> None:
        """Print a unified diff between original and refactored code."""
        FlextInfraRefactorCliSupport.header(f"Diff for {file_path.name}")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True),
            fromfile=f"{file_path.name} (original)",
            tofile=f"{file_path.name} (refactored)",
            lineterm="",
        )
        diff_text = "".join(diff)
        if diff_text:
            FlextInfraRefactorCliSupport.info(diff_text)
            return
        FlextInfraRefactorCliSupport.info("No changes")

    @staticmethod
    def print_rules_table(rules: list[dict[str, str | bool]]) -> None:
        """Print a formatted table of available refactor rules."""
        FlextInfraRefactorCliSupport.header("Available Rules")
        if not rules:
            FlextInfraRefactorCliSupport.info("No rules loaded.")
            return
        id_width = (
            max(
                len(str(item.get(c.Infra.ReportKeys.ID, c.Infra.Defaults.UNKNOWN)))
                for item in rules
            )
            + 2
        )
        name_width = (
            max(
                len(str(item.get(c.Infra.Toml.NAME, c.Infra.Defaults.UNKNOWN)))
                for item in rules
            )
            + 2
        )
        header = (
            f"{'ID':<{id_width}} {'Name':<{name_width}} {'Severity':<10} {'Status'}"
        )
        FlextInfraRefactorCliSupport.info(header)
        FlextInfraRefactorCliSupport.info("-" * len(header))
        for rule in rules:
            status = "✓" if rule[c.Infra.ReportKeys.ENABLED] else "✗"
            rid = f"{rule['id']:<{id_width}}"
            rname = f"{rule['name']:<{name_width}}"
            rsev = f"{rule['severity']:<10}"
            FlextInfraRefactorCliSupport.info(f"{rid} {rname} {rsev} {status}")
            if rule["description"]:
                FlextInfraRefactorCliSupport.info(f"  - {rule['description']}")

    @staticmethod
    def print_summary(results: list[m.Infra.Refactor.Result], *, dry_run: bool) -> None:
        """Print a summary of refactor results."""
        modified = sum(1 for item in results if item.modified)
        failed = sum(1 for item in results if not item.success)
        unchanged = sum(1 for item in results if item.success and (not item.modified))
        FlextInfraRefactorCliSupport.header("Summary")
        FlextInfraRefactorCliSupport.info(f"Total files: {len(results)}")
        FlextInfraRefactorCliSupport.info(f"Modified: {modified}")
        FlextInfraRefactorCliSupport.debug(f"Unchanged: {unchanged}")
        FlextInfraRefactorCliSupport.info(f"Failed: {failed}")
        if dry_run:
            FlextInfraRefactorCliSupport.info("[DRY-RUN] No changes applied")
        elif failed == 0:
            FlextInfraRefactorCliSupport.info("All changes applied successfully")
        else:
            FlextInfraRefactorCliSupport.info(f"{failed} files failed")

    @staticmethod
    def print_violation_summary(
        analysis: m.Infra.Refactor.ViolationAnalysisReport,
    ) -> None:
        """Print a summary of violation analysis results."""
        FlextInfraRefactorCliSupport.header("Violation Analysis")
        FlextInfraRefactorCliSupport.info(f"Files scanned: {analysis.files_scanned}")
        if not analysis.totals:
            FlextInfraRefactorCliSupport.info("No tracked violations found.")
            return
        totals_ranked = sorted(analysis.totals.items(), key=itemgetter(1), reverse=True)
        FlextInfraRefactorCliSupport.info("Top pattern counts:")
        for name, count in totals_ranked:
            FlextInfraRefactorCliSupport.info(f"  - {name}: {count}")
        if not analysis.top_files:
            return
        FlextInfraRefactorCliSupport.info("Hottest files:")
        for entry in analysis.top_files[:10]:
            FlextInfraRefactorCliSupport.info(f"  - {entry.file}: {entry.total}")

    @staticmethod
    def run_cli(engine_cls: type[FlextInfraRefactorEngine]) -> int:
        """Run the refactor engine CLI with argument parsing."""
        parser = argparse.ArgumentParser(
            description="Flext Refactor Engine - Declarative code transformation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        mode_group = parser.add_mutually_exclusive_group(required=True)
        _ = mode_group.add_argument("--project", "-p", type=Path)
        _ = mode_group.add_argument("--workspace", "-w", type=Path)
        _ = mode_group.add_argument("--file", "-f", type=Path)
        _ = mode_group.add_argument("--files", nargs="+", type=Path)
        _ = mode_group.add_argument("--list-rules", "-l", action="store_true")
        _ = parser.add_argument("--rules", "-r", type=str)
        _ = parser.add_argument("--pattern", default=c.Infra.Extensions.PYTHON_GLOB)
        _ = parser.add_argument("--dry-run", "-n", action="store_true")
        _ = parser.add_argument("--show-diff", "-d", action="store_true")
        _ = parser.add_argument("--impact-map-output", type=Path)
        _ = parser.add_argument("--analyze-violations", action="store_true")
        _ = parser.add_argument("--analysis-output", type=Path)
        _ = parser.add_argument("--config", "-c", type=Path)
        args = parser.parse_args()
        engine = engine_cls(config_path=args.config)
        config_result = engine.load_config()
        if not config_result.is_success:
            FlextInfraRefactorCliSupport.error(f"Config error: {config_result.error}")
            return 1
        if args.rules:
            rule_filters = [
                item.strip() for item in args.rules.split(",") if item.strip()
            ]
            engine.set_rule_filters(rule_filters)
        rules_result = engine.load_rules()
        if not rules_result.is_success:
            FlextInfraRefactorCliSupport.error(f"Rules error: {rules_result.error}")
            return 1
        if args.list_rules:
            FlextInfraRefactorCliSupport.print_rules_table(engine.list_rules())
            return 0
        if args.analyze_violations:
            files_to_analyze: list[Path] = []
            if args.project:
                scan_dirs = frozenset(
                    engine.rule_loader.extract_project_scan_dirs(engine.config),
                )
                iter_result = u.Infra.iter_python_files(
                    workspace_root=args.project,
                    project_roots=[args.project],
                    include_tests=c.Infra.Directories.TESTS in scan_dirs,
                    include_examples=c.Infra.Directories.EXAMPLES in scan_dirs,
                    include_scripts=c.Infra.Directories.SCRIPTS in scan_dirs,
                    src_dirs=scan_dirs or None,
                )
                if iter_result.is_failure:
                    FlextInfraRefactorCliSupport.error(
                        iter_result.error
                        or f"File iteration failed for project: {args.project}"
                    )
                    return 1
                ignore_items, extension_items = (
                    engine.rule_loader.extract_engine_file_filters(
                        engine.config,
                    )
                )
                ignore_patterns = {str(item) for item in ignore_items}
                allowed_extensions = {str(item) for item in extension_items}
                files_to_analyze = [
                    file_path
                    for file_path in iter_result.value
                    if (
                        fnmatch.fnmatch(
                            str(file_path.relative_to(args.project)), args.pattern
                        )
                        or fnmatch.fnmatch(file_path.name, args.pattern)
                    )
                    and (
                        not allowed_extensions or file_path.suffix in allowed_extensions
                    )
                    and file_path.name not in ignore_patterns
                    and not any(
                        part in ignore_patterns
                        for part in file_path.relative_to(args.project).parts
                    )
                    and not any(
                        fnmatch.fnmatch(
                            str(file_path.relative_to(args.project)),
                            ignore_pattern,
                        )
                        for ignore_pattern in ignore_patterns
                    )
                ]
            elif args.workspace:
                files_to_analyze = engine.collect_workspace_files(
                    args.workspace,
                    pattern=args.pattern,
                )
            elif args.file:
                if not args.file.exists():
                    FlextInfraRefactorCliSupport.error(f"File not found: {args.file}")
                    return 1
                files_to_analyze = [args.file]
            elif args.files:
                files_to_analyze = [item for item in args.files if item.exists()]
            analysis = FlextInfraRefactorViolationAnalyzer.analyze_files(
                files_to_analyze,
            )
            FlextInfraRefactorCliSupport.print_violation_summary(analysis)
            if args.analysis_output is not None:
                _ = u.Infra.write_json(
                    args.analysis_output,
                    analysis.model_dump(mode="json"),
                    ensure_ascii=True,
                )
                FlextInfraRefactorCliSupport.info(
                    f"Analysis report written: {args.analysis_output}",
                )
            sys.exit(0)
        results: list[m.Infra.Refactor.Result] = []
        if args.project:
            results = engine.refactor_project(
                args.project,
                dry_run=args.dry_run,
                pattern=args.pattern,
            )
        elif args.workspace:
            results = engine.refactor_workspace(
                args.workspace,
                dry_run=args.dry_run,
                pattern=args.pattern,
            )
        elif args.file:
            if not args.file.exists():
                FlextInfraRefactorCliSupport.error(f"File not found: {args.file}")
                return 1
            original_code = args.file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            result_single = engine.refactor_file(args.file, dry_run=args.dry_run)
            results = [result_single]
            if args.show_diff and result_single.modified:
                refactored_code = result_single.refactored_code or original_code
                FlextInfraRefactorCliSupport.print_diff(
                    original_code,
                    refactored_code,
                    args.file,
                )
        elif args.files:
            existing_files = [item for item in args.files if item.exists()]
            missing_files = [item for item in args.files if not item.exists()]
            for file_path in missing_files:
                FlextInfraRefactorCliSupport.error(f"File not found: {file_path}")
            results = engine.refactor_files(existing_files, dry_run=args.dry_run)
        FlextInfraRefactorCliSupport.print_summary(results, dry_run=args.dry_run)
        if args.impact_map_output is not None:
            _ = FlextInfraRefactorCliSupport.write_impact_map(
                results,
                args.impact_map_output,
            )
        failed = sum(1 for item in results if not item.success)
        return 0 if failed == 0 else 1


class FlextInfraRefactorMROMigrationValidator:
    """Validate MRO migration completeness across a workspace."""

    @classmethod
    def validate(cls, *, workspace_root: Path, target: str) -> tuple[int, int]:
        """Validate migration status and return remaining candidates count."""
        file_results, _ = FlextInfraRefactorMROMigrationScanner.scan_workspace(
            workspace_root=workspace_root,
            target=target,
        )
        remaining = sum(len(item.candidates) for item in file_results)
        return (remaining, 0)


__all__ = [
    "FlextInfraRefactorCliSupport",
    "FlextInfraRefactorMROMigrationValidator",
    "FlextInfraRefactorRuleDefinitionValidator",
    "PostCheckGate",
]
