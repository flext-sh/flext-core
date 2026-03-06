"""Refactor engine and CLI for flext_infra.refactor."""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import re
import sys
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from typing import Any, cast

import libcst as cst
import yaml

from flext_core import r
from flext_infra import c, m, output
from flext_infra.refactor.analysis import FlextInfraRefactorViolationAnalyzer
from flext_infra.refactor.rule import FlextInfraRefactorRule
from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule
from flext_infra.refactor.rules.class_reconstructor import (
    FlextInfraRefactorClassReconstructorRule,
)
from flext_infra.refactor.rules.ensure_future_annotations import (
    FlextInfraRefactorEnsureFutureAnnotationsRule,
)
from flext_infra.refactor.rules.import_modernizer import (
    FlextInfraRefactorImportModernizerRule,
)
from flext_infra.refactor.rules.legacy_removal import (
    FlextInfraRefactorLegacyRemovalRule,
)
from flext_infra.refactor.rules.mro_redundancy_checker import (
    FlextInfraRefactorMRORedundancyChecker,
)
from flext_infra.refactor.rules.pattern_corrections import (
    FlextInfraRefactorPatternCorrectionsRule,
)
from flext_infra.refactor.rules.symbol_propagation import (
    FlextInfraRefactorSignaturePropagationRule,
    FlextInfraRefactorSymbolPropagationRule,
)
from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager


class FlextInfraRefactorEngine:
    """Engine de refatoracao que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize engine state and config file path."""
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[FlextInfraRefactorRule] = []
        self.file_rules: list[ClassNestingRefactorRule] = []
        self.rule_filters: list[str] = []
        self.safety_manager = self._build_safety_manager()

    @staticmethod
    def _build_safety_manager() -> FlextInfraRefactorSafetyManager:
        return FlextInfraRefactorSafetyManager()

    @staticmethod
    def _discover_workspace_projects(workspace_root: Path) -> list[Path]:
        projects: list[Path] = []

        root_has_pyproject = (
            workspace_root / c.Infra.Files.PYPROJECT_FILENAME
        ).exists()
        root_has_makefile = (workspace_root / c.Infra.Files.MAKEFILE_FILENAME).exists()
        if root_has_pyproject and root_has_makefile:
            projects.append(workspace_root)

        for entry in sorted(workspace_root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not (entry / c.Infra.Files.PYPROJECT_FILENAME).exists():
                continue
            if not (entry / c.Infra.Files.MAKEFILE_FILENAME).exists():
                continue
            projects.append(entry)

        return projects

    @staticmethod
    def _extract_engine_file_filters(
        config_value: object,
    ) -> tuple[list[str], list[str]]:
        if not isinstance(config_value, dict):
            return [], []

        typed_config = cast("dict[str, object]", config_value)

        ignore_raw = typed_config.get("ignore_patterns", None)
        ext_raw = typed_config.get("file_extensions", None)
        ignore_patterns: list[str] = []
        if isinstance(ignore_raw, list):
            ignore_patterns.extend(
                item
                for item in cast("list[object]", ignore_raw)
                if isinstance(item, str)
            )

        extensions: list[str] = []
        if isinstance(ext_raw, list):
            extensions.extend(
                item for item in cast("list[object]", ext_raw) if isinstance(item, str)
            )
        return ignore_patterns, extensions

    @staticmethod
    def _extract_project_scan_dirs(config_value: object) -> list[str]:
        if not isinstance(config_value, dict):
            return ["src", "tests", "scripts", "examples"]

        typed_config = cast("dict[str, object]", config_value)
        scan_dirs_raw = typed_config.get("project_scan_dirs", None)
        if not isinstance(scan_dirs_raw, list):
            return ["src", "tests", "scripts", "examples"]

        scan_dirs = [
            item.strip()
            for item in cast("list[object]", scan_dirs_raw)
            if isinstance(item, str) and item.strip()
        ]
        if not scan_dirs:
            return ["src", "tests", "scripts", "examples"]
        return scan_dirs

    @staticmethod
    def _project_name_from_path(file_path: Path) -> str:
        """Infer project name from path using nearest pyproject marker."""
        for parent in file_path.parents:
            if (parent / c.Infra.Files.PYPROJECT_FILENAME).exists() and (
                parent / c.Infra.Files.MAKEFILE_FILENAME
            ).exists():
                return parent.name
        return "unknown"

    @staticmethod
    def build_impact_map(
        results: list[m.Infra.Refactor.Result],
    ) -> list[dict[str, str]]:
        """Build structured impact map from rule change messages."""
        impact_map: list[dict[str, str]] = []
        symbol_pattern = re.compile(r"^(.*):\s+(.+)\s+->\s+(.+?)(?:\s+\(|$)")
        added_pattern = re.compile(r"^\[(.+)\]\s+Added keyword:\s+(.+)$")
        removed_pattern = re.compile(r"^\[(.+)\]\s+Removed keyword:\s+(.+)$")

        for result in results:
            if not result.success:
                impact_map.append({
                    "project": FlextInfraRefactorEngine._project_name_from_path(
                        result.file_path
                    ),
                    "file": str(result.file_path),
                    "kind": "failure",
                    "old": "",
                    "new": "",
                    "status": result.error or "failed",
                })
                continue

            if not result.changes:
                continue

            project_name = FlextInfraRefactorEngine._project_name_from_path(
                result.file_path
            )
            for change in result.changes:
                symbol_match = symbol_pattern.match(change)
                if symbol_match is not None:
                    _, old_symbol, new_symbol = symbol_match.groups()
                    impact_map.append({
                        "project": project_name,
                        "file": str(result.file_path),
                        "kind": "rename",
                        "old": old_symbol.strip(),
                        "new": new_symbol.strip(),
                        "status": "changed",
                    })
                    continue

                add_match = added_pattern.match(change)
                if add_match is not None:
                    migration_id, payload = add_match.groups()
                    impact_map.append({
                        "project": project_name,
                        "file": str(result.file_path),
                        "kind": "signature_add",
                        "old": "",
                        "new": payload.strip(),
                        "status": migration_id,
                    })
                    continue

                remove_match = removed_pattern.match(change)
                if remove_match is not None:
                    migration_id, payload = remove_match.groups()
                    impact_map.append({
                        "project": project_name,
                        "file": str(result.file_path),
                        "kind": "signature_remove",
                        "old": payload.strip(),
                        "new": "",
                        "status": migration_id,
                    })

        return impact_map

    @staticmethod
    def main() -> None:
        """CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Flext Refactor Engine - Declarative code transformation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        mode_group = parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument("--project", "-p", type=Path)
        mode_group.add_argument("--workspace-root", "-w", type=Path)
        mode_group.add_argument("--file", "-f", type=Path)
        mode_group.add_argument("--files", nargs="+", type=Path)
        mode_group.add_argument("--list-rules", "-l", action="store_true")

        parser.add_argument("--rules", "-r", type=str)
        parser.add_argument("--pattern", default=c.Infra.Extensions.PYTHON_GLOB)
        parser.add_argument("--dry-run", "-n", action="store_true")
        parser.add_argument("--show-diff", "-d", action="store_true")
        parser.add_argument("--impact-map-output", type=Path)
        parser.add_argument("--analyze-violations", action="store_true")
        parser.add_argument("--analysis-output", type=Path)
        parser.add_argument("--config", "-c", type=Path)
        args = parser.parse_args()

        engine = FlextInfraRefactorEngine(config_path=args.config)
        config_result = engine.load_config()
        if not config_result.is_success:
            output.error(f"Config error: {config_result.error}")
            sys.exit(1)

        rules_result = engine.load_rules()
        if not rules_result.is_success:
            output.error(f"Rules error: {rules_result.error}")
            sys.exit(1)

        if args.list_rules:
            FlextInfraRefactorEngine.print_rules_table(engine.list_rules())
            sys.exit(0)

        if args.rules:
            rule_filters = [item.strip() for item in args.rules.split(",")]
            engine.set_rule_filters(rule_filters)
            engine.rules = []
            rules_result = engine.load_rules()
            if not rules_result.is_success:
                output.error(f"Rules error: {rules_result.error}")
                sys.exit(1)

        if args.analyze_violations:
            files_to_analyze: list[Path] = []
            if args.project:
                files_to_analyze = engine.collect_project_files(
                    args.project,
                    pattern=args.pattern,
                )
            elif args.workspace_root:
                files_to_analyze = engine.collect_workspace_files(
                    args.workspace_root,
                    pattern=args.pattern,
                )
            elif args.file:
                if not args.file.exists():
                    output.error(f"File not found: {args.file}")
                    sys.exit(1)
                files_to_analyze = [args.file]
            elif args.files:
                files_to_analyze = [item for item in args.files if item.exists()]

            analysis = FlextInfraRefactorViolationAnalyzer.analyze_files(
                files_to_analyze
            )
            FlextInfraRefactorEngine.print_violation_summary(analysis)
            if args.analysis_output is not None:
                args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
                args.analysis_output.write_text(
                    json.dumps(analysis, indent=2, ensure_ascii=True) + "\n",
                    encoding=c.Infra.Encoding.DEFAULT,
                )
                output.info(f"Analysis report written: {args.analysis_output}")
            sys.exit(0)

        results: list[m.Infra.Refactor.Result] = []
        if args.project:
            results = engine.refactor_project(
                args.project, dry_run=args.dry_run, pattern=args.pattern
            )
        elif args.workspace_root:
            results = engine.refactor_workspace(
                args.workspace_root,
                dry_run=args.dry_run,
                pattern=args.pattern,
            )
        elif args.file:
            if not args.file.exists():
                output.error(f"File not found: {args.file}")
                sys.exit(1)
            original_code = args.file.read_text(encoding=c.Infra.Encoding.DEFAULT)
            result_single = engine.refactor_file(args.file, dry_run=args.dry_run)
            results = [result_single]
            if args.show_diff and result_single.modified:
                refactored_code = result_single.refactored_code or original_code
                FlextInfraRefactorEngine.print_diff(
                    original_code, refactored_code, args.file
                )
        elif args.files:
            existing_files = [item for item in args.files if item.exists()]
            missing_files = [item for item in args.files if not item.exists()]
            for file_path in missing_files:
                output.error(f"File not found: {file_path}")
            results = engine.refactor_files(existing_files, dry_run=args.dry_run)

        FlextInfraRefactorEngine.print_summary(results, dry_run=args.dry_run)

        if args.impact_map_output is not None:
            _ = FlextInfraRefactorEngine.write_impact_map(
                results,
                args.impact_map_output,
            )
        failed = sum(1 for item in results if not item.success)
        sys.exit(0 if failed == 0 else 1)

    @staticmethod
    def print_diff(original: str, refactored: str, file_path: Path) -> None:
        """Print unified diff between original and refactored file contents."""
        output.header(f"Diff for {file_path.name}")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            refactored.splitlines(keepends=True),
            fromfile=f"{file_path.name} (original)",
            tofile=f"{file_path.name} (refactored)",
            lineterm="",
        )
        diff_text = "".join(diff)
        if diff_text:
            output.info(diff_text)
        else:
            output.info("No changes")

    @staticmethod
    def print_rules_table(rules: list[dict[str, Any]]) -> None:
        """Print rule table in terminal-friendly format."""
        output.header("Available Rules")
        if not rules:
            output.info("No rules loaded.")
            return

        id_width = max(len(item["id"]) for item in rules) + 2
        name_width = max(len(item["name"]) for item in rules) + 2
        header = (
            f"{'ID':<{id_width}} {'Name':<{name_width}} {'Severity':<10} {'Status'}"
        )
        output.info(header)
        output.info("-" * len(header))
        for rule in rules:
            status = "✓" if rule["enabled"] else "✗"
            line = f"{rule['id']:<{id_width}} {rule['name']:<{name_width}} {rule['severity']:<10} {status}"
            output.info(line)
            if rule["description"]:
                output.info(f"  - {rule['description']}")

    @staticmethod
    def print_summary(
        results: list[m.Infra.Refactor.Result],
        *,
        dry_run: bool,
    ) -> None:
        """Print refactor execution summary."""
        modified = sum(1 for item in results if item.modified)
        failed = sum(1 for item in results if not item.success)
        unchanged = sum(1 for item in results if item.success and not item.modified)

        output.header("Summary")
        output.info(f"Total files: {len(results)}")
        output.info(f"Modified: {modified}")
        output.debug(f"Unchanged: {unchanged}")
        output.info(f"Failed: {failed}")

        if dry_run:
            output.info("[DRY-RUN] No changes applied")
        elif failed == 0:
            output.info("All changes applied successfully")
        else:
            output.info(f"{failed} files failed")

    @staticmethod
    def print_violation_summary(analysis: Mapping[str, object]) -> None:
        """Print aggregate violation counts and hottest files."""

        def _to_int(value: object) -> int:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return 0

        output.header("Violation Analysis")
        totals_obj = analysis.get("totals", {})
        top_files_obj = analysis.get("top_files", [])
        files_scanned = _to_int(analysis.get("files_scanned", 0))

        output.info(f"Files scanned: {files_scanned}")
        if not isinstance(totals_obj, dict) or not totals_obj:
            output.info("No tracked violations found.")
            return

        typed_totals = cast("dict[object, object]", totals_obj)
        totals_ranked: list[tuple[str, int]] = []
        for raw_name, raw_count in typed_totals.items():
            if not isinstance(raw_name, str):
                continue
            totals_ranked.append((raw_name, _to_int(raw_count)))
        totals_ranked.sort(key=itemgetter(1), reverse=True)

        output.info("Top pattern counts:")
        for name, count in totals_ranked:
            output.info(f"  - {name}: {count}")

        if not isinstance(top_files_obj, list) or not top_files_obj:
            return
        output.info("Hottest files:")
        for entry in cast("list[object]", top_files_obj)[:10]:
            if not isinstance(entry, dict):
                continue
            typed_entry = cast("dict[object, object]", entry)
            file_name = str(typed_entry.get("file", ""))
            total_count = _to_int(typed_entry.get("total", 0))
            output.info(f"  - {file_name}: {total_count}")

    @staticmethod
    def write_impact_map(
        results: list[m.Infra.Refactor.Result],
        output_path: Path,
    ) -> bool:
        """Write impact map file in JSON format."""
        impact_map = FlextInfraRefactorEngine.build_impact_map(results)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(impact_map, indent=2, ensure_ascii=True) + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
            output.info(f"Impact map written: {output_path}")
            output.info(f"Impact map entries: {len(impact_map)}")
            return True
        except OSError as exc:
            output.error(f"Failed to write impact map {output_path}: {exc}")
            return False

    def collect_project_files(
        self,
        project_path: Path,
        *,
        pattern: str = c.Infra.Extensions.PYTHON_GLOB,
    ) -> list[Path]:
        """Collect project files that match current engine scope filters."""
        engine_config_obj = self.config.get("refactor_engine")
        scan_dirs = self._extract_project_scan_dirs(engine_config_obj)
        candidate_roots = [project_path / rel_dir for rel_dir in scan_dirs]
        existing_roots = [root for root in candidate_roots if root.exists()]

        if not existing_roots:
            output.error(
                f"No configured scan directories in {project_path}: {', '.join(scan_dirs)}"
            )
            return []

        ignore_items, extension_items = self._extract_engine_file_filters(
            engine_config_obj
        )
        ignore_patterns = {str(item) for item in ignore_items}
        allowed_extensions = {str(item) for item in extension_items}

        files: list[Path] = []
        for root_dir in existing_roots:
            for py_file in root_dir.rglob(pattern):
                relative_path = py_file.relative_to(project_path)
                relative_path_str = str(relative_path)

                if allowed_extensions and py_file.suffix not in allowed_extensions:
                    continue
                if py_file.name in ignore_patterns:
                    continue
                if any(part in ignore_patterns for part in relative_path.parts):
                    continue
                if any(
                    fnmatch.fnmatch(relative_path_str, ignore_pattern)
                    for ignore_pattern in ignore_patterns
                ):
                    continue
                files.append(py_file)
        return files

    def collect_workspace_files(
        self,
        workspace_root: Path,
        *,
        pattern: str = c.Infra.Extensions.PYTHON_GLOB,
    ) -> list[Path]:
        """Collect all candidate files under workspace projects."""
        root = workspace_root.resolve()
        project_paths = self._discover_workspace_projects(root)
        all_files: list[Path] = []
        for project in project_paths:
            all_files.extend(self.collect_project_files(project, pattern=pattern))
        return all_files

    def list_rules(self) -> list[dict[str, Any]]:
        """Return loaded rules metadata for listing."""
        return [
            {
                "id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity,
            }
            for rule in self.rules
        ]

    def load_config(self) -> r[dict[str, Any]]:
        """Load YAML configuration for this engine instance."""
        try:
            content = self.config_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            loaded = yaml.safe_load(content)
            self.config = loaded if loaded is not None else {}
            output.info(f"Loaded config from {self.config_path}")
            return r[dict[str, Any]].ok(self.config)
        except Exception as exc:
            return r[dict[str, Any]].fail(f"Failed to load config: {exc}")

    def load_rules(self) -> r[list[FlextInfraRefactorRule]]:
        """Load and instantiate enabled rules from rules directory."""
        try:
            rules_dir = self.config_path.parent / "rules"
            loaded_rules: list[FlextInfraRefactorRule] = []
            loaded_file_rules: list[ClassNestingRefactorRule] = []
            unknown_rules: list[str] = []

            for rule_file in sorted(rules_dir.glob("*.yml")):
                output.info(f"Loading rules from {rule_file.name}")
                rule_config = yaml.safe_load(
                    rule_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
                )
                if rule_config is None:
                    continue
                rules_raw = rule_config.get("rules", [])
                if not isinstance(rules_raw, list):
                    continue
                typed_rules = [
                    cast("dict[str, Any]", item)
                    for item in cast("list[object]", rules_raw)
                    if isinstance(item, dict)
                ]

                for typed_rule_def in typed_rules:
                    if "id" not in typed_rule_def:
                        continue
                    if not typed_rule_def.get("enabled", True):
                        continue

                    fix_action = (
                        str(
                            typed_rule_def.get(
                                "fix_action", typed_rule_def.get("action", "")
                            )
                        )
                        .strip()
                        .lower()
                    )
                    if fix_action == "nest_classes":
                        loaded_file_rules.append(ClassNestingRefactorRule())
                        continue

                    rule_validation = self._validate_rule_definition(typed_rule_def)
                    if rule_validation is not None:
                        unknown_rules.append(rule_validation)
                        continue

                    rule = self._build_rule(typed_rule_def)
                    if rule is None:
                        unknown_rules.append(
                            str(typed_rule_def.get("id", c.Infra.Defaults.UNKNOWN))
                        )
                        continue

                    if self.rule_filters:
                        if any(rule.matches_filter(item) for item in self.rule_filters):
                            loaded_rules.append(rule)
                    else:
                        loaded_rules.append(rule)

            if unknown_rules:
                unknown = ", ".join(sorted(unknown_rules))
                return r[list[FlextInfraRefactorRule]].fail(
                    f"Unknown rule mapping for: {unknown}"
                )

            self.rules = loaded_rules
            self.file_rules = loaded_file_rules
            output.info(f"Loaded {len(self.rules)} rules")
            if self.file_rules:
                output.info(f"Loaded {len(self.file_rules)} file rules")
            if self.rule_filters:
                output.info(f"Active filters: {', '.join(self.rule_filters)}")
            return r[list[FlextInfraRefactorRule]].ok(loaded_rules)
        except Exception as exc:
            return r[list[FlextInfraRefactorRule]].fail(f"Failed to load rules: {exc}")

    def refactor_file(
        self,
        file_path: Path,
        *,
        dry_run: bool = False,
    ) -> m.Infra.Refactor.Result:
        """Refactor one file with currently loaded rules."""
        try:
            if file_path.suffix != c.Infra.Extensions.PYTHON:
                return m.Infra.Refactor.Result(
                    file_path=file_path,
                    success=True,
                    modified=False,
                    changes=["Skipped non-Python file"],
                    refactored_code=None,
                )
            original_source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            source = original_source
            all_changes: list[str] = []
            file_rule_modified = False

            for file_rule in self.file_rules:
                file_rule_result = file_rule.apply(file_path, dry_run=True)
                if not file_rule_result.success:
                    return m.Infra.Refactor.Result(
                        file_path=file_path,
                        success=False,
                        modified=False,
                        error=file_rule_result.error,
                        changes=file_rule_result.changes,
                        refactored_code=None,
                    )
                if file_rule_result.modified and file_rule_result.refactored_code:
                    source = file_rule_result.refactored_code
                    file_rule_modified = True
                all_changes.extend(file_rule_result.changes)

            tree = cst.parse_module(source)

            for rule in self.rules:
                if rule.enabled:
                    tree, changes = rule.apply(tree, file_path)
                    all_changes.extend(changes)

            result_code = tree.code
            modified = file_rule_modified or (result_code != original_source)
            if not dry_run and modified:
                file_path.write_text(result_code, encoding=c.Infra.Encoding.DEFAULT)

            return m.Infra.Refactor.Result(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=all_changes,
                refactored_code=result_code,
            )
        except Exception as exc:
            return m.Infra.Refactor.Result(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(exc),
                changes=[],
                refactored_code=None,
            )

    def refactor_files(
        self,
        file_paths: list[Path],
        *,
        dry_run: bool = False,
    ) -> list[m.Infra.Refactor.Result]:
        """Refactor many files and collect individual results."""
        results: list[m.Infra.Refactor.Result] = []
        for file_path in file_paths:
            if file_path.suffix != c.Infra.Extensions.PYTHON:
                output.info(f"Skipped non-Python file: {file_path.name}")
                results.append(
                    m.Infra.Refactor.Result(
                        file_path=file_path,
                        success=True,
                        modified=False,
                        changes=["Skipped non-Python file"],
                        refactored_code=None,
                    )
                )
                continue
            result = self.refactor_file(file_path, dry_run=dry_run)
            results.append(result)
            if result.success:
                if result.modified:
                    output.info(
                        f"{'[DRY-RUN] ' if dry_run else ''}Modified: {file_path.name}"
                    )
                    for change in result.changes:
                        output.info(f"  - {change}")
                else:
                    output.debug(f"Unchanged: {file_path.name}")
            else:
                output.error(f"Failed: {file_path.name} - {result.error}")
        return results

    def refactor_project(
        self,
        project_path: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.Extensions.PYTHON_GLOB,
        apply_safety: bool = True,
    ) -> list[m.Infra.Refactor.Result]:
        """Refactor files under configured project directories matching the pattern."""
        stash_ref = ""
        if apply_safety and not dry_run:
            stash_result = self.safety_manager.create_pre_transformation_stash(
                project_path
            )
            if stash_result.is_failure:
                error_msg = stash_result.error or "pre-transformation stash failed"
                output.error(error_msg)
                return [
                    m.Infra.Refactor.Result(
                        file_path=project_path,
                        success=False,
                        modified=False,
                        error=error_msg,
                        changes=[],
                        refactored_code=None,
                    )
                ]
            stash_ref = stash_result.value

        files = self.collect_project_files(project_path, pattern=pattern)
        output.info(f"Found {len(files)} files to process")
        results = self.refactor_files(files, dry_run=dry_run)

        if apply_safety and not dry_run:
            checkpoint_result = self.safety_manager.save_checkpoint_state(
                project_path,
                status="transformed",
                stash_ref=stash_ref,
                processed_targets=[str(file_path) for file_path in files],
            )
            if checkpoint_result.is_failure:
                output.error(checkpoint_result.error or "checkpoint save failed")

            validation_result = self.safety_manager.run_semantic_validation(
                project_path
            )
            if validation_result.is_failure:
                error_msg = validation_result.error or "semantic validation failed"
                self.safety_manager.request_emergency_stop(error_msg)
                output.error(error_msg)
                rollback_result = self.safety_manager.rollback(project_path, stash_ref)
                if rollback_result.is_failure:
                    output.error(rollback_result.error or "rollback failed")
                results.append(
                    m.Infra.Refactor.Result(
                        file_path=project_path,
                        success=False,
                        modified=False,
                        error=error_msg,
                        changes=[],
                        refactored_code=None,
                    )
                )
            else:
                clear_result = self.safety_manager.clear_checkpoint()
                if clear_result.is_failure:
                    output.error(clear_result.error or "checkpoint clear failed")

        return results

    def refactor_workspace(
        self,
        workspace_root: Path,
        *,
        dry_run: bool = False,
        pattern: str = c.Infra.Extensions.PYTHON_GLOB,
        apply_safety: bool = True,
    ) -> list[m.Infra.Refactor.Result]:
        """Refactor all discoverable workspace projects with one command."""
        root = workspace_root.resolve()
        if not root.exists() or not root.is_dir():
            output.error(f"Invalid workspace root: {workspace_root}")
            return []

        project_paths = self._discover_workspace_projects(root)
        if not project_paths:
            output.error(
                f"No projects discovered under workspace root: {workspace_root}"
            )
            return []

        output.info(f"Discovered {len(project_paths)} projects in workspace")
        results: list[m.Infra.Refactor.Result] = []
        processed_targets: list[str] = []
        stash_ref = ""

        if apply_safety and not dry_run:
            stash_result = self.safety_manager.create_pre_transformation_stash(root)
            if stash_result.is_failure:
                error_msg = stash_result.error or "pre-transformation stash failed"
                output.error(error_msg)
                return [
                    m.Infra.Refactor.Result(
                        file_path=root,
                        success=False,
                        modified=False,
                        error=error_msg,
                        changes=[],
                        refactored_code=None,
                    )
                ]
            stash_ref = stash_result.value

        for project in project_paths:
            if apply_safety and self.safety_manager.is_emergency_stop_requested():
                break
            output.header(f"Project: {project}")
            project_results = self.refactor_project(
                project,
                dry_run=dry_run,
                pattern=pattern,
                apply_safety=False,
            )
            results.extend(project_results)

            if apply_safety and not dry_run:
                processed_targets.append(str(project))
                checkpoint_result = self.safety_manager.save_checkpoint_state(
                    root,
                    status="running",
                    stash_ref=stash_ref,
                    processed_targets=list(processed_targets),
                )
                if checkpoint_result.is_failure:
                    output.error(checkpoint_result.error or "checkpoint save failed")

        if apply_safety and not dry_run:
            validation_result = self.safety_manager.run_semantic_validation(root)
            if validation_result.is_failure:
                error_msg = validation_result.error or "semantic validation failed"
                self.safety_manager.request_emergency_stop(error_msg)
                output.error(error_msg)
                rollback_result = self.safety_manager.rollback(root, stash_ref)
                if rollback_result.is_failure:
                    output.error(rollback_result.error or "rollback failed")
                results.append(
                    m.Infra.Refactor.Result(
                        file_path=root,
                        success=False,
                        modified=False,
                        error=error_msg,
                        changes=[],
                        refactored_code=None,
                    )
                )
            else:
                clear_result = self.safety_manager.clear_checkpoint()
                if clear_result.is_failure:
                    output.error(clear_result.error or "checkpoint clear failed")

        return results

    def set_rule_filters(self, filters: list[str]) -> None:
        """Set active rule filters used while loading rules."""
        self.rule_filters = [item.lower() for item in filters]

    def _build_rule(
        self,
        rule_def: Mapping[str, Any],
    ) -> FlextInfraRefactorRule | None:
        rule_id = str(rule_def.get("id", c.Infra.Defaults.UNKNOWN))
        fix_action = (
            str(rule_def.get("fix_action", rule_def.get("action", ""))).strip().lower()
        )
        check = str(rule_def.get("check", "")).strip().lower()

        if (
            fix_action in c.Infra.Refactor.FUTURE_FIX_ACTIONS
            or check in c.Infra.Refactor.FUTURE_CHECKS
        ):
            return FlextInfraRefactorEnsureFutureAnnotationsRule(rule_def)
        if fix_action in c.Infra.Refactor.LEGACY_FIX_ACTIONS:
            return FlextInfraRefactorLegacyRemovalRule(rule_def)
        if fix_action in c.Infra.Refactor.IMPORT_FIX_ACTIONS:
            return FlextInfraRefactorImportModernizerRule(rule_def)
        if fix_action in c.Infra.Refactor.CLASS_FIX_ACTIONS:
            return FlextInfraRefactorClassReconstructorRule(rule_def)
        if fix_action in c.Infra.Refactor.MRO_FIX_ACTIONS:
            return FlextInfraRefactorMRORedundancyChecker(rule_def)
        if fix_action in c.Infra.Refactor.PROPAGATION_FIX_ACTIONS:
            if fix_action == "propagate_signature_migrations":
                return FlextInfraRefactorSignaturePropagationRule(rule_def)
            return FlextInfraRefactorSymbolPropagationRule(rule_def)
        if fix_action in c.Infra.Refactor.PATTERN_FIX_ACTIONS:
            return FlextInfraRefactorPatternCorrectionsRule(rule_def)

        rule_id_lower = rule_id.lower()
        if "ensure-future" in rule_id_lower or "future-annotations" in rule_id_lower:
            return FlextInfraRefactorEnsureFutureAnnotationsRule(rule_def)
        if any(
            key in rule_id_lower
            for key in ["legacy", "alias", "deprecated", "wrapper", "bypass"]
        ):
            return FlextInfraRefactorLegacyRemovalRule(rule_def)
        if any(key in rule_id_lower for key in ["import", "modernize"]):
            return FlextInfraRefactorImportModernizerRule(rule_def)
        if any(key in rule_id_lower for key in ["class", "reorder", "method"]):
            return FlextInfraRefactorClassReconstructorRule(rule_def)
        if "mro" in rule_id_lower:
            return FlextInfraRefactorMRORedundancyChecker(rule_def)
        if any(
            key in rule_id_lower for key in ["propagate", "symbol-rename", "rename"]
        ):
            if "signature" in rule_id_lower:
                return FlextInfraRefactorSignaturePropagationRule(rule_def)
            return FlextInfraRefactorSymbolPropagationRule(rule_def)
        if any(
            key in rule_id_lower
            for key in ["redundant-cast", "dict-to-mapping", "container-invariance"]
        ):
            return FlextInfraRefactorPatternCorrectionsRule(rule_def)
        return None

    def _default_config_path(self) -> Path:
        return Path(__file__).parent / "config.yml"

    def _validate_rule_definition(self, rule_def: Mapping[str, Any]) -> str | None:
        """Return validation error for malformed declarative rules."""
        rule_id = str(rule_def.get("id", c.Infra.Defaults.UNKNOWN))
        fix_action = str(rule_def.get("fix_action", "")).strip().lower()
        if not fix_action:
            return None

        if fix_action in c.Infra.Refactor.PROPAGATION_FIX_ACTIONS:
            if fix_action == "propagate_symbol_renames" and not isinstance(
                rule_def.get("import_symbol_renames"), dict
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


__all__ = ["FlextInfraRefactorEngine"]
