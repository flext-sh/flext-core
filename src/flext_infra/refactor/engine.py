"""Refactor engine and CLI for flext_infra.refactor."""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import libcst as cst
import yaml

from flext_core import r
from flext_infra import c
from flext_infra.output import output
from flext_infra.refactor.result import FlextInfraRefactorResult
from flext_infra.refactor.rule import FlextInfraRefactorRule
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
from flext_infra.refactor.rules.symbol_propagation import (
    FlextInfraRefactorSignaturePropagationRule,
    FlextInfraRefactorSymbolPropagationRule,
)


class FlextInfraRefactorEngine:
    """Engine de refatoracao que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize engine state and config file path."""
        self.config_path = config_path or self._default_config_path()
        self.config: dict[str, Any] = {}
        self.rules: list[FlextInfraRefactorRule] = []
        self.rule_filters: list[str] = []

    def _default_config_path(self) -> Path:
        return Path(__file__).parent / "config.yml"

    def set_rule_filters(self, filters: list[str]) -> None:
        """Set active rule filters used while loading rules."""
        self.rule_filters = [item.lower() for item in filters]

    def load_config(self) -> r[dict[str, Any]]:
        """Load YAML configuration for this engine instance."""
        try:
            content = self.config_path.read_text(encoding="utf-8")
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
            unknown_rules: list[str] = []

            for rule_file in sorted(rules_dir.glob("*.yml")):
                output.info(f"Loading rules from {rule_file.name}")
                rule_config = yaml.safe_load(rule_file.read_text(encoding="utf-8"))
                if rule_config is None:
                    continue
                rules = rule_config.get("rules", [])

                for rule_def in rules:
                    if not rule_def.get("enabled", True):
                        continue

                    rule = self._build_rule(rule_def)
                    if rule is None:
                        unknown_rules.append(str(rule_def.get("id", "unknown")))
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
            output.info(f"Loaded {len(self.rules)} rules")
            if self.rule_filters:
                output.info(f"Active filters: {', '.join(self.rule_filters)}")
            return r[list[FlextInfraRefactorRule]].ok(loaded_rules)
        except Exception as exc:
            return r[list[FlextInfraRefactorRule]].fail(f"Failed to load rules: {exc}")

    def _build_rule(
        self,
        rule_def: dict[str, Any],
    ) -> FlextInfraRefactorRule | None:
        rule_id = str(rule_def.get("id", "unknown"))
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
        return None

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

    def refactor_file(
        self,
        file_path: Path,
        *,
        dry_run: bool = False,
    ) -> FlextInfraRefactorResult:
        """Refactor one file with currently loaded rules."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = cst.parse_module(source)
            all_changes: list[str] = []

            for rule in self.rules:
                if rule.enabled:
                    tree, changes = rule.apply(tree, file_path)
                    all_changes.extend(changes)

            result_code = tree.code
            modified = result_code != source
            if not dry_run and modified:
                file_path.write_text(result_code, encoding="utf-8")

            return FlextInfraRefactorResult(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=all_changes,
                refactored_code=result_code,
            )
        except Exception as exc:
            return FlextInfraRefactorResult(
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
    ) -> list[FlextInfraRefactorResult]:
        """Refactor many files and collect individual results."""
        results: list[FlextInfraRefactorResult] = []
        for file_path in file_paths:
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
                    output.info(f"Unchanged: {file_path.name}")
            else:
                output.error(f"Failed: {file_path.name} - {result.error}")
        return results

    def refactor_project(
        self,
        project_path: Path,
        *,
        dry_run: bool = False,
        pattern: str = "*.py",
    ) -> list[FlextInfraRefactorResult]:
        """Refactor files under configured project directories matching the pattern."""
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

        output.info(f"Found {len(files)} files to process")
        return self.refactor_files(files, dry_run=dry_run)

    def refactor_workspace(
        self,
        workspace_root: Path,
        *,
        dry_run: bool = False,
        pattern: str = "*.py",
    ) -> list[FlextInfraRefactorResult]:
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
        results: list[FlextInfraRefactorResult] = []
        for project in project_paths:
            output.header(f"Project: {project}")
            results.extend(
                self.refactor_project(project, dry_run=dry_run, pattern=pattern)
            )
        return results

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
    def print_summary(
        results: list[FlextInfraRefactorResult],
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
        output.info(f"Unchanged: {unchanged}")
        output.info(f"Failed: {failed}")

        if dry_run:
            output.info("[DRY-RUN] No changes applied")
        elif failed == 0:
            output.info("All changes applied successfully")
        else:
            output.info(f"{failed} files failed")

    @staticmethod
    def build_impact_map(
        results: list[FlextInfraRefactorResult],
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
    def write_impact_map(
        results: list[FlextInfraRefactorResult],
        output_path: Path,
    ) -> bool:
        """Write impact map file in JSON format."""
        impact_map = FlextInfraRefactorEngine.build_impact_map(results)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(impact_map, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            output.info(f"Impact map written: {output_path}")
            output.info(f"Impact map entries: {len(impact_map)}")
            return True
        except OSError as exc:
            output.error(f"Failed to write impact map {output_path}: {exc}")
            return False

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
    def _discover_workspace_projects(workspace_root: Path) -> list[Path]:
        projects: list[Path] = []

        root_has_pyproject = (workspace_root / "pyproject.toml").exists()
        root_has_makefile = (workspace_root / "Makefile").exists()
        if root_has_pyproject and root_has_makefile:
            projects.append(workspace_root)

        for entry in sorted(workspace_root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not (entry / "pyproject.toml").exists():
                continue
            if not (entry / "Makefile").exists():
                continue
            projects.append(entry)

        return projects

    @staticmethod
    def _project_name_from_path(file_path: Path) -> str:
        """Infer project name from path using nearest pyproject marker."""
        for parent in file_path.parents:
            if (parent / "pyproject.toml").exists() and (parent / "Makefile").exists():
                return parent.name
        return "unknown"

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
        parser.add_argument("--pattern", default="*.py")
        parser.add_argument("--dry-run", "-n", action="store_true")
        parser.add_argument("--show-diff", "-d", action="store_true")
        parser.add_argument("--impact-map-output", type=Path)
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

        results: list[FlextInfraRefactorResult] = []
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
            original_code = args.file.read_text(encoding="utf-8")
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


__all__ = ["FlextInfraRefactorEngine"]
