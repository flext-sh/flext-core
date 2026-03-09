"""Refactor engine and CLI for flext_infra.refactor."""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import libcst as cst

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.refactor.rule import (
    FlextInfraRefactorRule,
    FlextInfraRefactorRuleLoader,
)
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
from flext_infra.refactor.rules.mro_class_migration import (
    FlextInfraRefactorMROClassMigrationRule,
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
from flext_infra.refactor.validation import (
    FlextInfraRefactorCliSupport,
    FlextInfraRefactorRuleDefinitionValidator,
)

output = FlextInfraRefactorCliSupport


class FlextInfraRefactorEngine:
    """Engine de refatoracao que orquestra regras declarativas."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize engine state and config file path."""
        self.config_path = config_path or self._default_config_path()
        self.config: t.ConfigurationMapping = {}
        self.rules: list[FlextInfraRefactorRule] = []
        self.file_rules: list[ClassNestingRefactorRule] = []
        self.rule_filters: list[str] = []
        self.rule_loader = FlextInfraRefactorRuleLoader(self.config_path)
        self.rule_validator = FlextInfraRefactorRuleDefinitionValidator()
        self.safety_manager = self._build_safety_manager()

    @staticmethod
    def _build_safety_manager() -> FlextInfraRefactorSafetyManager:
        return FlextInfraRefactorSafetyManager()

    @staticmethod
    def build_impact_map(
        results: list[m.Infra.Refactor.Result],
    ) -> list[dict[str, str]]:
        """Build a normalized impact-map payload from refactor results."""
        return FlextInfraRefactorCliSupport.build_impact_map(results)

    @staticmethod
    def write_impact_map(
        results: list[m.Infra.Refactor.Result],
        output_path: Path,
    ) -> bool:
        """Write the impact-map payload to the target output path."""
        return FlextInfraRefactorCliSupport.write_impact_map(results, output_path)

    @classmethod
    def main(cls) -> None:
        """Run the refactor CLI entrypoint and exit with status code."""
        runner: Callable[[type], int] = FlextInfraRefactorCliSupport.run_cli
        sys.exit(runner(cls))

    def collect_project_files(
        self,
        project_path: Path,
        *,
        pattern: str = c.Infra.Extensions.PYTHON_GLOB,
    ) -> list[Path]:
        """Collect project files that match current engine scope filters."""
        scan_dirs = self.rule_loader.extract_project_scan_dirs(self.config)
        candidate_roots = [project_path / rel_dir for rel_dir in scan_dirs]
        existing_roots = [root for root in candidate_roots if root.exists()]
        if not existing_roots:
            dirs = ", ".join(scan_dirs)
            output.error(f"No configured scan directories in {project_path}: {dirs}")
            return []
        ignore_items, extension_items = self.rule_loader.extract_engine_file_filters(
            self.config,
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
        project_paths = self.rule_loader.discover_workspace_projects(root)
        all_files: list[Path] = []
        for project in project_paths:
            all_files.extend(self.collect_project_files(project, pattern=pattern))
        return all_files

    def list_rules(self) -> list[dict[str, str | bool]]:
        """Return loaded rules metadata for listing."""
        return [
            {
                c.Infra.ReportKeys.ID: rule.rule_id,
                c.Infra.Toml.NAME: rule.name,
                "description": rule.description,
                c.Infra.ReportKeys.ENABLED: rule.enabled,
                "severity": rule.severity,
            }
            for rule in self.rules
        ]

    def load_config(self) -> r[t.ConfigurationMapping]:
        """Load YAML configuration for this engine instance."""
        result = self.rule_loader.load_config()
        if result.is_success:
            self.config = result.value
            output.info(f"Loaded config from {self.config_path}")
        return result

    def load_rules(self) -> r[list[FlextInfraRefactorRule]]:
        """Load and instantiate enabled rules from rules directory."""
        rules_result = self.rule_loader.load_rules(
            self.rule_filters,
            self.rule_validator,
            self._build_rule,
            self._build_file_rules,
        )
        if rules_result.is_failure:
            return r[list[FlextInfraRefactorRule]].fail(rules_result.error or "")
        loaded_rules, loaded_file_rules = rules_result.value
        self.rules = loaded_rules
        self.file_rules = loaded_file_rules
        output.info(f"Loaded {len(self.rules)} rules")
        if self.file_rules:
            output.info(f"Loaded {len(self.file_rules)} file rules")
        if self.rule_filters:
            output.info(f"Active filters: {', '.join(self.rule_filters)}")
        return r[list[FlextInfraRefactorRule]].ok(loaded_rules)

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
            modified = file_rule_modified or result_code != original_source
            if not dry_run and modified:
                u.write_file(file_path, result_code, encoding=c.Infra.Encoding.DEFAULT)
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
                    ),
                )
                continue
            result = self.refactor_file(file_path, dry_run=dry_run)
            results.append(result)
            if result.success:
                if result.modified:
                    output.info(
                        f"{('[DRY-RUN] ' if dry_run else '')}Modified: {file_path.name}",
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
        if apply_safety and (not dry_run):
            stash_result = self.safety_manager.create_pre_transformation_stash(
                project_path,
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
                    ),
                ]
            stash_ref = stash_result.value
        files = self.collect_project_files(project_path, pattern=pattern)
        output.info(f"Found {len(files)} files to process")
        results = self.refactor_files(files, dry_run=dry_run)
        if apply_safety and (not dry_run):
            checkpoint_result = self.safety_manager.save_checkpoint_state(
                project_path,
                status="transformed",
                stash_ref=stash_ref,
                processed_targets=[str(file_path) for file_path in files],
            )
            if checkpoint_result.is_failure:
                output.error(checkpoint_result.error or "checkpoint save failed")
            validation_result = self.safety_manager.run_semantic_validation(
                project_path,
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
                    ),
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
        project_paths = self.rule_loader.discover_workspace_projects(root)
        if not project_paths:
            output.error(
                f"No projects discovered under workspace root: {workspace_root}",
            )
            return []
        output.info(f"Discovered {len(project_paths)} projects in workspace")
        results: list[m.Infra.Refactor.Result] = []
        processed_targets: list[str] = []
        stash_ref = ""
        if apply_safety and (not dry_run):
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
                    ),
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
            if apply_safety and (not dry_run):
                processed_targets.append(str(project))
                checkpoint_result = self.safety_manager.save_checkpoint_state(
                    root,
                    status="running",
                    stash_ref=stash_ref,
                    processed_targets=list(processed_targets),
                )
                if checkpoint_result.is_failure:
                    output.error(checkpoint_result.error or "checkpoint save failed")
        if apply_safety and (not dry_run):
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
                    ),
                )
            else:
                clear_result = self.safety_manager.clear_checkpoint()
                if clear_result.is_failure:
                    output.error(clear_result.error or "checkpoint clear failed")
        return results

    def set_rule_filters(self, filters: list[str]) -> None:
        """Set active rule filters using normalized lowercase rule ids."""
        self.rule_filters = [item.lower() for item in filters]

    def _build_file_rules(self) -> list[ClassNestingRefactorRule]:
        return [ClassNestingRefactorRule()]

    def _build_rule(
        self,
        rule_def: Mapping[str, t.ContainerValue],
    ) -> FlextInfraRefactorRule | None:
        rule_id = str(rule_def.get(c.Infra.ReportKeys.ID, c.Infra.Defaults.UNKNOWN))
        fix_action = (
            str(
                rule_def.get(
                    c.Infra.ReportKeys.FIX_ACTION,
                    rule_def.get(c.Infra.ReportKeys.ACTION, ""),
                ),
            )
            .strip()
            .lower()
        )
        check = str(rule_def.get(c.Infra.Verbs.CHECK, "")).strip().lower()
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
            if fix_action == "migrate_to_class_mro":
                return FlextInfraRefactorMROClassMigrationRule(rule_def)
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
            if "migrate-to-class-mro" in rule_id_lower:
                return FlextInfraRefactorMROClassMigrationRule(rule_def)
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


__all__ = ["FlextInfraRefactorEngine"]
