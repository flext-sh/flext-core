"""Auto-fix engine for namespace violations.

AST-based auto-fixer that moves standalone Final constants to constants.py
and standalone TypeVar/TypeAlias definitions to typings.py.

Uses text-based line operations for file writes (format-preserving),
following the pattern from refactor/analysis.py rewrite_source/insert_import.
AST is used only for detection and dependency analysis (business logic).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
import operator
from collections.abc import Sequence
from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import FlextInfraUtilitiesDiscovery, c, m, u
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.codegen.transforms import FlextInfraCodegenTransforms
from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator
from flext_infra.refactor.engine import FlextInfraRefactorEngine
from flext_infra.refactor.migrate_to_class_mro import (
    FlextInfraRefactorMigrateToClassMRO,
)
from flext_infra.refactor.namespace_rewriter import NamespaceEnforcementRewriter


class FlextInfraCodegenFixer(s[list[m.Infra.Codegen.AutoFixResult]]):
    """AST-based auto-fixer for namespace violations (Rules 1-2)."""

    _workspace_root: Path

    def __init__(self, workspace_root: Path) -> None:
        """Initialize codegen fixer with workspace root."""
        super().__init__()
        self._workspace_root = workspace_root

    # ------------------------------------------------------------------
    # AST analysis helpers (no file I/O, tree mutation for analysis only)
    # ------------------------------------------------------------------

    @classmethod
    def _all_deps_resolvable(cls, node: ast.stmt, target_tree: ast.Module) -> bool:
        """Check if all names used in node are available in the target module.

        Called AFTER _copy_required_imports to verify the copy succeeded.
        A name is available if it's imported or defined in the target module.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
        if not names_used:
            return True
        available: set[str] = set(dir(_builtins_module))
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    available.add(imported_name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        available.add(imported_name)
            else:
                name = FlextInfraCodegenTransforms.get_node_name(stmt)
                if name:
                    available.add(name)
        return all(n in available for n in names_used)

    @classmethod
    def _copy_required_imports(
        cls,
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> None:
        """Copy imports needed by node from source_tree to target_tree.

        Mutates target_tree for analysis accumulation only — the tree is
        never written to disk via ast.unparse.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
        if not names_used:
            return
        source_imports: dict[str, ast.stmt] = {}
        for stmt in source_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    top_name = imported_name.split(".")[0]
                    source_imports[top_name] = stmt
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        source_imports[imported_name] = stmt
        target_available: set[str] = set()
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    target_available.add(imported_name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported_name = alias.asname or alias.name
                    if imported_name != "*":
                        target_available.add(imported_name)
        seen_modules: set[str] = set()
        imports_to_add: list[ast.stmt] = []
        for name in sorted(names_used):
            if name in target_available:
                continue
            if name not in source_imports:
                continue
            import_stmt = source_imports[name]
            import_key = ast.unparse(import_stmt)
            if import_key in seen_modules:
                continue
            seen_modules.add(import_key)
            imports_to_add.append(import_stmt)
        if not imports_to_add:
            return
        last_import_idx = 0
        for i, stmt in enumerate(target_tree.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            ):
                last_import_idx = i + 1
        for i, imp in enumerate(imports_to_add):
            target_tree.body.insert(last_import_idx + i, imp)

    @classmethod
    def _needs_first_party_import(
        cls,
        node: ast.stmt,
        source_tree: ast.Module,
        target_tree: ast.Module,
    ) -> bool:
        """Check if moving node to target would require a first-party import.

        First-party imports (from flext_*) into typings.py create circular
        import chains because typings.py is imported by result.py, runtime.py,
        and other foundational modules. If the node depends on names that come
        from first-party imports and those names are NOT already available in
        the target module, moving the node is unsafe.
        """
        names_used = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
        node_name = FlextInfraCodegenTransforms.get_node_name(node)
        type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
        names_used = frozenset(
            n for n in names_used if n != node_name and n not in type_params
        )
        if not names_used:
            return False
        target_available: set[str] = set(dir(_builtins_module))
        for stmt in target_tree.body:
            if isinstance(stmt, ast.Import):
                target_available.update(
                    (alias.asname or alias.name).split(".")[0] for alias in stmt.names
                )
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        target_available.add(imported)
            else:
                found = FlextInfraCodegenTransforms.get_node_name(stmt)
                if found:
                    target_available.add(found)
        missing = names_used - target_available
        if not missing:
            return False
        for stmt in source_tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module:
                if stmt.module.startswith("flext"):
                    for alias in stmt.names:
                        imported = alias.asname or alias.name
                        if imported in missing:
                            return True
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    top = (alias.asname or alias.name).split(".")[0]
                    if top.startswith("flext") and top in missing:
                        return True
        return False

    # ------------------------------------------------------------------
    # Text-based write helpers (format-preserving, no ast.unparse)
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_import_text(source: str, import_stmt: str) -> str:
        """Insert an import statement after module docstring/imports."""
        return u.Infra.insert_import_statement(source, import_stmt)

    @classmethod
    def _collect_import_texts_for_nodes(
        cls,
        nodes: Sequence[ast.stmt],
        source_lines: list[str],
        source_tree: ast.Module,
        target_text: str,
    ) -> list[str]:
        """Collect import text lines from source needed by moved nodes.

        Returns import statement strings that should be added to the target
        file. Skips imports already present in the target text.
        """
        all_names: set[str] = set()
        for node in nodes:
            names = FlextInfraCodegenTransforms.get_top_level_names_in_node(node)
            node_name = FlextInfraCodegenTransforms.get_node_name(node)
            type_params = FlextInfraCodegenTransforms.get_type_param_names(node)
            all_names.update(
                n for n in names if n != node_name and n not in type_params
            )
        if not all_names:
            return []
        import_texts: list[str] = []
        seen: set[str] = set()
        for stmt in source_tree.body:
            if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            provided: set[str] = set()
            if isinstance(stmt, ast.Import):
                provided.update(
                    (alias.asname or alias.name).split(".")[0] for alias in stmt.names
                )
            else:
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        provided.add(imported)
            if not (provided & all_names):
                continue
            start = stmt.lineno
            end = stmt.end_lineno or start
            text = "\n".join(source_lines[start - 1 : end]).strip()
            if text not in seen and text not in target_text:
                seen.add(text)
                import_texts.append(text)
        return import_texts

    @classmethod
    def _write_changes(
        cls,
        *,
        source_path: Path,
        target_path: Path,
        nodes_moved: Sequence[ast.stmt],
        moved_names: list[str],
        source_tree: ast.Module,
        pkg_name: str,
        target_module: str,
    ) -> None:
        """Apply moves via text operations (format-preserving, no ast.unparse).

        Follows the pattern from refactor/analysis.py rewrite_source().
        Steps: extract definitions by line range, remove from source,
        add re-export import to source, add imports+definitions to target,
        normalize both files with ruff.
        """
        encoding = c.Infra.Encoding.DEFAULT
        source_text = source_path.read_text(encoding=encoding)
        source_lines = source_text.splitlines()
        target_text = target_path.read_text(encoding=encoding)

        # 1. Extract node definitions by line range
        extracted: list[str] = []
        ranges: list[tuple[int, int]] = []
        for node in nodes_moved:
            start = node.lineno
            end = node.end_lineno or node.lineno
            block = "\n".join(source_lines[start - 1 : end])
            extracted.append(block)
            ranges.append((start, end))

        # 2. Collect required imports for target
        import_texts = cls._collect_import_texts_for_nodes(
            nodes_moved,
            source_lines,
            source_tree,
            target_text,
        )

        # 3. Remove nodes from source (reverse order preserves line numbers)
        for start, end in sorted(ranges, key=operator.itemgetter(0), reverse=True):
            del source_lines[start - 1 : end]

        # 4. Add re-export import to source
        source_result = "\n".join(source_lines)
        re_export = f"from {pkg_name}.{target_module} import " + ", ".join(
            sorted(moved_names)
        )
        source_result = cls._insert_import_text(source_result, re_export)
        if source_text.endswith("\n") and not source_result.endswith("\n"):
            source_result += "\n"

        # 5. Add imports + definitions to target
        target_result = target_text
        for imp in import_texts:
            target_result = cls._insert_import_text(target_result, imp)
        for block in extracted:
            target_result = target_result.rstrip() + "\n\n\n" + block + "\n"

        # 6. Write files
        source_path.write_text(source_result, encoding=encoding)
        target_path.write_text(target_result, encoding=encoding)

        # 7. Normalize with ruff
        u.Infra.run_ruff_fix(source_path)
        u.Infra.run_ruff_fix(target_path)

    # ------------------------------------------------------------------
    # File system helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_package_dir(project_root: Path) -> Path | None:
        """Find the first Python package under src/."""
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return None
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / c.Infra.Files.INIT_PY).exists():
                return child
        return None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @override
    def execute(self) -> r[list[m.Infra.Codegen.AutoFixResult]]:
        """Execute auto-fix across all workspace projects."""
        return r[list[m.Infra.Codegen.AutoFixResult]].ok(self.run())

    def fix_project(self, project_path: Path) -> m.Infra.Codegen.AutoFixResult:
        """Auto-fix namespace violations in a single project.

        Each rule parses the source file fresh so that line numbers stay
        accurate after a preceding rule modifies the file on disk.
        """
        prefix = FlextInfraNamespaceValidator.derive_prefix(project_path)
        if not prefix:
            return m.Infra.Codegen.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )
        pkg_dir = self._find_package_dir(project_path)
        if pkg_dir is None:
            return m.Infra.Codegen.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )
        violations_fixed: list[m.Infra.Codegen.CensusViolation] = []
        violations_skipped: list[m.Infra.Codegen.CensusViolation] = []
        files_modified: set[str] = set()
        src_dir = project_path / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return m.Infra.Codegen.AutoFixResult(
                project=project_path.name,
                violations_fixed=[],
                violations_skipped=[],
                files_modified=[],
            )
        checkpoint_result = u.Infra.create_checkpoint(
            self._workspace_root,
            label=f"codegen-fix:{project_path.name}",
        )
         stash_ref = checkpoint_result.value_or("")
        report = self._apply_project_mro_migrations(
            project_path=project_path,
            files_modified=files_modified,
        )
        self._record_mro_migration_result(
            report=report,
            violations_fixed=violations_fixed,
            violations_skipped=violations_skipped,
        )
        self._apply_refactor_engine_pass(
            project_path=project_path,
            files_modified=files_modified,
            violations_skipped=violations_skipped,
        )
        self._apply_namespace_enforcement_pass(
            project_path=project_path,
            files_modified=files_modified,
        )
        self._run_lazy_propagation(
            project_path=project_path,
            files_modified=files_modified,
        )
        try:
            self._cleanup_stale_all_entries(files_modified=files_modified)
            self._normalize_rewritten_python_files(files_modified=files_modified)
        except (OSError, UnicodeDecodeError):
            _ = u.Infra.rollback_to_checkpoint(self._workspace_root, stash_ref)
            raise
        return m.Infra.Codegen.AutoFixResult(
            project=project_path.name,
            violations_fixed=violations_fixed,
            violations_skipped=violations_skipped,
            files_modified=sorted(files_modified),
        )

    def _apply_project_mro_migrations(
        self,
        *,
        project_path: Path,
        files_modified: set[str],
    ) -> m.Infra.Refactor.MROMigrationReport:
        service = FlextInfraRefactorMigrateToClassMRO(workspace_root=project_path)
        report = service.run(target="all", apply_changes=True)
        files_modified.update(migration.file for migration in report.migrations)
        files_modified.update(rewrite.file for rewrite in report.rewrites)
        return report

    def _apply_refactor_engine_pass(
        self,
        *,
        project_path: Path,
        files_modified: set[str],
        violations_skipped: list[m.Infra.Codegen.CensusViolation],
    ) -> None:
        engine = FlextInfraRefactorEngine()
        config_result = engine.load_config()
        if config_result.is_failure:
            message = config_result.error or "Failed to load refactor engine config"
            violations_skipped.append(
                m.Infra.Codegen.CensusViolation(
                    module=str(project_path),
                    rule="NS-ENGINE",
                    line=1,
                    message=message,
                    fixable=False,
                ),
            )
            return
        engine.set_rule_filters(
            [
                "modernize-constants-import",
                "modernize-models-import",
                "modernize-result-import",
                "ban-lazy-imports",
                "ensure-future-annotations",
                "remove-compatibility-aliases",
                "remove-wrapper-functions",
                "remove-deprecated-classes",
                "remove-import-bypasses",
                "fix-container-invariance-annotations",
                "remove-validated-redundant-casts",
            ],
        )
        rules_result = engine.load_rules()
        if rules_result.is_failure:
            message = rules_result.error or "Failed to load filtered refactor rules"
            violations_skipped.append(
                m.Infra.Codegen.CensusViolation(
                    module=str(project_path),
                    rule="NS-ENGINE",
                    line=1,
                    message=message,
                    fixable=False,
                ),
            )
            return
        results = engine.refactor_project(
            project_path,
            dry_run=False,
            apply_safety=False,
        )
        files_modified.update(
            str(result.file_path)
            for result in results
            if result.success and result.modified
        )
        violations_skipped.extend(
            m.Infra.Codegen.CensusViolation(
                module=str(result.file_path),
                rule="NS-ENGINE",
                line=1,
                message=result.error or "Refactor engine pass failed",
                fixable=False,
            )
            for result in results
            if not result.success
        )

    def _apply_namespace_enforcement_pass(
        self,
        *,
        project_path: Path,
        files_modified: set[str],
    ) -> None:
        py_files_result = u.Infra.iter_python_files(
            workspace_root=project_path,
            project_roots=[project_path],
            src_dirs=frozenset(c.Infra.Refactor.MRO_SCAN_DIRECTORIES),
        )
        if py_files_result.is_failure:
            return
        py_files = py_files_result.value
        src_files = [
            file_path
            for file_path in py_files
            if c.Infra.Paths.DEFAULT_SRC_DIR in file_path.parts
        ]
        before_snapshot = self._snapshot_files(file_paths=src_files)
        NamespaceEnforcementRewriter.rewrite_import_alias_violations(py_files=src_files)
        NamespaceEnforcementRewriter.rewrite_runtime_alias_violations(
            py_files=src_files
        )
        NamespaceEnforcementRewriter.rewrite_missing_future_annotations(
            py_files=src_files
        )
        changed_paths = self._detect_changed_files(
            before_snapshot=before_snapshot,
            file_paths=src_files,
        )
        files_modified.update(changed_paths)

    @staticmethod
    def _record_mro_migration_result(
        *,
        report: m.Infra.Refactor.MROMigrationReport,
        violations_fixed: list[m.Infra.Codegen.CensusViolation],
        violations_skipped: list[m.Infra.Codegen.CensusViolation],
    ) -> None:
        for migration in report.migrations:
            violations_fixed.extend(
                m.Infra.Codegen.CensusViolation(
                    module=migration.file,
                    rule="NS-MRO",
                    line=1,
                    message=(
                        "Moved symbol "
                        f"'{moved_symbol}' into namespace class via MRO migration"
                    ),
                    fixable=True,
                )
                for moved_symbol in migration.moved_symbols
            )
        if report.remaining_violations > 0:
            violations_skipped.append(
                m.Infra.Codegen.CensusViolation(
                    module=report.workspace,
                    rule="NS-MRO",
                    line=1,
                    message=(
                        "MRO migration finished with "
                        f"{report.remaining_violations} remaining violations"
                    ),
                    fixable=False,
                ),
            )
        if report.mro_failures > 0:
            violations_skipped.append(
                m.Infra.Codegen.CensusViolation(
                    module=report.workspace,
                    rule="NS-MRO",
                    line=1,
                    message=f"MRO validation reported {report.mro_failures} failures",
                    fixable=False,
                ),
            )
        violations_skipped.extend(
            m.Infra.Codegen.CensusViolation(
                module=report.workspace,
                rule="NS-MRO",
                line=1,
                message=warning,
                fixable=False,
            )
            for warning in report.warnings
        )
        violations_skipped.extend(
            m.Infra.Codegen.CensusViolation(
                module=report.workspace,
                rule="NS-MRO",
                line=1,
                message=error,
                fixable=False,
            )
            for error in report.errors
        )

    def _cleanup_stale_all_entries(self, *, files_modified: set[str]) -> None:
        for file_path in sorted(files_modified):
            path = Path(file_path)
            if not path.exists() or path.suffix != c.Infra.Extensions.PYTHON:
                continue
            if path.name == c.Infra.Files.INIT_PY:
                continue
            if self._prune_stale_all_assignment(path=path):
                files_modified.add(str(path))

    @staticmethod
    def _prune_stale_all_assignment(*, path: Path) -> bool:
        try:
            source = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except (OSError, UnicodeDecodeError):
            return False
        # NOTE: source text needed below - cannot delegate to u.Infra.parse_module_ast
        tree = u.Infra.parse_ast_from_source(source)
        if tree is None:
            return False
        assignment: ast.Assign | None = None
        exports: list[str] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or target.id != "__all__":
                continue
            if not isinstance(stmt.value, (ast.List, ast.Tuple)):
                continue
            names: list[str] = []
            is_literal_list = True
            for element in stmt.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    names.append(element.value)
                    continue
                is_literal_list = False
                break
            if not is_literal_list:
                continue
            assignment = stmt
            exports = names
            break
        if assignment is None or len(exports) == 0:
            return False
        available: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.FunctionDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.AsyncFunctionDef):
                available.add(stmt.name)
                continue
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    available.add(imported.split(".")[0])
                continue
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    imported = alias.asname or alias.name
                    if imported != "*":
                        available.add(imported)
                continue
            found_name = FlextInfraCodegenTransforms.get_node_name(stmt)
            if found_name:
                available.add(found_name)
        filtered = [
            name for name in exports if name in available or name.startswith("__")
        ]
        if filtered == exports:
            return False
        block = "__all__ = [\n" + "\n".join(f'    "{name}",' for name in filtered)
        if len(filtered) == 0:
            block = "__all__ = []"
        else:
            block += "\n]"
        lines = source.splitlines()
        if assignment.lineno <= 0 or assignment.end_lineno is None:
            return False
        start = assignment.lineno - 1
        end = assignment.end_lineno
        updated_lines = [*lines[:start], block, *lines[end:]]
        updated = "\n".join(updated_lines)
        if source.endswith("\n"):
            updated += "\n"
        if updated == source:
            return False
        path.write_text(updated, encoding=c.Infra.Encoding.DEFAULT)
        return True

    def _normalize_rewritten_python_files(self, *, files_modified: set[str]) -> None:
        for file_path in sorted(files_modified):
            path = Path(file_path)
            if not path.exists() or path.suffix != c.Infra.Extensions.PYTHON:
                continue
            u.Infra.run_ruff_fix(path)

    def _run_lazy_propagation(
        self,
        *,
        project_path: Path,
        files_modified: set[str],
    ) -> None:
        before_snapshot = self._snapshot_init_files(project_path=project_path)
        _ = FlextInfraCodegenLazyInit(workspace_root=project_path).run(check_only=False)
        after_snapshot = self._snapshot_init_files(project_path=project_path)
        for path_str, updated in after_snapshot.items():
            previous = before_snapshot.get(path_str)
            if previous == updated:
                continue
            files_modified.add(path_str)

    @staticmethod
    def _snapshot_init_files(*, project_path: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for root_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES:
            root = project_path / root_name
            if not root.is_dir():
                continue
            for init_file in u.Infra.iter_directory_python_files(
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

    def run(self) -> list[m.Infra.Codegen.AutoFixResult]:
        """Run auto-fix on all projects in workspace.

        Returns:
            List of AutoFixResult models, one per project.

        """
        discovery = FlextInfraUtilitiesDiscovery()
        projects_result = discovery.discover_projects(self._workspace_root)
        if not projects_result.is_success:
            return []
        results: list[m.Infra.Codegen.AutoFixResult] = []
        discovered: list[m.Infra.Workspace.ProjectInfo] = projects_result.unwrap()
        for project in discovered:
            if project.name in c.Infra.Codegen.EXCLUDED_PROJECTS:
                continue
            if project.stack.startswith(c.Infra.Gates.GO):
                continue
            result = self.fix_project(project.path)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Rule implementations (parse fresh, text-based writes)
    # ------------------------------------------------------------------

    def _fix_rule1(
        self,
        *,
        source_file: Path,
        pkg_dir: Path,
        violations_fixed: list[m.Infra.Codegen.CensusViolation],
        violations_skipped: list[m.Infra.Codegen.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 1 — move loose Final constants to constants.py."""
        tree = u.Infra.parse_module_ast(source_file)
        if tree is None:
            return
        finals = FlextInfraCodegenTransforms.find_standalone_finals(tree)
        if not finals:
            return
        target_path = pkg_dir / "constants.py"
        if not target_path.exists():
            return
        target_tree = u.Infra.parse_module_ast(target_path)
        if target_tree is None:
            return
        nodes_to_move: list[ast.AnnAssign] = []
        for node in finals:
            target_name = ""
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
            if target_name.startswith("_"):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-001",
                        line=node.lineno,
                        message=f"Final constant '{target_name}' is private — skipped",
                        fixable=False,
                    ),
                )
                continue
            nodes_to_move.append(node)
        if not nodes_to_move:
            return
        pkg_name = pkg_dir.name
        actually_moved: list[ast.AnnAssign] = []
        moved_names: list[str] = []
        for node in nodes_to_move:
            target_name = ""
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
            if FlextInfraCodegenTransforms.name_exists_in_module(
                target_name, target_tree
            ):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-001",
                        line=node.lineno,
                        message=f"Final constant '{target_name}' already in constants.py — skipped",
                        fixable=False,
                    ),
                )
                continue
            self._copy_required_imports(node, tree, target_tree)
            if not self._all_deps_resolvable(node, target_tree):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-001",
                        line=node.lineno,
                        message=f"Final constant '{target_name}' has unresolvable deps — skipped",
                        fixable=False,
                    ),
                )
                continue
            # Insert into target_tree for analysis accumulation
            insert_idx = FlextInfraCodegenTransforms.find_insert_position(target_tree)
            target_tree.body.insert(insert_idx, node)
            violations_fixed.append(
                m.Infra.Codegen.CensusViolation(
                    module=str(source_file),
                    rule="NS-001",
                    line=node.lineno,
                    message=f"Loose Final constant '{target_name}' moved to constants.py",
                    fixable=True,
                ),
            )
            actually_moved.append(node)
            moved_names.append(target_name)
        if actually_moved:
            self._write_changes(
                source_path=source_file,
                target_path=target_path,
                nodes_moved=actually_moved,
                moved_names=moved_names,
                source_tree=tree,
                pkg_name=pkg_name,
                target_module="constants",
            )
            files_modified.add(str(source_file))
            files_modified.add(str(target_path))

    def _fix_rule2(
        self,
        *,
        source_file: Path,
        pkg_dir: Path,
        violations_fixed: list[m.Infra.Codegen.CensusViolation],
        violations_skipped: list[m.Infra.Codegen.CensusViolation],
        files_modified: set[str],
    ) -> None:
        """Fix Rule 2 — move loose TypeVars/TypeAliases to typings.py."""
        tree = u.Infra.parse_module_ast(source_file)
        if tree is None:
            return
        typevars = FlextInfraCodegenTransforms.find_standalone_typevars(tree)
        typealiases = FlextInfraCodegenTransforms.find_standalone_typealiases(tree)
        if not typevars and not typealiases:
            return
        target_path = pkg_dir / "typings.py"
        if not target_path.exists():
            return
        target_tree = u.Infra.parse_module_ast(target_path)
        if target_tree is None:
            return
        nodes_to_move: list[ast.stmt] = []
        for tv_node in typevars:
            target_name = ""
            if tv_node.targets:
                target = tv_node.targets[0]
                if isinstance(target, ast.Name):
                    target_name = target.id
            if target_name.startswith("_"):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-002",
                        line=tv_node.lineno,
                        message=f"TypeVar '{target_name}' is private — skipped",
                        fixable=False,
                    ),
                )
                continue
            nodes_to_move.append(tv_node)
        for alias_node in typealiases:
            target_name = FlextInfraCodegenTransforms.get_node_name(alias_node)
            if target_name.startswith("_"):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-002",
                        line=alias_node.lineno,
                        message=f"TypeAlias '{target_name}' is private — skipped",
                        fixable=False,
                    ),
                )
                continue
            nodes_to_move.append(alias_node)
        if not nodes_to_move:
            return
        pkg_name = pkg_dir.name
        actually_moved: list[ast.stmt] = []
        moved_names: list[str] = []
        for move_node in nodes_to_move:
            target_name = FlextInfraCodegenTransforms.get_node_name(move_node)
            if not target_name:
                continue
            if FlextInfraCodegenTransforms.name_exists_in_module(
                target_name, target_tree
            ):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-002",
                        line=move_node.lineno,
                        message=f"'{target_name}' already in typings.py — skipped",
                        fixable=False,
                    ),
                )
                continue
            if self._needs_first_party_import(move_node, tree, target_tree):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-002",
                        line=move_node.lineno,
                        message=f"'{target_name}' needs first-party import — circular risk, skipped",
                        fixable=False,
                    ),
                )
                continue
            self._copy_required_imports(move_node, tree, target_tree)
            if not self._all_deps_resolvable(move_node, target_tree):
                violations_skipped.append(
                    m.Infra.Codegen.CensusViolation(
                        module=str(source_file),
                        rule="NS-002",
                        line=move_node.lineno,
                        message=f"'{target_name}' has unresolvable deps — skipped",
                        fixable=False,
                    ),
                )
                continue
            # Insert into target_tree for analysis accumulation
            insert_idx = FlextInfraCodegenTransforms.find_insert_position(target_tree)
            target_tree.body.insert(insert_idx, move_node)
            kind = "TypeVar" if isinstance(move_node, ast.Assign) else "TypeAlias"
            violations_fixed.append(
                m.Infra.Codegen.CensusViolation(
                    module=str(source_file),
                    rule="NS-002",
                    line=move_node.lineno,
                    message=f"{kind} '{target_name}' moved to typings.py",
                    fixable=True,
                ),
            )
            actually_moved.append(move_node)
            moved_names.append(target_name)
        if actually_moved:
            self._write_changes(
                source_path=source_file,
                target_path=target_path,
                nodes_moved=actually_moved,
                moved_names=moved_names,
                source_tree=tree,
                pkg_name=pkg_name,
                target_module=c.Infra.Directories.TYPINGS,
            )
            files_modified.add(str(source_file))
            files_modified.add(str(target_path))


__all__ = ["FlextInfraCodegenFixer"]
