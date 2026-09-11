"""Census fix application + post-apply regeneration — extracted concern."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, u
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.detectors.class_placement_detector import (
    FlextInfraClassPlacementDetector,
)
from flext_infra.detectors.compatibility_alias_detector import (
    FlextInfraCompatibilityAliasDetector,
)
from flext_infra.detectors.inline_import_detector import FlextInfraInlineImportDetector
from flext_infra.detectors.manual_typing_alias_detector import (
    FlextInfraManualTypingAliasDetector,
)
from flext_infra.detectors.private_import_bypass_detector import (
    FlextInfraPrivateImportBypassDetector,
)
from flext_infra.refactor._census_apply_formatting import (
    FlextInfraRefactorCensusApplyFormattingMixin,
)
from flext_infra.refactor.classvar_constant_autofix import (
    FlextInfraRefactorClassvarConstantAutofix,
)

if TYPE_CHECKING:
    from flext_infra import m, p, t

_log = u.fetch_logger(__name__)


class FlextInfraRefactorCensusApplyMixin(FlextInfraRefactorCensusApplyFormattingMixin):
    """Apply supported auto-fixes + removal candidates, then regenerate inits.

    Composed into FlextInfraRefactorCensus via inheritance; borrows the
    detector-context / fix-key / runtime-alias-rewrite helpers + root +
    dry_run_gate_names from the facade and sibling mixins via FLEXT.
    """

    if TYPE_CHECKING:

        @property
        def root(self) -> Path: ...

        @property
        def dry_run_gate_names(self) -> t.StrSequence: ...
        @staticmethod
        def _detector_context(
            rope: p.Infra.RopeWorkspaceDsl,
            file_path: Path,
            *,
            convention: m.Infra.RopeModuleConvention | None = None,
            parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation]
            | None = None,
        ) -> m.Infra.DetectorContext: ...
        @staticmethod
        def _fix_key(file_path: Path, object_name: str, action: str = "") -> str: ...
        @staticmethod
        def _rewrite_runtime_alias_source(
            source: str, *, alias: str, target_name: str
        ) -> str: ...

    def _apply_supported_fixes(
        self, rope: p.Infra.RopeWorkspaceDsl, report: m.Infra.Census.WorkspaceReport
    ) -> frozenset[str]:
        """Apply supported fixes."""
        applied: set[str] = set()
        touched_paths: set[Path] = set()
        applied_actions: set[str] = set()
        requested_fixes: dict[tuple[Path, str], set[str]] = defaultdict(set)
        for project in report.projects:
            for fix in project.fixes:
                requested_fixes[Path(fix.source_file), fix.action].add(fix.object_name)
        for (file_path, action), object_names in requested_fixes.items():
            parse_failures: list[m.Infra.ParseFailureViolation] = []
            ctx = self._detector_context(rope, file_path, parse_failures=parse_failures)
            changed = False
            if action == "rewrite_runtime_alias":
                convention = rope.convention(file_path)
                alias = (
                    convention.module_policy.expected_alias
                    or c.Infra.NAMESPACE_FILE_TO_FAMILY.get(file_path.name, "")
                )
                target_name = next(iter(sorted(object_names)), "")
                if not alias or not target_name:
                    continue
                source = rope.source(file_path)
                updated = self._rewrite_runtime_alias_source(
                    source, alias=alias, target_name=target_name
                )
                if updated == source:
                    continue
                resource = rope.resource(file_path)
                if resource is None:
                    continue
                resource.write(updated)
                changed = True
            elif action == "rewrite_manual_typing_alias":
                if ctx.project_root is None:
                    continue
                manual_typing_violations: tuple[
                    m.Infra.ManualTypingAliasViolation, ...
                ] = tuple(
                    violation
                    for violation in FlextInfraManualTypingAliasDetector.detect_file(
                        ctx
                    )
                    if violation.name in object_names
                )
                if not manual_typing_violations:
                    continue
                u.Infra.rewrite_manual_typing_alias_violations(
                    project_root=ctx.project_root,
                    violations=manual_typing_violations,
                    parse_failures=parse_failures,
                    gates=self.dry_run_gate_names,
                )
                changed = True
            elif action == "rewrite_compatibility_alias":
                compatibility_violations: tuple[
                    m.Infra.CompatibilityAliasViolation, ...
                ] = tuple(
                    violation
                    for violation in FlextInfraCompatibilityAliasDetector.detect_file(
                        ctx
                    )
                    if violation.alias_name in object_names
                )
                if not compatibility_violations:
                    continue
                u.Infra.rewrite_compatibility_alias_violations(
                    violations=compatibility_violations, parse_failures=parse_failures
                )
                changed = True
            elif action == "rewrite_private_import_bypass":
                private_import_violations: tuple[
                    m.Infra.PrivateImportBypassViolation, ...
                ] = tuple(
                    violation
                    for violation in FlextInfraPrivateImportBypassDetector.detect_file(
                        ctx
                    )
                    if violation.imported_symbol in object_names
                    and violation.symbol_exported
                )
                if not private_import_violations:
                    continue
                u.Infra.rewrite_private_import_bypass_violations(
                    rope_project=ctx.rope_project,
                    violations=private_import_violations,
                    parse_failures=parse_failures,
                )
                changed = True
            elif action in {"hoist_inline_import", "rewrite_library_abstraction"}:
                changed = self._apply_hoist_inline_imports(
                    rope=rope,
                    file_path=file_path,
                    object_names=object_names,
                    action=action,
                )
            elif action == "rewrite_foreign_canonical_alias":
                foreign_canonical_violations: tuple[
                    m.Infra.CompatibilityAliasViolation, ...
                ] = tuple(
                    violation
                    for violation in FlextInfraCompatibilityAliasDetector.detect_file(
                        ctx
                    )
                    if violation.alias_name in object_names
                    and FlextInfraCompatibilityAliasDetector.fix_action_for(
                        violation,
                        current_project=u.Infra.package_name(file_path).split(".")[0],
                    )
                    == "rewrite_foreign_canonical_alias"
                )
                if not foreign_canonical_violations:
                    continue
                u.Infra.rewrite_foreign_canonical_alias_violations(
                    rope_project=ctx.rope_project,
                    violations=foreign_canonical_violations,
                    parse_failures=parse_failures,
                )
                changed = True
            elif action == "classvar_relocation":
                changed = self._apply_classvar_relocation(
                    rope=rope, file_path=file_path, object_names=object_names
                )
            elif action == "remove_stub_file":
                if file_path.suffix == ".pyi" and file_path.exists():
                    file_path.unlink()
                    changed = True
            elif action == "one_class_per_module":
                changed = self._apply_one_class_per_module(
                    rope=rope, file_path=file_path, object_names=object_names
                )
            elif action == "fix_silent_failure_sentinels":
                resource = rope.resource(file_path)
                if resource is None:
                    continue
                _updated, changes = u.Infra.fix_silent_failure_sentinels(
                    rope.rope_project,
                    resource,
                    apply=True,
                    kinds=frozenset(object_names) if object_names else None,
                )
                changed = len(changes) > 0
            elif action in {"deep_namespace_refactor", "manual"}:
                # Manual-only actions: reported in dry-run, no-op during apply.
                pass
            if not changed:
                continue
            applied_actions.add(action)
            touched_paths.add(file_path.resolve())
            applied.update(
                self._fix_key(file_path, object_name, action)
                for object_name in object_names
            )
        for candidate in report.removal_candidates:
            apply_result = u.Infra.apply_simple_removal_candidate(
                rope, self.root, candidate, gates=self.dry_run_gate_names
            )
            if apply_result.failure:
                msg = apply_result.error or (
                    "simple removal apply failed for "
                    f"{candidate.file_path}:{candidate.line} {candidate.object_name}"
                )
                _log.warning(
                    "census_apply_candidate_rejected",
                    candidate=candidate.file_path,
                    object_name=candidate.object_name,
                    error=msg,
                )
                continue
            if apply_result.unwrap_or(False):
                applied.add(
                    self._fix_key(Path(candidate.file_path), candidate.object_name)
                )
                touched_paths.add(Path(candidate.file_path).resolve())
                touched_paths.update(
                    Path(site.file_path).resolve()
                    for site in candidate.script_reference_sites
                )
        stub_only = (
            applied_actions == {"remove_stub_file"} and not report.removal_candidates
        )
        if applied:
            self._ruff_fix_touched_files(touched_paths)
            if not stub_only:
                self._regenerate_inits_via_codegen()
            rope.reload()
        return frozenset(applied)

    def _apply_hoist_inline_imports(
        self,
        *,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        object_names: set[str],
        action: str = "hoist_inline_import",
    ) -> bool:
        """Hoist inline imports to module top using rope + AST offsets.

        Callers filter via ``FlextInfraInlineImportDetector.fix_action_for``:
        ``hoist_inline_import`` for stdlib imports (no cycle risk) and
        ``rewrite_library_abstraction`` for libraries owned by another FLEXT
        project.
        """
        resource = rope.resource(file_path)
        if resource is None:
            return False
        ctx = self._detector_context(rope, file_path)
        violations = [
            violation
            for violation in FlextInfraInlineImportDetector.detect_file(ctx)
            if violation.current_import in object_names
            and FlextInfraInlineImportDetector.fix_action_for(
                module_name=violation.module_name, is_importlib=violation.is_importlib
            )
            == action
        ]
        if not violations:
            return False
        try:
            pymodule = u.Infra.get_pymodule(rope.rope_project, resource)
            tree = pymodule.get_ast()
        except (*u.Infra.rope_runtime_errors(), TypeError):
            return False
        if not isinstance(tree, ast.Module):
            return False
        source = rope.source(file_path)
        lines = source.splitlines(keepends=True)
        line_ranges_to_remove: list[tuple[int, int]] = []
        imports_to_add: list[tuple[str, tuple[str, ...]]] = []
        for violation in violations:
            target = self._find_inline_import_node(tree, violation.line)
            if target is None:
                continue
            start_line = target.lineno
            end_line = getattr(target, "end_lineno", start_line) or start_line
            line_ranges_to_remove.append((start_line, end_line))
            if isinstance(target, ast.Import):
                imports_to_add.extend(("", (alias.name,)) for alias in target.names)
            else:
                module_name = target.module or ""
                imports_to_add.append((
                    module_name,
                    tuple(alias.name for alias in target.names),
                ))
        if not line_ranges_to_remove:
            return False
        updated_lines = self._remove_line_ranges(lines, line_ranges_to_remove)
        new_source = "".join(updated_lines)
        try:
            compile(new_source, str(file_path), "exec")
        except SyntaxError:
            return False
        resource.write(new_source)
        for module_name, names in imports_to_add:
            if module_name:
                u.Infra.add_import(
                    rope.rope_project, resource, module_name, names, apply=True
                )
            else:
                for name in names:
                    top_module = name.split(".")[0]
                    u.Infra.add_import(
                        rope.rope_project,
                        resource,
                        top_module,
                        (top_module,),
                        apply=True,
                    )
        return True

    def _apply_classvar_relocation(
        self, *, rope: p.Infra.RopeWorkspaceDsl, file_path: Path, object_names: set[str]
    ) -> bool:
        """Apply ENFORCE-079: move ClassVar constants to the _constants module."""
        ctx = self._detector_context(rope, file_path)
        violations = [
            violation
            for violation in FlextInfraClassPlacementDetector.detect_file(ctx)
            if violation.action == "classvar_relocation"
            and violation.name in object_names
        ]
        if not violations or ctx.project_root is None:
            return False
        convention = rope.convention(file_path)
        constants_module = self._derive_constants_module(convention)
        if not constants_module:
            return False
        changed = False
        for violation in violations:
            class_full_name = f"{convention.module_name}.{violation.base_class}"
            applied_one = False
            try:
                FlextInfraRefactorClassvarConstantAutofix.apply(
                    repository_root=ctx.project_root,
                    class_full_name=class_full_name,
                    constant_name=violation.name,
                    constants_module=constants_module,
                    dry_run=False,
                )
                applied_one = True
            except c.EXC_BROAD_IO_TYPE:
                applied_one = False
            changed = changed or applied_one
        return changed

    def _apply_one_class_per_module(
        self, *, rope: p.Infra.RopeWorkspaceDsl, file_path: Path, object_names: set[str]
    ) -> bool:
        """Apply ENFORCE-067: split a single misplaced class to its own module."""
        ctx = self._detector_context(rope, file_path)
        violations = [
            violation
            for violation in FlextInfraClassPlacementDetector.detect_file(ctx)
            if violation.action == "one_class_per_module"
            and violation.name in object_names
            and violation.fixable
        ]
        if not violations or ctx.project_root is None:
            return False
        convention = rope.convention(file_path)
        package_dir = (
            convention.project_layout.package_dir
            if convention.project_layout is not None
            else convention.package_dir
        )
        changed = False
        for violation in violations:
            family = violation.family
            if not family:
                continue
            family_dir = c.Infra.FAMILY_DIRECTORIES.get(family)
            if family_dir is None:
                continue
            target_file = (
                package_dir / family_dir / f"{self._to_snake_case(violation.name)}.py"
            )
            if self._move_class_to_file(
                rope=rope,
                source_file=file_path,
                target_file=target_file,
                class_name=violation.name,
            ):
                changed = True
        return changed

    @staticmethod
    def _derive_constants_module(convention: m.Infra.RopeModuleConvention) -> str:
        """Return the canonical _constants module for a source convention."""
        base_package = convention.package_name.split(".")[0]
        return f"{base_package}._constants"

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a PascalCase class name to a snake_case module name."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def _move_class_to_file(
        self,
        *,
        rope: p.Infra.RopeWorkspaceDsl,
        source_file: Path,
        target_file: Path,
        class_name: str,
    ) -> bool:
        """Move a single top-level class from ``source_file`` to ``target_file``."""
        source_resource = rope.resource(source_file)
        if source_resource is None:
            return False
        source = source_resource.read()
        prefix = f"class {class_name}"
        offset = source.find(prefix)
        if offset < 0:
            return False
        # Place the cursor on the class name, not on the ``class`` keyword.
        offset += len("class ")

        target_file.parent.mkdir(parents=True, exist_ok=True)
        if not target_file.exists():
            target_file.write_text(
                f"{c.Infra.FUTURE_ANNOTATIONS}\n", encoding=c.Cli.ENCODING_DEFAULT
            )
            rope.reload()

        target_resource = rope.resource(target_file)
        if target_resource is None:
            return False
        try:
            mover = u.Infra.create_move(rope.rope_project, source_resource, offset)
        except (*u.Infra.rope_runtime_errors(), TypeError):
            return False
        try:
            changes = mover.get_changes(target_resource)
            rope.rope_project.do(changes)
        except c.EXC_BROAD_IO_TYPE:
            return False
        return True

    @staticmethod
    def _find_inline_import_node(
        tree: ast.Module, line: int
    ) -> ast.Import | ast.ImportFrom | None:
        """Find an Import/ImportFrom node at ``line`` inside a function body."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            if node.lineno != line:
                continue
            parent = _find_parent(tree, node)
            while parent is not None:
                if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                    return node
                parent = _find_parent(tree, parent)
        return None

    @staticmethod
    def _remove_line_ranges(
        lines: list[str], ranges: list[tuple[int, int]]
    ) -> list[str]:
        """Remove 1-based inclusive line ranges, returning the updated line list."""
        drop: set[int] = set()
        for start, end in ranges:
            drop.update(range(start, end + 1))
        return [line for index, line in enumerate(lines, start=1) if index not in drop]

    def _regenerate_inits_via_codegen(self) -> None:
        """Regenerate every ``__init__.py`` via the canonical lazy-init service."""
        FlextInfraCodegenLazyInit(repository_root=self.root).generate_inits(
            check_only=False
        )


def _find_parent(tree: ast.AST, target: ast.AST) -> ast.AST | None:
    """Return the parent AST node of ``target`` within ``tree``."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is target:
                return parent
    return None


__all__: list[str] = ["FlextInfraRefactorCensusApplyMixin"]
