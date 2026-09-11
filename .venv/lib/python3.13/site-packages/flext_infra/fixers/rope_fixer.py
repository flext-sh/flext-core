"""Rope-backed fix adapter for enforcement rules with semantic refactoring.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra.detectors.class_placement_detector import (
    FlextInfraClassPlacementDetector,
)
from flext_infra.detectors.compatibility_alias_detector import (
    FlextInfraCompatibilityAliasDetector,
)
from flext_infra.detectors.inline_import_detector import FlextInfraInlineImportDetector
from flext_infra.detectors.private_import_bypass_detector import (
    FlextInfraPrivateImportBypassDetector,
)
from flext_infra.fixers.base import FlextInfraFixerAdapter
from flext_infra.refactor.classvar_constant_autofix import (
    FlextInfraRefactorClassvarConstantAutofix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t


class FlextInfraRopeFixerAdapter(FlextInfraFixerAdapter):
    """Apply fixes by running rope-backed refactor operations per violation.

    Targets are canonical rope refactor short-names declared in the enforcement
    catalog.  Each target uses rope (and the existing detector infrastructure)
    to locate violations and rewrite source safely.
    """

    kind: ClassVar[str] = "rope"
    _PRIVATE_NAMESPACE_MIN_PARTS: ClassVar[int] = 2

    def __init__(self, repository_root: Path) -> None:
        """Bind the repository root used to open rope projects."""
        super().__init__(repository_root)

    @staticmethod
    def _build_project_fix_result(
        project_dir: Path,
        fixed: list[m.Infra.FixedViolation],
        previewed: list[m.Infra.PreviewedViolation],
        skipped: list[m.Infra.SkippedViolation],
        failed: list[m.Infra.FailedFix],
        files_modified: set[str],
    ) -> m.Infra.ProjectFixResult:
        """Build the immutable ``ProjectFixResult`` from accumulated lists."""
        return m.Infra.ProjectFixResult(
            project=project_dir.name,
            fixed=tuple(fixed),
            previewed=tuple(previewed),
            skipped=tuple(skipped),
            failed=tuple(failed),
            files_modified=tuple(files_modified),
        )

    def _record_file_fix_outcome(
        self,
        ctx: m.Infra.FixEnforcementCommand,
        rule_id: str,
        file_path: Path,
        message: str,
        fixed: list[m.Infra.FixedViolation],
        previewed: list[m.Infra.PreviewedViolation],
    ) -> None:
        """Record a per-file fix outcome as fixed or previewed based on ``ctx.apply``."""
        file_path_str = str(file_path)
        if ctx.apply:
            fixed.append(
                m.Infra.FixedViolation(
                    rule_id=rule_id, file_path=file_path_str, message=message
                )
            )
        else:
            previewed.append(
                m.Infra.PreviewedViolation(
                    rule_id=rule_id, file_path=file_path_str, message=message
                )
            )

    @override
    def can_fix(self, fix_action: me.EnforcementFixAction) -> bool:
        """Return whether this adapter handles ``fix_action``."""
        return (
            fix_action.kind == self.kind
            and fix_action.target in self._target_dispatch()
        )

    @override
    def fix_project(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Apply rope fixes grouped by target."""
        if not violations:
            return m.Infra.ProjectFixResult(project=project_dir.name)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        for target, target_violations in self._group_by_target(violations).items():
            handler = self._target_dispatch().get(target)
            if handler is None:
                rule_id = target_violations[0][0].id
                failed.append(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        error=f"rope target {target} not registered",
                    )
                )
                continue
            result = handler(project_dir, target_violations, ctx)
            fixed.extend(result.fixed)
            previewed.extend(result.previewed)
            skipped.extend(result.skipped)
            failed.extend(result.failed)
            files_modified.update(result.files_modified)
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _target_dispatch(
        self,
    ) -> dict[
        str,
        Callable[
            [
                Path,
                t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
                m.Infra.FixEnforcementCommand,
            ],
            m.Infra.ProjectFixResult,
        ],
    ]:
        """Return bound rope target handlers for this adapter instance."""
        return {
            "classvar_relocation": self._fix_classvar_relocation,
            "rewrite_compatibility_alias": self._fix_compatibility_alias,
            "fix_silent_failure_sentinels": self._fix_silent_failure_sentinels,
            "hoist_inline_import": self._fix_hoist_inline_import,
            "rewrite_private_import_bypass": self._fix_private_import_bypass,
            "rewrite_library_abstraction": self._fix_library_abstraction,
            "one_class_per_module": self._fix_one_class_per_module,
            "remove_stub_file": self._fix_remove_stub_file,
        }

    @staticmethod
    def _package_name_for_dir(package_dir: Path, *, project_root: Path) -> str:
        """Return the import package for a directory inside a project."""
        try:
            relative_parts = package_dir.relative_to(project_root).parts
        except ValueError:
            return ""
        if not relative_parts:
            return ""
        root_name = relative_parts[0]
        if root_name == c.Infra.DEFAULT_SRC_DIR:
            package_parts = relative_parts[1:]
        elif root_name in c.Infra.ROOT_WRAPPER_SEGMENTS:
            package_parts = relative_parts
        else:
            package_parts = ()
        return ".".join(package_parts)

    @classmethod
    def _module_name_for_file(cls, file_path: Path, *, project_root: Path) -> str:
        """Return the import module for a Python file inside a project."""
        package_name = cls._package_name_for_dir(
            file_path.parent, project_root=project_root
        )
        if file_path.name == c.Infra.INIT_PY:
            return package_name
        return f"{package_name}.{file_path.stem}" if package_name else ""

    @staticmethod
    def _module_file_for_name(module_name: str, *, project_root: Path) -> Path:
        """Return the source file path for an importable module name."""
        src_dir: str = c.Infra.DEFAULT_SRC_DIR
        module_path = Path(*module_name.split(".")).with_suffix(".py")
        return project_root / src_dir / module_path

    @staticmethod
    def _constants_module_for_file(
        file_path: Path, *, module_name: str, project_root: Path
    ) -> str:
        """Return the canonical project constants module for a source file."""
        module_parts = tuple(module_name.split("."))
        package_name = module_parts[0] if module_parts else ""
        if not package_name:
            return ""
        if package_name in c.Infra.ROOT_WRAPPER_SEGMENTS:
            if "_constants" in module_parts:
                return ""
            if file_path.stem in {"", "__init__", "__main__", "__version__"}:
                return ""
            return FlextInfraRopeFixerAdapter._wrapper_constants_module_for_file(
                module_parts=module_parts, project_root=project_root
            )
        package_root = project_root / c.Infra.DEFAULT_SRC_DIR / package_name
        try:
            relative_parts = file_path.relative_to(package_root).parts
        except ValueError:
            return ""
        if not relative_parts:
            return ""
        first_part = relative_parts[0]
        if first_part == "_constants":
            return ""
        if first_part.startswith("_"):
            if (
                len(relative_parts)
                < FlextInfraRopeFixerAdapter._PRIVATE_NAMESPACE_MIN_PARTS
            ):
                return ""
            domain = Path(relative_parts[1]).stem
        elif len(relative_parts) == 1:
            domain = file_path.stem
        else:
            domain = first_part
        normalized_domain = domain.removeprefix("_")
        if normalized_domain in {"", "__init__", "__main__", "__version__"}:
            return ""
        return f"{package_name}._constants.{normalized_domain}"

    @staticmethod
    def _wrapper_constants_module_for_file(
        *, module_parts: t.StrSequence, project_root: Path
    ) -> str:
        """Return nearest existing wrapper ``_constants`` module."""
        candidate_parts = tuple(module_parts[:-1])
        while candidate_parts:
            candidate_module = ".".join((*candidate_parts, "_constants"))
            if FlextInfraRopeFixerAdapter._constants_module_exists(
                candidate_module, project_root=project_root
            ):
                return candidate_module
            candidate_parts = candidate_parts[:-1]
        return ".".join((*module_parts[:-1], "_constants"))

    @staticmethod
    def _constants_module_exists(module_name: str, *, project_root: Path) -> bool:
        """Return whether ``module_name`` resolves to an on-disk constants module."""
        relative = Path(*module_name.split("."))
        module_file = project_root / relative.with_suffix(".py")
        package_init = project_root / relative / c.Infra.INIT_PY
        return module_file.is_file() or package_init.is_file()

    def _detect_and_rewrite_files[V](
        self,
        *,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
        detector: Callable[[m.Infra.DetectorContext], t.SequenceOf[V]],
        filter_violations: Callable[[t.SequenceOf[V]], t.SequenceOf[V]],
        rewrite: Callable[[t.SequenceOf[V]], None],
        empty_reason: str,
        change_message: Callable[[int, bool], str],
        detector_error_detail: str,
        rewrite_error_detail: str,
    ) -> m.Infra.ProjectFixResult:
        """Detect violations per file and apply a deterministic rewrite."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        file_paths = self._collect_file_paths(project_dir, violations)
        if not file_paths:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        reason="no files in violation batch",
                    ),
                ),
            )
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in file_paths:
                detect_ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                try:
                    file_violations = detector(detect_ctx)
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"{detector_error_detail}: {exc}",
                        )
                    )
                    continue
                selected = filter_violations(file_violations)
                if not selected:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason=empty_reason,
                        )
                    )
                    continue
                try:
                    rewrite(selected)
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"{rewrite_error_detail}: {exc}",
                        )
                    )
                    continue
                if ctx.apply:
                    files_modified.add(str(file_path))
                message = change_message(len(selected), ctx.apply)
                self._record_file_fix_outcome(
                    ctx, rule_id, file_path, message, fixed, previewed
                )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _fix_silent_failure_sentinels(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Rewrite deterministic silent-failure sentinels to failed Results."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in self._collect_file_paths(project_dir, violations):
                resource = u.Infra.get_resource_from_path(rope_project, file_path)
                if resource is None:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="rope resource not found",
                        )
                    )
                    continue
                try:
                    _updated, changes = u.Infra.fix_silent_failure_sentinels(
                        rope_project, resource, apply=ctx.apply
                    )
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"silent failure sentinel fix failed: {exc}",
                        )
                    )
                    continue
                if not changes:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="no changes produced",
                        )
                    )
                    continue
                if ctx.apply:
                    files_modified.add(str(file_path))
                message = (
                    f"{'rewrote' if ctx.apply else 'would rewrite'} "
                    f"{len(changes)} silent sentinel fix(es)"
                )
                self._record_file_fix_outcome(
                    ctx, rule_id, file_path, message, fixed, previewed
                )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _fix_compatibility_alias(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Rewrite compatibility aliases using the canonical detector + rewriter."""

        def _rewrite(
            file_violations: t.SequenceOf[m.Infra.CompatibilityAliasViolation],
        ) -> None:
            if ctx.apply:
                u.Infra.rewrite_compatibility_alias_violations(
                    violations=file_violations, parse_failures=[]
                )

        return self._detect_and_rewrite_files(
            project_dir=project_dir,
            violations=violations,
            ctx=ctx,
            detector=FlextInfraCompatibilityAliasDetector.detect_file,
            filter_violations=lambda v: v,
            rewrite=_rewrite,
            empty_reason="no compatibility alias violations",
            change_message=lambda count, applying: (
                f"{'rewrote' if applying else 'would rewrite'} "
                f"{count} compatibility alias violation(s)"
            ),
            detector_error_detail="compatibility alias detector failed",
            rewrite_error_detail="compatibility alias rewrite failed",
        )

    def _fix_remove_stub_file(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Remove source ``.pyi`` stubs when apply mode is enabled."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        file_paths = self._collect_file_paths(project_dir, violations)
        stub_paths = tuple(path for path in file_paths if path.suffix == ".pyi")
        if not stub_paths:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        reason="no .pyi stubs in violation batch",
                    ),
                ),
            )
        for file_path in stub_paths:
            if not file_path.is_file():
                skipped.append(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        reason="stub file not found",
                    )
                )
                continue
            message = (
                f"{'would remove' if not ctx.apply else 'removed'} "
                f"source stub {file_path.relative_to(project_dir)}"
            )
            if not ctx.apply:
                previewed.append(
                    m.Infra.PreviewedViolation(
                        rule_id=rule_id, file_path=str(file_path), message=message
                    )
                )
                continue
            try:
                file_path.unlink()
            except OSError as exc:
                failed.append(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(file_path),
                        error=f"stub removal failed: {exc}",
                    )
                )
                continue
            files_modified.add(str(file_path))
            fixed.append(
                m.Infra.FixedViolation(
                    rule_id=rule_id, file_path=str(file_path), message=message
                )
            )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _fix_private_import_bypass(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Rewrite private-module imports to their canonical facade equivalents."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        file_paths = self._collect_file_paths(project_dir, violations)
        if not file_paths:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        reason="no files in violation batch",
                    ),
                ),
            )
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in file_paths:
                detect_ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                try:
                    file_violations = FlextInfraPrivateImportBypassDetector.detect_file(
                        detect_ctx
                    )
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"private import bypass detector failed: {exc}",
                        )
                    )
                    continue
                auto_fixable = tuple(v for v in file_violations if v.symbol_exported)
                if not auto_fixable:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="no auto-fixable private import bypass violations",
                        )
                    )
                    continue
                try:
                    u.Infra.rewrite_private_import_bypass_violations(
                        rope_project=rope_project,
                        violations=auto_fixable,
                        parse_failures=[],
                        apply=ctx.apply,
                    )
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"private import bypass rewrite failed: {exc}",
                        )
                    )
                    continue
                if ctx.apply:
                    files_modified.add(str(file_path))
                message = (
                    f"{'rewrote' if ctx.apply else 'would rewrite'} "
                    f"{len(auto_fixable)} private import bypass violation(s)"
                )
                self._record_file_fix_outcome(
                    ctx, rule_id, file_path, message, fixed, previewed
                )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _fix_library_abstraction(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Hoist detector-approved FLEXT library imports to module scope."""
        return self._fix_inline_import_action(
            project_dir,
            violations,
            ctx,
            target_action="rewrite_library_abstraction",
            empty_reason="no library abstraction inline imports",
        )

    def _fix_hoist_inline_import(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Hoist detector-approved inline stdlib imports to module scope."""
        return self._fix_inline_import_action(
            project_dir,
            violations,
            ctx,
            target_action="hoist_inline_import",
            empty_reason="no hoistable inline imports",
        )

    def _fix_inline_import_action(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
        *,
        target_action: str,
        empty_reason: str,
    ) -> m.Infra.ProjectFixResult:
        """Apply one detector-approved inline-import fix action."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in self._collect_file_paths(project_dir, violations):
                detect_ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                try:
                    detected = FlextInfraInlineImportDetector.detect_file(detect_ctx)
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"inline import detector failed: {exc}",
                        )
                    )
                    continue
                hoistable = tuple(
                    violation
                    for violation in detected
                    if FlextInfraInlineImportDetector.fix_action_for(
                        module_name=violation.module_name,
                        is_importlib=violation.is_importlib,
                    )
                    == target_action
                )
                if not hoistable:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason=empty_reason,
                        )
                    )
                    continue
                resource = u.Infra.get_resource_from_path(rope_project, file_path)
                if resource is None:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="rope resource not found",
                        )
                    )
                    continue
                try:
                    updated, changes = self._hoist_inline_import_source(
                        resource.read(), hoistable, file_path=file_path
                    )
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"inline import hoist failed: {exc}",
                        )
                    )
                    continue
                if not changes:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="no changes produced",
                        )
                    )
                    continue
                if ctx.apply:
                    resource.write(updated)
                    files_modified.add(str(file_path))
                message = (
                    f"{'hoisted' if ctx.apply else 'would hoist'} "
                    f"{len(changes)} inline import(s) for {target_action}"
                )
                self._record_file_fix_outcome(
                    ctx, rule_id, file_path, message, fixed, previewed
                )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    @classmethod
    def _hoist_inline_import_source(
        cls,
        source: str,
        violations: t.SequenceOf[m.Infra.InlineImportViolation],
        *,
        file_path: Path,
    ) -> t.Infra.TransformResult:
        """Return source with hoistable inline import statements at module scope."""
        tree = ast.parse(source, filename=str(file_path))
        lines = source.splitlines(keepends=True)
        line_ranges: list[tuple[int, int]] = []
        import_lines: list[str] = []
        for violation in violations:
            node = cls._find_inline_import_node(tree, violation.line)
            if node is None:
                continue
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line) or start_line
            line_ranges.append((start_line, end_line))
            import_lines.append(cls._import_line_for_node(node))
        if not line_ranges:
            return source, []
        updated_lines = list(lines)
        for start_line, end_line in sorted(line_ranges, reverse=True):
            del updated_lines[start_line - 1 : end_line]
        insert_at = u.Infra.find_import_insert_position(updated_lines)
        unique_imports = cls._unique_new_imports(updated_lines, import_lines)
        for index, import_line in enumerate(unique_imports):
            updated_lines.insert(insert_at + index, import_line)
        updated = "".join(updated_lines)
        compile(updated, str(file_path), "exec")
        return updated, unique_imports

    @staticmethod
    def _find_inline_import_node(
        tree: ast.Module, line: int
    ) -> ast.Import | ast.ImportFrom | None:
        """Find the import statement at a detector-reported line."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno == line:
                return node
        return None

    @staticmethod
    def _import_line_for_node(node: ast.Import | ast.ImportFrom) -> str:
        """Render a top-level import line from an AST import node."""
        if isinstance(node, ast.Import):
            names = ", ".join(
                f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                for alias in node.names
            )
            return f"import {names}\n"
        module_prefix = "." * node.level
        module_name = f"{module_prefix}{node.module or ''}"
        names = ", ".join(
            f"{alias.name} as {alias.asname}" if alias.asname else alias.name
            for alias in node.names
        )
        return f"from {module_name} import {names}\n"

    @staticmethod
    def _unique_new_imports(
        lines: t.StrSequence, import_lines: t.StrSequence
    ) -> list[str]:
        """Return import lines that are not already present."""
        existing = {line.strip() for line in lines if line.strip()}
        unique: list[str] = []
        for import_line in import_lines:
            stripped = import_line.strip()
            if stripped in existing or stripped in {line.strip() for line in unique}:
                continue
            unique.append(import_line)
        return unique

    def _fix_classvar_relocation(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Move class-level constants to canonical _constants modules."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()

        with u.Infra.open_project(self._repository_root) as rope_project:
            for _rule, probe in violations:
                file_path_str = getattr(probe, "file_path", "")
                file_path = Path(file_path_str)
                if not file_path.is_file():
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=file_path_str,
                            reason="file not found",
                        )
                    )
                    continue
                detect_ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                try:
                    all_violations = FlextInfraClassPlacementDetector.detect_file(
                        detect_ctx
                    )
                except c.EXC_BROAD_RUNTIME:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=file_path_str,
                            error="detector raised runtime error",
                        )
                    )
                    continue
                classvar_violations = [
                    v for v in all_violations if v.action == "classvar_relocation"
                ]
                if not classvar_violations:
                    continue
                module_name = self._module_name_for_file(
                    file_path, project_root=project_dir
                )
                if not module_name:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=file_path_str,
                            reason="could not resolve module name",
                        )
                    )
                    continue
                constants_module = self._constants_module_for_file(
                    file_path, module_name=module_name, project_root=project_dir
                )
                if not constants_module:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=file_path_str,
                            error=(
                                "could not resolve canonical constants module "
                                f"for {module_name}"
                            ),
                        )
                    )
                    continue
                for violation in classvar_violations:
                    class_full_name = f"{module_name}.{violation.base_class}"
                    try:
                        result = FlextInfraRefactorClassvarConstantAutofix.apply(
                            project_dir,
                            class_full_name,
                            violation.name,
                            constants_module,
                            dry_run=not ctx.apply,
                        )
                    except c.EXC_BROAD_RUNTIME as exc:
                        failed.append(
                            m.Infra.FailedFix(
                                rule_id=rule_id,
                                file_path=file_path_str,
                                error=f"autofix failed: {exc}",
                            )
                        )
                        continue
                    touched = tuple(result.touched_files)
                    if ctx.apply:
                        files_modified.update(str(project_dir / p) for p in touched)
                    message = (
                        f"{'would move' if not ctx.apply else 'moved'} "
                        f"{violation.base_class}.{violation.name} -> "
                        f"{constants_module}"
                    )
                    if ctx.apply:
                        fixed.append(
                            m.Infra.FixedViolation(
                                rule_id=rule_id,
                                file_path=file_path_str,
                                message=message,
                            )
                        )
                    else:
                        previewed.append(
                            m.Infra.PreviewedViolation(
                                rule_id=rule_id,
                                file_path=file_path_str,
                                message=message,
                            )
                        )

        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _fix_one_class_per_module(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Move extra governed classes to their own canonical modules."""
        rule_id = self._rule_id(violations)
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        files_modified: set[str] = set()
        file_paths = self._collect_file_paths(project_dir, violations)
        if not file_paths:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        reason="no files in violation batch",
                    ),
                ),
            )
        layout = u.Infra.layout(project_dir)
        if layout is None:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                failed=(
                    m.Infra.FailedFix(
                        rule_id=rule_id,
                        file_path=str(project_dir),
                        error="could not resolve project layout",
                    ),
                ),
            )
        package_dir = layout.package_dir
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in file_paths:
                resource = u.Infra.fetch_python_resource(
                    rope_project, file_path, skip_protected=True, skip_settings=True
                )
                if resource is None:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="rope resource not found",
                        )
                    )
                    continue
                detect_ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                try:
                    all_violations = FlextInfraClassPlacementDetector.detect_file(
                        detect_ctx
                    )
                except c.EXC_BROAD_RUNTIME as exc:
                    failed.append(
                        m.Infra.FailedFix(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            error=f"class placement detector failed: {exc}",
                        )
                    )
                    continue
                ocpm_violations = [
                    v for v in all_violations if v.action == "one_class_per_module"
                ]
                if not ocpm_violations:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="no one_class_per_module violations",
                        )
                    )
                    continue
                governed_classes = [
                    ci
                    for ci in u.Infra.get_class_info(rope_project, resource)
                    if not ci.name.startswith("_")
                    and any(v.name == ci.name for v in ocpm_violations)
                ]
                if len(governed_classes) <= 1:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="only one governed class; no extras to move",
                        )
                    )
                    continue
                governed_classes.sort(key=operator.attrgetter("line"))
                extras = governed_classes[1:]
                moved_any = False
                for extra_ci in extras:
                    extra_violation = next(
                        (v for v in ocpm_violations if v.name == extra_ci.name), None
                    )
                    family = extra_violation.family if extra_violation else ""
                    target_file = self._target_file_for_extra_class(
                        package_dir=package_dir,
                        file_path=file_path,
                        class_name=extra_ci.name,
                        family=family,
                    )
                    move_result = self._move_class_to_module(
                        rope_project=rope_project,
                        source_file=file_path,
                        target_file=target_file,
                        class_name=extra_ci.name,
                        apply=ctx.apply,
                    )
                    if move_result.failure:
                        failed.append(
                            m.Infra.FailedFix(
                                rule_id=rule_id,
                                file_path=str(file_path),
                                error=move_result.error or "failed to move class",
                            )
                        )
                        continue
                    moved_any = True
                    if ctx.apply:
                        files_modified.add(str(file_path))
                        files_modified.add(str(target_file))
                    message = (
                        f"{'would move' if not ctx.apply else 'moved'} "
                        f"{extra_ci.name} -> "
                        f"{target_file.relative_to(project_dir)}"
                    )
                    if ctx.apply:
                        fixed.append(
                            m.Infra.FixedViolation(
                                rule_id=rule_id,
                                file_path=str(file_path),
                                message=message,
                            )
                        )
                    else:
                        previewed.append(
                            m.Infra.PreviewedViolation(
                                rule_id=rule_id,
                                file_path=str(file_path),
                                message=message,
                            )
                        )
                if not moved_any:
                    skipped.append(
                        m.Infra.SkippedViolation(
                            rule_id=rule_id,
                            file_path=str(file_path),
                            reason="no classes could be moved",
                        )
                    )
        return self._build_project_fix_result(
            project_dir, fixed, previewed, skipped, failed, files_modified
        )

    def _target_file_for_extra_class(
        self, *, package_dir: Path, file_path: Path, class_name: str, family: str
    ) -> Path:
        """Return the target module path for an extra governed class."""
        snake_name = self._to_snake_case(class_name)
        if family:
            family_dir = c.Infra.FAMILY_DIRECTORIES.get(family)
            if isinstance(family_dir, str) and family_dir:
                return package_dir / family_dir / f"{snake_name}.py"
        return file_path.parent / f"_{file_path.stem}_{snake_name}.py"

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a PascalCase class name to a snake_case module name."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def _move_class_to_module(
        self,
        *,
        rope_project: t.Infra.RopeProject,
        source_file: Path,
        target_file: Path,
        class_name: str,
        apply: bool,
    ) -> p.Result[str]:
        """Move a single top-level class from ``source_file`` to ``target_file``."""
        source_resource = u.Infra.get_resource_from_path(rope_project, source_file)
        if source_resource is None:
            return r[str].fail("source rope resource not found")
        source = source_resource.read()
        prefix = f"class {class_name}"
        offset = source.find(prefix)
        if offset < 0:
            return r[str].fail(f"class {class_name} not found in source")
        offset += len("class ")

        if not apply:
            return r[str].ok(str(target_file))

        target_file.parent.mkdir(parents=True, exist_ok=True)
        if not target_file.exists():
            target_file.write_text(
                f"{c.Infra.FUTURE_ANNOTATIONS}\n", encoding=c.Cli.ENCODING_DEFAULT
            )
            rope_project.validate()

        target_resource = u.Infra.get_resource_from_path(rope_project, target_file)
        if target_resource is None:
            return r[str].fail("target rope resource not found")
        try:
            mover = u.Infra.create_move(rope_project, source_resource, offset)
        except c.EXC_BROAD_RUNTIME as exc:
            return r[str].fail(f"create_move failed: {exc}")
        try:
            changes = mover.get_changes(target_resource)
            rope_project.do(changes)
        except c.EXC_BROAD_RUNTIME as exc:
            return r[str].fail(f"rope move failed: {exc}")
        return r[str].ok(str(target_file))
