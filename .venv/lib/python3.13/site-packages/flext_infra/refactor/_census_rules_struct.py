"""Census structural rule scanners — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import m
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
from flext_infra.detectors.silent_failure_detector import (
    FlextInfraSilentFailureDetector,
)
from flext_infra.refactor._census_rules_shared import (
    FlextInfraRefactorCensusRulesSharedMixin,
)

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraRefactorCensusRulesStructMixin(
    FlextInfraRefactorCensusRulesSharedMixin
):
    """Compatibility and structural rule scanners for one module.

    Composed into FlextInfraRefactorCensus via inheritance; borrows the
    detector-context + violation/fix builders from sibling mixins via FLEXT.
    """

    def _rule_class_placement(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        symbol_index: dict[str, tuple[str, int]],
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Detect + plan fixes for misplaced class declarations."""
        _ = symbol_index
        ctx = self._detector_context(rope, file_path, convention=convention)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        for detector_violation in FlextInfraClassPlacementDetector.detect_file(ctx):
            matched = (
                self._named_object(objects, detector_violation.name)
                if objects is not None
                else None
            )
            object_kind = matched.kind if matched is not None else "class"
            if selected_kinds and object_kind not in selected_kinds:
                continue
            action = detector_violation.action
            violations.append(
                self._raw_violation(
                    project=project_name,
                    object_name=detector_violation.name,
                    object_kind=object_kind,
                    kind="class_placement",
                    file_path=file_path,
                    line=detector_violation.line,
                    description=detector_violation.suggestion,
                    fixable=detector_violation.fixable,
                    fix_action=action,
                )
            )
            fixes.append(
                m.Infra.Census.Fix(
                    object_name=detector_violation.name,
                    action=action,
                    source_file=str(file_path),
                    files_changed=1,
                    applied=self._fix_key(file_path, detector_violation.name, action)
                    in applied,
                )
            )
        return violations, fixes

    def _rule_private_import_bypass(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        symbol_index: dict[str, tuple[str, int]],
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Detect + plan fixes for private-import bypass violations."""
        _ = objects, symbol_index
        ctx = self._detector_context(rope, file_path, convention=convention)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        for detector_violation in FlextInfraPrivateImportBypassDetector.detect_file(
            ctx
        ):
            object_kind = "import"
            if selected_kinds and object_kind not in selected_kinds:
                continue
            fixable = detector_violation.symbol_exported
            action = "rewrite_private_import_bypass" if fixable else "manual"
            violations.append(
                self._raw_violation(
                    project=project_name,
                    object_name=detector_violation.imported_symbol,
                    object_kind=object_kind,
                    kind="private_import_bypass",
                    file_path=file_path,
                    line=detector_violation.line,
                    description=detector_violation.detail,
                    fixable=fixable,
                    fix_action=action,
                )
            )
            if fixable:
                fixes.append(
                    m.Infra.Census.Fix(
                        object_name=detector_violation.imported_symbol,
                        action=action,
                        source_file=str(file_path),
                        files_changed=1,
                        applied=self._fix_key(
                            file_path, detector_violation.imported_symbol, action
                        )
                        in applied,
                    )
                )
        return violations, fixes

    def _rule_compatibility_alias(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        symbol_index: dict[str, tuple[str, int]],
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Detect + plan fixes for compatibility-alias violations."""
        ctx = self._detector_context(rope, file_path, convention=convention)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        for detector_violation in FlextInfraCompatibilityAliasDetector.detect_file(ctx):
            matched = (
                self._named_object(objects, detector_violation.alias_name)
                if objects is not None
                else None
            )
            object_kind = matched.kind if matched is not None else "assignment"
            if matched is None:
                target_symbol = symbol_index.get(detector_violation.target_name)
                if target_symbol is not None and target_symbol[0] in {
                    "class",
                    "function",
                }:
                    object_kind = target_symbol[0]
            if selected_kinds and object_kind not in selected_kinds:
                continue
            action = FlextInfraCompatibilityAliasDetector.fix_action_for(
                detector_violation, current_project=project_name
            )
            violations.append(
                self._raw_violation(
                    project=project_name,
                    object_name=detector_violation.alias_name,
                    object_kind=object_kind,
                    kind="compatibility_alias",
                    file_path=file_path,
                    line=detector_violation.line,
                    description=(
                        "Compatibility alias "
                        f"'{detector_violation.alias_name}' should use "
                        f"'{detector_violation.target_name}' directly"
                    ),
                    fixable=True,
                    fix_action=action,
                )
            )
            fixes.append(
                m.Infra.Census.Fix(
                    object_name=detector_violation.alias_name,
                    action=action,
                    source_file=str(file_path),
                    files_changed=1,
                    applied=self._fix_key(
                        file_path, detector_violation.alias_name, action
                    )
                    in applied,
                )
            )
        return violations, fixes

    def _rule_inline_import(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        symbol_index: dict[str, tuple[str, int]],
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Detect + plan fixes for inline/lazy imports inside function bodies."""
        _ = objects, symbol_index
        ctx = self._detector_context(rope, file_path, convention=convention)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        for detector_violation in FlextInfraInlineImportDetector.detect_file(ctx):
            object_kind = "import"
            if selected_kinds and object_kind not in selected_kinds:
                continue
            action = FlextInfraInlineImportDetector.fix_action_for(
                module_name=detector_violation.module_name,
                is_importlib=detector_violation.is_importlib,
            )
            fixable = action == "hoist_inline_import"
            violations.append(
                self._raw_violation(
                    project=project_name,
                    object_name=detector_violation.current_import,
                    object_kind=object_kind,
                    kind="inline_import",
                    file_path=file_path,
                    line=detector_violation.line,
                    description=detector_violation.detail,
                    fixable=fixable,
                    fix_action=action,
                )
            )
            if fixable:
                fixes.append(
                    m.Infra.Census.Fix(
                        object_name=detector_violation.current_import,
                        action=action,
                        source_file=str(file_path),
                        files_changed=1,
                        applied=self._fix_key(
                            file_path, detector_violation.current_import, action
                        )
                        in applied,
                    )
                )
        return violations, fixes

    def _rule_silent_failure(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        symbol_index: dict[str, tuple[str, int]],
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Detect exception-silencing patterns; auto-fix deterministic sentinels."""
        _ = objects, symbol_index
        ctx = self._detector_context(rope, file_path, convention=convention)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        for detector_violation in FlextInfraSilentFailureDetector.detect_violations(
            ctx
        ):
            object_kind = "statement"
            if selected_kinds and object_kind not in selected_kinds:
                continue
            action = detector_violation.fix_action
            fixable = action == "fix_silent_failure_sentinels"
            violations.append(
                self._raw_violation(
                    project=project_name,
                    object_name=detector_violation.kind,
                    object_kind=object_kind,
                    kind="silent_failure",
                    file_path=file_path,
                    line=detector_violation.line,
                    description=detector_violation.detail,
                    fixable=fixable,
                    fix_action=action,
                )
            )
            if fixable:
                fixes.append(
                    m.Infra.Census.Fix(
                        object_name=detector_violation.kind,
                        action=action,
                        source_file=str(file_path),
                        files_changed=1,
                        applied=self._fix_key(
                            file_path, detector_violation.kind, action
                        )
                        in applied,
                    )
                )
        return violations, fixes


__all__: list[str] = ["FlextInfraRefactorCensusRulesStructMixin"]
