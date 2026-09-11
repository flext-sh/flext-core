"""Census per-module rule dispatch — extracted concern."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import m
from flext_infra._enforcement.engine import FlextInfraEnforcementEngine
from flext_infra.refactor._census_rules_shared import (
    FlextInfraRefactorCensusRulesSharedMixin,
)

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t


class FlextInfraRefactorCensusRulesDispatchMixin(
    FlextInfraRefactorCensusRulesSharedMixin
):
    """Run every selected structural rule for one module and collect outcomes.

    Parent of FlextInfraRefactorCensusCollectMixin (its ``_scan_module`` calls
    ``_module_rules``); borrows the symbol-index, rule-selection filter, and
    the four per-rule detectors from sibling mixins via FLEXT.
    """

    if TYPE_CHECKING:

        @classmethod
        def _lightweight_symbol_index(
            cls, rope: p.Infra.RopeWorkspaceDsl, file_path: Path
        ) -> dict[str, tuple[str, int]]: ...
        @staticmethod
        def _include_rule(
            rule: str,
            *,
            rule_names: t.StrSequence | None,
            selected_rules: frozenset[str] | None = None,
        ) -> bool: ...
        def _rule_runtime_alias(
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
        def _rule_manual_typing_alias(
            self,
            rope: p.Infra.RopeWorkspaceDsl,
            file_path: Path,
            *,
            project_name: str,
            objects: tuple[m.Infra.Census.Object, ...] | None,
            applied: frozenset[str],
            selected_kinds: frozenset[str],
            convention: m.Infra.RopeModuleConvention,
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...
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
        ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]: ...

    def _module_rules(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        objects: tuple[m.Infra.Census.Object, ...] | None,
        project_name: str,
        applied: frozenset[str],
        kind_names: t.StrSequence | None,
        rule_names: t.StrSequence | None,
        selected_kinds: frozenset[str] | None = None,
        selected_rules: frozenset[str] | None = None,
        convention: m.Infra.RopeModuleConvention | None = None,
    ) -> tuple[tuple[m.Infra.Census.Violation, ...], tuple[m.Infra.Census.Fix, ...]]:
        """Run every selected structural rule for one module and collect outcomes."""
        resolved_convention = convention or rope.convention(file_path)
        resolved_kinds = (
            selected_kinds
            if selected_kinds is not None
            else (frozenset(kind_names) if kind_names else frozenset())
        )
        symbol_index = self._lightweight_symbol_index(rope, file_path)
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []

        def selected(rule_name: str) -> bool:
            return self._include_rule(
                rule_name, rule_names=rule_names, selected_rules=selected_rules
            )

        if selected("runtime_alias"):
            v, f = self._rule_runtime_alias(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("manual_typing_alias"):
            v, f = self._rule_manual_typing_alias(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("class_placement"):
            v, f = self._rule_class_placement(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("private_import_bypass"):
            v, f = self._rule_private_import_bypass(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("compatibility_alias"):
            v, f = self._rule_compatibility_alias(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("inline_import"):
            v, f = self._rule_inline_import(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        if selected("silent_failure"):
            v, f = self._rule_silent_failure(
                rope,
                file_path,
                project_name=project_name,
                objects=objects,
                applied=applied,
                selected_kinds=resolved_kinds,
                symbol_index=symbol_index,
                convention=resolved_convention,
            )
            violations.extend(v)
            fixes.extend(f)
        v, f = self._rule_declarative(
            rope,
            file_path,
            project_name=project_name,
            applied=applied,
            selected_kinds=resolved_kinds,
            selected_rules=selected_rules,
            rule_names=rule_names,
            convention=resolved_convention,
        )
        violations.extend(v)
        fixes.extend(f)
        return (tuple(violations), tuple(fixes))

    @staticmethod
    def _declarative_catalog_rules() -> tuple[me.EnforcementRuleSpec, ...]:
        """Return enabled catalog rules handled by the declarative engine."""
        return FlextInfraEnforcementEngine.declarative_rules()

    def _rule_declarative(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        file_path: Path,
        *,
        project_name: str,
        applied: frozenset[str],
        selected_kinds: frozenset[str],
        selected_rules: frozenset[str] | None,
        rule_names: t.StrSequence | None,
        convention: m.Infra.RopeModuleConvention,
    ) -> tuple[list[m.Infra.Census.Violation], list[m.Infra.Census.Fix]]:
        """Run catalog-driven declarative rules for one module."""
        violations: list[m.Infra.Census.Violation] = []
        fixes: list[m.Infra.Census.Fix] = []
        rules = self._declarative_catalog_rules()
        if not rules:
            return violations, fixes
        ctx = self._detector_context(rope, file_path, convention=convention)
        for rule in rules:
            if not self._include_rule(
                rule.id, rule_names=rule_names, selected_rules=selected_rules
            ):
                continue
            kind = FlextInfraEnforcementEngine.violation_kind(rule)
            object_kind = FlextInfraEnforcementEngine.object_kind(kind)
            if selected_kinds and kind not in selected_kinds:
                continue
            probes = FlextInfraEnforcementEngine.detect_declarative(rule, ctx)
            fix_action = rule.fix_action
            fixable = fix_action is not None and fix_action.safe
            action = fix_action.target if fix_action is not None else ""
            for probe in probes:
                line = getattr(probe, "line", 0)
                if not isinstance(line, int) or line < 0:
                    line = 0
                object_name = FlextInfraEnforcementEngine.object_name(probe, kind)
                description = FlextInfraEnforcementEngine.description(
                    rule, probe, object_name
                )
                violations.append(
                    self._raw_violation(
                        project=project_name,
                        object_name=object_name,
                        object_kind=object_kind,
                        kind=kind,
                        file_path=file_path,
                        line=line,
                        description=description,
                        fixable=fixable,
                        fix_action=action,
                    )
                )
                if action:
                    fixes.append(
                        m.Infra.Census.Fix(
                            object_name=object_name,
                            action=action,
                            source_file=str(file_path),
                            files_changed=1,
                            applied=self._fix_key(file_path, object_name, action)
                            in applied,
                        )
                    )
        return violations, fixes


__all__: list[str] = ["FlextInfraRefactorCensusRulesDispatchMixin"]
