"""Census collection-gate + module-selection helpers — extracted concern."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, config, m
from flext_infra._enforcement.engine import FlextInfraEnforcementEngine

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t


class FlextInfraRefactorCensusCollectHelpersMixin:
    """Decide what to collect + select modules for the requested rule set.

    Composed into FlextInfraRefactorCensus via inheritance; owns the
    lightweight-rule set, the collection-gate helpers, and the top-level
    ``_collect_report`` orchestrator (scan selected modules → assemble).
    """

    _LIGHTWEIGHT_MODULE_RULES: ClassVar[frozenset[str]] = frozenset({
        "runtime_alias",
        "manual_typing_alias",
        "compatibility_alias",
    })
    _PYI_GLOB: ClassVar[str] = "*.pyi"
    _PYI_SUFFIX: ClassVar[str] = ".pyi"

    if TYPE_CHECKING:

        @staticmethod
        def _selected_families(family_names: t.StrSequence | None) -> frozenset[str]:
            """Resolve selected families through the composed object mixin."""
            ...

        def _scan_module(
            self,
            rope: p.Infra.RopeWorkspaceDsl,
            module: m.Infra.RopeModuleIndexEntry,
            config: m.Infra.Census.ScanConfig,
            *,
            project_objects: dict[str, list[m.Infra.Census.Object]],
            project_violations: dict[str, list[m.Infra.Census.Violation]],
            project_fixes: dict[str, list[m.Infra.Census.Fix]],
            report_projects: set[str],
        ) -> None:
            """Scan through the composed census collection mixin."""
            ...

        def _assemble_report(
            self,
            rope: p.Infra.RopeWorkspaceDsl,
            *,
            project_objects: dict[str, list[m.Infra.Census.Object]],
            project_violations: dict[str, list[m.Infra.Census.Violation]],
            project_fixes: dict[str, list[m.Infra.Census.Fix]],
            report_projects: set[str],
            rule_names: t.StrSequence | None,
            selected_rules: frozenset[str] | None,
        ) -> m.Infra.Census.WorkspaceReport:
            """Assemble through the composed census collection mixin."""
            ...

    @staticmethod
    def _should_collect_object_references(rule_names: t.StrSequence | None) -> bool:
        """Decide whether to collect object references."""
        if rule_names is None:
            return True
        return "unused" in rule_names

    @classmethod
    def _should_collect_object_inventory(
        cls,
        rule_names: t.StrSequence | None,
        *,
        selected_rules: frozenset[str] | None = None,
    ) -> bool:
        """Decide whether to collect the full object inventory."""
        if selected_rules is None:
            selected_rules = frozenset(rule_names) if rule_names else None
        if not selected_rules:
            return True
        declarative_rules = cls._declarative_rules_for_selection(rule_names)
        declarative_rule_ids = frozenset(rule.id for rule in declarative_rules)
        if declarative_rule_ids and selected_rules <= declarative_rule_ids:
            return False
        return not selected_rules <= cls._LIGHTWEIGHT_MODULE_RULES

    @staticmethod
    def _declarative_rules_for_selection(
        rule_names: t.StrSequence | None,
    ) -> tuple[me.EnforcementRuleSpec, ...]:
        """Return catalog declarative rules selected by the census request."""
        return FlextInfraEnforcementEngine.declarative_rules(rule_names)

    @staticmethod
    def _rule_requires_stub_file(rule: me.EnforcementRuleSpec) -> bool:
        """Return whether ``rule`` must scan ``.pyi`` files outside Rope modules."""
        return FlextInfraEnforcementEngine.rule_requires_stub_file(rule)

    @staticmethod
    def _project_name_for_module(
        module: m.Infra.RopeModuleIndexEntry, convention: m.Infra.RopeModuleConvention
    ) -> str:
        """Project name for a module entry."""
        layout = convention.project_layout
        if layout is not None:
            return layout.project_name
        if module.project_root is not None:
            return module.project_root.name
        return ""

    @staticmethod
    def _is_production_module(module: m.Infra.RopeModuleIndexEntry) -> bool:
        """Return whether a module belongs to one configured production root."""
        project_root = module.project_root
        if project_root is None:
            return False
        try:
            relative_path = module.file_path.resolve().relative_to(
                project_root.resolve()
            )
        except ValueError:
            return False
        return bool(relative_path.parts) and (
            relative_path.parts[0] in config.Infra.source_scan.roots
        )

    def _modules_for_rules(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        *,
        project_names: t.StrSequence | None,
        rule_names: t.StrSequence | None,
    ) -> tuple[m.Infra.RopeModuleIndexEntry, ...]:
        """Modules for rules."""
        modules = tuple(
            module
            for module in rope.modules(project_names=project_names)
            if self._is_production_module(module)
        )
        declarative_rules = self._declarative_rules_for_selection(rule_names)
        if any(self._rule_requires_stub_file(rule) for rule in declarative_rules):
            modules = (*modules, *self._stub_modules(rope, modules, project_names))
        return modules

    @classmethod
    def _stub_modules(
        cls,
        rope: p.Infra.RopeWorkspaceDsl,
        modules: t.SequenceOf[m.Infra.RopeModuleIndexEntry],
        project_names: t.StrSequence | None,
    ) -> tuple[m.Infra.RopeModuleIndexEntry, ...]:
        """Return synthetic module entries for selected workspace ``.pyi`` files."""
        known_paths = frozenset(module.file_path.resolve() for module in modules)
        roots = tuple(
            sorted({
                module.project_root.resolve()
                for module in modules
                if module.project_root is not None
            })
        )
        project_filter = frozenset(project_names or ())
        entries: list[m.Infra.RopeModuleIndexEntry] = []
        for root in roots:
            if project_filter and root.name not in project_filter:
                continue
            src_root = root / c.Infra.DEFAULT_SRC_DIR
            if not src_root.is_dir():
                continue
            for stub_path in sorted(src_root.rglob(cls._PYI_GLOB)):
                resolved = stub_path.resolve()
                if resolved in known_paths:
                    continue
                entries.append(cls._stub_module_entry(rope, root, src_root, resolved))
        return tuple(entries)

    @classmethod
    def _stub_module_entry(
        cls,
        rope: p.Infra.RopeWorkspaceDsl,
        project_root: Path,
        src_root: Path,
        stub_path: Path,
    ) -> m.Infra.RopeModuleIndexEntry:
        """Build a RopeModuleIndexEntry for a ``.pyi`` file."""
        relative = stub_path.relative_to(src_root)
        module_parts = relative.with_suffix("").parts
        if relative.name == c.Infra.INIT_PYI:
            module_parts = module_parts[:-1]
        module_name = ".".join(module_parts)
        package_name = module_parts[0] if module_parts else project_root.name
        resource_path = str(stub_path.relative_to(rope.rope_repository_root))
        return m.Infra.RopeModuleIndexEntry(
            file_path=stub_path,
            resource_path=resource_path,
            module_name=module_name,
            package_name=package_name,
            package_dir=stub_path.parent,
            project_root=project_root,
            is_package_init=stub_path.name == c.Infra.INIT_PYI,
        )

    def _collect_report(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        *,
        project_names: t.StrSequence | None,
        kind_names: t.StrSequence | None,
        family_names: t.StrSequence | None,
        rule_names: t.StrSequence | None,
        include_local_scopes: bool,
        applied: frozenset[str],
    ) -> m.Infra.Census.WorkspaceReport:
        """Scan selected modules then assemble the workspace census report."""
        selected_families = self._selected_families(family_names)
        selected_rules: frozenset[str] | None = (
            frozenset(rule_names) if rule_names else None
        )
        config = m.Infra.Census.ScanConfig(
            kind_names=kind_names,
            rule_names=rule_names,
            selected_families=selected_families,
            selected_kinds=frozenset(kind_names) if kind_names else None,
            selected_rules=selected_rules,
            collect_object_inventory=self._should_collect_object_inventory(
                rule_names, selected_rules=selected_rules
            ),
            include_object_references=self._should_collect_object_references(
                rule_names
            ),
            include_local_scopes=include_local_scopes,
            applied=applied,
        )
        project_objects: dict[str, list[m.Infra.Census.Object]] = defaultdict(list)
        project_violations: dict[str, list[m.Infra.Census.Violation]] = defaultdict(
            list
        )
        project_fixes: dict[str, list[m.Infra.Census.Fix]] = defaultdict(list)
        report_projects: set[str] = set()
        for module in self._modules_for_rules(
            rope, project_names=project_names, rule_names=rule_names
        ):
            self._scan_module(
                rope,
                module,
                config,
                project_objects=project_objects,
                project_violations=project_violations,
                project_fixes=project_fixes,
                report_projects=report_projects,
            )
        return self._assemble_report(
            rope,
            project_objects=project_objects,
            project_violations=project_violations,
            project_fixes=project_fixes,
            report_projects=report_projects,
            rule_names=rule_names,
            selected_rules=selected_rules,
        )


__all__: list[str] = ["FlextInfraRefactorCensusCollectHelpersMixin"]
