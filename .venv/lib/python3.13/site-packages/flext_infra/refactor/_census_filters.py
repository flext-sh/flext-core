"""Census duplicate-grouping, object/rule filters, and runtime-alias helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from flext_infra import m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorCensusFiltersMixin:
    """Duplicate detection, inclusion filters, and runtime-alias rewriting.

    Composed into FlextInfraRefactorCensus via inheritance; self-contained
    static helpers over report Object/convention models (no census state).
    """

    _MIN_DUPLICATE_DEFINITIONS: ClassVar[int] = 2

    @staticmethod
    def _duplicate_groups(
        project_objects: tuple[list[m.Infra.Census.Object], ...],
    ) -> tuple[m.Infra.Census.DuplicateGroup, ...]:
        """Duplicate groups."""

        def object_location(item: m.Infra.Census.Object) -> tuple[str, str, int]:
            return item.project, item.file_path, item.line

        groups: dict[tuple[str, str, str], list[m.Infra.Census.Object]] = defaultdict(
            list
        )
        for item in (obj for objects in project_objects for obj in objects):
            owner = item.scope_path.rpartition(".")[0]
            groups[item.kind, item.name, owner].append(item)
        duplicates: list[m.Infra.Census.DuplicateGroup] = []
        for key in sorted(groups):
            definitions = groups[key]
            if (
                len(definitions)
                < FlextInfraRefactorCensusFiltersMixin._MIN_DUPLICATE_DEFINITIONS
            ):
                continue
            canonical = min(definitions, key=object_location)
            duplicates.append(
                m.Infra.Census.DuplicateGroup(
                    name=definitions[0].name,
                    kind=definitions[0].kind,
                    definitions=tuple(definitions),
                    canonical=canonical.project,
                    value_identical=len({item.fingerprint for item in definitions})
                    == 1,
                )
            )
        return tuple(duplicates)

    @staticmethod
    def _include_object(
        item: m.Infra.Census.Object,
        *,
        kind_names: t.StrSequence | None,
        selected_families: frozenset[str],
        selected_kinds: frozenset[str] | None = None,
    ) -> bool:
        """Include object.

        ``selected_kinds`` is a precomputed frozenset of ``kind_names``; when
        omitted it is rebuilt from ``kind_names`` (kept for back-compat). Hot
        callers must pass the precomputed set to avoid per-object frozenset
        construction.
        """
        kinds = (
            selected_kinds
            if selected_kinds is not None
            else (frozenset(kind_names) if kind_names else None)
        )
        if kinds and item.kind not in kinds:
            return False
        if not selected_families:
            return True
        return (
            item.actual_tier.lower() in selected_families
            or item.expected_tier.lower() in selected_families
        )

    @staticmethod
    def _include_rule(
        rule: str,
        *,
        rule_names: t.StrSequence | None,
        selected_rules: frozenset[str] | None = None,
    ) -> bool:
        """Include rule.

        ``selected_rules`` is a precomputed frozenset of ``rule_names``;
        callers in hot loops MUST pass it to avoid per-call set construction.
        """
        if selected_rules is None:
            return rule_names is None or rule in frozenset(rule_names)
        return rule in selected_rules

    @staticmethod
    def _named_object(
        objects: tuple[m.Infra.Census.Object, ...], name: str
    ) -> m.Infra.Census.Object | None:
        """Named object."""
        return next(
            (item for item in objects if name in {item.scope_path, item.name}), None
        )

    @staticmethod
    def _runtime_alias_target(
        convention: m.Infra.RopeModuleConvention,
        objects: tuple[m.Infra.Census.Object, ...] | None,
    ) -> m.Infra.Census.Object | None:
        """Runtime alias target."""
        if objects is None:
            return None
        target_name = FlextInfraRefactorCensusFiltersMixin._runtime_alias_target_name(
            convention
        )
        if not target_name:
            return None
        return FlextInfraRefactorCensusFiltersMixin._named_object(objects, target_name)

    @staticmethod
    def _runtime_alias_target_name(convention: m.Infra.RopeModuleConvention) -> str:
        """Return the expected runtime alias target name."""
        layout = convention.project_layout
        family = convention.module_policy.expected_family or ""
        if layout is None or not family:
            return ""
        return (
            family
            if family.startswith(layout.class_stem)
            else f"{layout.class_stem}{family}"
        )

    @staticmethod
    def _rewrite_runtime_alias_source(
        source: str, *, alias: str, target_name: str
    ) -> str:
        """Rewrite runtime alias source."""
        filtered_lines = [
            line
            for line in source.splitlines()
            if not line.strip().startswith(f"{alias} =")
        ]
        cleaned_source = "\n".join(filtered_lines).rstrip()
        if cleaned_source:
            cleaned_source = f"{cleaned_source}\n"
        updated_source: str = u.Infra.ensure_runtime_alias(
            cleaned_source, alias=alias, target_name=target_name
        )
        return updated_source


__all__: list[str] = ["FlextInfraRefactorCensusFiltersMixin"]
