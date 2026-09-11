"""Shared enforcement collection primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from flext_infra import c, m

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p


@dataclass(frozen=True, slots=True)
class FlextInfraEnforcementEvaluation:
    """Collected rule probes and collection failures for one project."""

    violations: list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]]
    failures: list[m.Infra.FailedFix]


class FlextInfraEnforcementCollectionBase:
    """Reusable probe, path, and failure helpers for enforcement collectors."""

    @staticmethod
    def collect_project_probe(
        project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]]:
        """Return one project-level probe for gate-backed rules."""
        return [(rule, FlextInfraEnforcementCollectionBase.probe_for_path(project_dir))]

    @staticmethod
    def stub_file_paths(project_dir: Path) -> tuple[Path, ...]:
        """Return source stub files while respecting canonical excluded dirs."""
        paths: set[Path] = set()
        for path in project_dir.rglob("*.pyi"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(project_dir).parts
            if any(part in c.Infra.ITERATION_EXCLUDED_PARTS for part in relative_parts):
                continue
            paths.add(path.resolve())
        return tuple(sorted(paths))

    @staticmethod
    def probe_for_path(path: Path) -> p.AttributeProbe:
        """Build the minimal structural probe consumed by fixer adapters."""
        return SimpleNamespace(file_path=str(path))

    @staticmethod
    def collection_failure(
        project_dir: Path, rule: me.EnforcementRuleSpec, message: str
    ) -> m.Infra.FailedFix:
        """Build a failed-fix record for collection/routing errors."""
        return m.Infra.FailedFix(
            rule_id=rule.id, file_path=str(project_dir), error=message
        )

    def _empty_failure(
        self, project_dir: Path, rule: me.EnforcementRuleSpec, message: str
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Return a typed empty collection plus one structured failure."""
        return [], [self.collection_failure(project_dir, rule, message)]


__all__: list[str] = [
    "FlextInfraEnforcementCollectionBase",
    "FlextInfraEnforcementEvaluation",
]
