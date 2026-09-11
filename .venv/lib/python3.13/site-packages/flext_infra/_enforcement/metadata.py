"""Declarative enforcement metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra.refactor.declarative_enforcement import (
    FlextInfraRefactorDeclarativeEnforcement,
)

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import m, p, t


class FlextInfraEnforcementMetadata:
    """Shared rendering metadata for declarative enforcement probes."""

    @staticmethod
    def detect_declarative(
        rule: me.EnforcementRuleSpec, ctx: m.Infra.DetectorContext
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Detect one declarative rule for one detector context."""
        return FlextInfraRefactorDeclarativeEnforcement.detect(rule, ctx)

    @staticmethod
    def violation_kind(rule: me.EnforcementRuleSpec) -> str:
        """Return the normalized violation kind declared by catalog metadata."""
        source = rule.source
        if source.kind == "flext_infra_detector":
            raw = source.violation_field
        elif source.kind == "beartype":
            predicate_kind = source.predicate_kind
            raw = str(getattr(predicate_kind, "value", predicate_kind))
        else:
            raw = rule.id.lower().replace("-", "_")
        return (
            raw.removesuffix("_violations").removesuffix("_violation") or "declarative"
        )

    @staticmethod
    def object_kind(kind: str) -> str:
        """Return a stable object kind from a normalized violation kind."""
        return kind.rsplit("_", maxsplit=1)[-1] if kind else "object"

    @staticmethod
    def object_name(probe: p.AttributeProbe, kind: str) -> str:
        """Return a stable object name from a declarative probe."""
        name = getattr(probe, "object_name", "")
        if name:
            return str(name)
        literal = getattr(probe, "literal", "")
        if literal:
            return f"{kind}_{literal}"
        file_path = getattr(probe, "file_path", "")
        if file_path:
            return Path(str(file_path)).name
        return kind or "declarative"

    @staticmethod
    def description(
        rule: me.EnforcementRuleSpec, probe: p.AttributeProbe, object_name: str
    ) -> str:
        """Return a human-readable description for a declarative violation."""
        base = rule.description
        literal = getattr(probe, "literal", "")
        if literal:
            return f"{base}: {literal}"
        if object_name and object_name != rule.id.lower().replace("-", "_"):
            return f"{base}: {object_name}"
        return base


__all__: list[str] = ["FlextInfraEnforcementMetadata"]
