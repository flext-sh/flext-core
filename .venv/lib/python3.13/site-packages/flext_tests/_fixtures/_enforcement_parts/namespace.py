"""Canonical namespace-detector builder owned by the enforcement plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import m, p
from flext_tests._fixtures._enforcement_parts.items import (
    EnforcementCollector,
    EnforcementItem,
)
from flext_tests._fixtures._enforcement_parts.validators import dispatch_infra_detector

if TYPE_CHECKING:
    import pytest


class NamespaceDetectorBuilder(p.Tests.EnforcementBuilder):
    """Build enforcement items from the canonical namespace report."""

    def __call__(
        self,
        session: pytest.Session,
        cfg: m.Tests.EnforcementDispatcherConfig,
        rule: m.EnforcementRuleSpec,
        context: m.Tests.EnforcementBuildContext,
    ) -> list[pytest.Item]:
        """Return one synthetic item per project carrying violations."""
        _ = cfg
        if context.infra_report is None:
            return []
        grouped = dispatch_infra_detector(rule, context.infra_report)
        collector = EnforcementCollector.from_parent(
            parent=session, name="flext-enforcement"
        )
        return [
            EnforcementItem.from_parent(
                name=f"{rule.id}[{project}]",
                parent=collector,
                rule=rule,
                project=project,
                violations=tuple(violations),
            )
            for project, violations in grouped.items()
            if violations
        ]


__all__: list[str] = ["NamespaceDetectorBuilder"]
