"""Enforcement item construction for the pytest plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import m
from flext_tests._fixtures._enforcement_parts.config import active_rules
from flext_tests._fixtures._enforcement_parts.discovery import (
    collected_project_names,
    collected_validator_targets,
    load_infra_report,
)
from flext_tests._fixtures._enforcement_parts.items import EnforcementCollector
from flext_tests._fixtures._enforcement_parts.namespace import NamespaceDetectorBuilder
from flext_tests._fixtures._enforcement_parts.validators import (
    build_tests_validator_items,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from flext_tests import p, t


def build_items(
    session: pytest.Session,
    cfg: m.Tests.EnforcementDispatcherConfig,
    *,
    collected_items: t.SequenceOf[pytest.Item],
) -> list[pytest.Item]:
    """Build synthetic enforcement items for active collection-time rules."""
    workspace_root = cfg.workspace_root
    if workspace_root is None:
        return []
    rules = active_rules(cfg)
    validator_targets = collected_validator_targets(
        items=collected_items, workspace_root=workspace_root
    )
    infra_report = _load_infra_report_if_needed(rules, workspace_root, collected_items)

    collector = EnforcementCollector.from_parent(
        parent=session, name="flext-enforcement"
    )
    context = m.Tests.EnforcementBuildContext(
        infra_report=infra_report,
        validator_targets=validator_targets,
        workspace_root=workspace_root,
    )
    namespace_builder = NamespaceDetectorBuilder()
    items: list[pytest.Item] = []
    for rule in rules:
        if rule.source.kind == m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR.value:
            items.extend(namespace_builder(session, cfg, rule, context))
        elif rule.source.kind == "flext_tests_validator":
            items.extend(build_tests_validator_items(collector, rule, context))
    return items


def _load_infra_report_if_needed(
    rules: tuple[m.EnforcementRuleSpec, ...],
    workspace_root: Path,
    collected_items: t.SequenceOf[pytest.Item],
) -> p.AttributeProbe | None:
    """Load the workspace infra report only when a rule needs it."""
    if not any(rule.source.kind == "flext_infra_detector" for rule in rules):
        return None
    project_names = collected_project_names(
        items=collected_items, workspace_root=workspace_root
    )
    if not project_names:
        return None
    # NOTE (multi-agent, mro-wkii.17.21): detector failures must stop collection;
    # replacing them with None would silently disable the active enforcement rule.
    return load_infra_report(workspace_root, project_names=project_names).unwrap()


__all__: list[str] = ["build_items"]
