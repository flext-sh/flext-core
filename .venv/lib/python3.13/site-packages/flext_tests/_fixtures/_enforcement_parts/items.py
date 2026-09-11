"""Pytest collection items for enforcement violations."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from flext_tests import m, p, t


class EnforcementItem(pytest.Item):
    """Pytest item representing one ``(project, rule_id)`` violation group."""

    def __init__(
        self,
        name: str,
        parent: pytest.Collector,
        *,
        rule: m.EnforcementRuleSpec,
        project: str,
        violations: t.SequenceOf[m.Violation | p.AttributeProbe],
    ) -> None:
        super().__init__(name, parent)
        self._rule = rule
        self._project = project
        self._violations = tuple(violations)

    @override
    def runtest(self) -> None:
        if not self._violations:
            return
        detail_lines = [f"  - {self._format_violation(v)}" for v in self._violations]
        header = (
            f"{self._rule.id} ({self._rule.severity}) in {self._project}: "
            f"{len(self._violations)} violation(s)"
        )
        raise EnforcementViolationError("\n".join([header, *detail_lines]))

    @staticmethod
    def _format_violation(violation: p.AttributeProbe) -> str:
        rule_id = getattr(violation, "rule_id", "")
        file_path = getattr(violation, "file_path", None) or getattr(
            violation, "file", ""
        )
        line = getattr(violation, "line_number", None) or getattr(violation, "line", "")
        description = (
            getattr(violation, "description", None)
            or getattr(violation, "detail", None)
            or getattr(violation, "suggestion", None)
            or ""
        )
        parts = [str(rule_id) if rule_id else ""]
        if file_path:
            parts.append(str(file_path))
        if line:
            parts.append(f"line {line}")
        if description:
            parts.append(str(description))
        return " | ".join(part for part in parts if part)

    @override
    def repr_failure(
        self, excinfo: pytest.ExceptionInfo[BaseException], style: str | None = None
    ) -> str:
        _ = style
        return str(excinfo.value)

    @override
    def reportinfo(self) -> tuple[Path | str, int | None, str]:
        return (
            f"flext-enforce::{self._rule.id}",
            None,
            f"{self._rule.id} [{self._project}] {self._rule.description}",
        )


class EnforcementCollector(pytest.Collector):
    """Synthetic collector that owns every ``EnforcementItem`` for the session."""

    def __init__(
        self,
        name: str,
        parent: pytest.Session,
        *,
        items: t.SequenceOf[EnforcementItem] = (),
    ) -> None:
        super().__init__(name, parent)
        self._items: list[EnforcementItem] = list(items)

    def add(self, item: EnforcementItem) -> None:
        self._items.append(item)

    @override
    def collect(self) -> Iterable[pytest.Item]:
        return list(self._items)


class EnforcementViolationError(Exception):
    """Raised by ``EnforcementItem.runtest`` when violations are present."""


__all__: list[str] = [
    "EnforcementCollector",
    "EnforcementItem",
    "EnforcementViolationError",
]
