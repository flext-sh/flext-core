"""Base adapter interface for enforcement fixers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import m, p, t


class FlextInfraFixerAdapter:
    """Abstract adapter that routes enforcement-catalog violations to fixes.

    Subclasses declare a ``kind`` ClassVar matching ``EnforcementFixAction.kind``
    and implement ``can_fix`` / ``fix_project``.
    """

    kind: ClassVar[str] = ""

    def __init__(self, repository_root: Path) -> None:
        """Bind the repository root used during fix execution."""
        self._repository_root = repository_root

    def can_fix(self, fix_action: me.EnforcementFixAction) -> bool:
        """Return whether this adapter handles ``fix_action``."""
        return fix_action.kind == self.kind

    def fix_project(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Apply fixes for the given violations in ``project_dir``."""
        msg = f"{self.__class__.__name__}.fix_project must be implemented"
        raise NotImplementedError(msg)

    @staticmethod
    def _group_by_target(
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
    ) -> dict[str, list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]]]:
        """Group violations by the fix target declared in their catalog action."""
        grouped: dict[str, list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]]] = {}
        for rule, probe in violations:
            fix_action = rule.fix_action
            if fix_action is None:
                continue
            grouped.setdefault(fix_action.target, []).append((rule, probe))
        return grouped

    @staticmethod
    def _collect_file_paths(
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
    ) -> tuple[Path, ...]:
        """Extract unique existing file paths from violation probes."""
        seen: set[Path] = set()
        paths: list[Path] = []
        for _rule, probe in violations:
            raw_path = getattr(probe, "file_path", None) or getattr(probe, "file", "")
            if not raw_path:
                continue
            file_path = Path(str(raw_path))
            if not file_path.is_absolute():
                file_path = project_dir / file_path
            file_path = file_path.resolve()
            if file_path.is_file() and file_path not in seen:
                seen.add(file_path)
                paths.append(file_path)
        return tuple(paths)

    @staticmethod
    def _rule_id(
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
    ) -> str:
        """Return the first rule id in a grouped violation batch."""
        return violations[0][0].id if violations else ""


__all__: list[str] = ["FlextInfraFixerAdapter"]
