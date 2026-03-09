"""FLEXT infrastructure check services."""

from __future__ import annotations

from typing import TypeAlias

from flext_infra import m
from flext_infra.check.fix_pyrefly_config import FlextInfraConfigFixer
from flext_infra.check.workspace_check import FlextInfraWorkspaceChecker, run_cli

_CheckIssue: TypeAlias = m.Infra.Check.Issue
_GateExecution: TypeAlias = m.Infra.Check.GateExecution
_ProjectResult: TypeAlias = m.Infra.Check.ProjectResult

__all__ = [
    "FlextInfraConfigFixer",
    "FlextInfraWorkspaceChecker",
    "_CheckIssue",
    "_GateExecution",
    "_ProjectResult",
    "run_cli",
]
