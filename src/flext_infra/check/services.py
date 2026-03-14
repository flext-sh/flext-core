"""FLEXT infrastructure check services."""

from __future__ import annotations

from typing import TypeAlias

from flext_infra import m
from flext_infra.check.fix_pyrefly_config import FlextInfraConfigFixer
from flext_infra.check.workspace_check import FlextInfraWorkspaceChecker, run_cli

CheckIssue: TypeAlias = m.Infra.Check.Issue
GateExecution: TypeAlias = m.Infra.Check.GateExecution
ProjectResult: TypeAlias = m.Infra.Check.ProjectResult

__all__ = [
    "CheckIssue",
    "FlextInfraConfigFixer",
    "FlextInfraWorkspaceChecker",
    "GateExecution",
    "ProjectResult",
    "run_cli",
]
