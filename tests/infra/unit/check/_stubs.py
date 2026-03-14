"""Shared stubs and spy helpers for check tests.

Provides lightweight test doubles with monkeypatch-based substitution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace

from flext_infra.check.services import (
    CheckIssue,
    GateExecution,
    ProjectResult,
)

from ...models import m
from ...typings import t


class Spy:
    """Lightweight call-recording spy for monkeypatch substitution."""

    def __init__(
        self,
        return_value: t.Infra.InfraValue = None,
        side_effect: list[t.Infra.InfraValue] | None = None,
    ) -> None:
        self.call_count: int = 0
        self.call_args: (
            tuple[tuple[t.Infra.InfraValue, ...], dict[str, t.Infra.InfraValue]] | None
        ) = None
        self.call_args_list: list[
            tuple[tuple[t.Infra.InfraValue, ...], dict[str, t.Infra.InfraValue]]
        ] = []
        self.called: bool = False
        self._return_value: t.Infra.InfraValue = return_value
        self._side_effect: list[t.Infra.InfraValue] | None = (
            list(side_effect) if side_effect else None
        )

    def __call__(
        self, *args: t.Infra.InfraValue, **kwargs: t.Infra.InfraValue
    ) -> t.Infra.InfraValue:
        self.called = True
        self.call_count += 1
        self.call_args = (args, kwargs)
        self.call_args_list.append((args, kwargs))
        if self._side_effect:
            return self._side_effect.pop(0)
        return self._return_value

    @property
    def kwargs(self) -> dict[str, t.Infra.InfraValue]:
        """Return kwargs from last call."""
        if self.call_args is None:
            return {}
        return self.call_args[1]

    @property
    def args(self) -> tuple[object, ...]:
        """Return positional args from last call."""
        if self.call_args is None:
            return ()
        return self.call_args[0]


def make_cmd_result(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> SimpleNamespace:
    """Create a SimpleNamespace mimicking subprocess result."""
    return SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        exit_code=returncode,
    )


def make_gate_exec(
    gate: str = "lint",
    project: str = "p",
    passed: bool = True,
    issues: list[CheckIssue] | None = None,
) -> GateExecution:
    """Create a _GateExecution with defaults."""
    return GateExecution(
        result=m.Infra.Check.GateResult(
            gate=gate, project=project, passed=passed, errors=[], duration=0.0
        ),
        issues=issues or [],
        raw_output="",
    )


def make_issue(
    file: str = "a.py",
    line: int = 1,
    column: int = 1,
    code: str = "E1",
    message: str = "Error",
) -> CheckIssue:
    """Create a _CheckIssue with defaults."""
    return CheckIssue(
        file=file,
        line=line,
        column=column,
        code=code,
        message=message,
        severity="error",
    )


def make_project(
    name: str = "p",
    gates: dict[str, GateExecution] | None = None,
) -> ProjectResult:
    """Create a _ProjectResult with defaults."""
    return ProjectResult(
        project=name,
        gates=gates or {"lint": make_gate_exec()},
    )


__all__ = [
    "Spy",
    "make_cmd_result",
    "make_gate_exec",
    "make_issue",
    "make_project",
]
