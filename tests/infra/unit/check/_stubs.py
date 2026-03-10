"""Shared stubs and spy helpers for check tests.

Provides lightweight test doubles with monkeypatch-based substitution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import SimpleNamespace

from flext_core import t
from flext_infra import m
from flext_infra.check.services import (
    _CheckIssue,
    _GateExecution,
    _ProjectResult,
)


class Spy:
    """Lightweight call-recording spy for monkeypatch substitution."""

    def __init__(
        self,
        return_value: t.ContainerValue = None,
        side_effect: list[t.ContainerValue] | None = None,
    ) -> None:
        self.call_count: int = 0
        self.call_args: (
            tuple[tuple[t.ContainerValue, ...], dict[str, t.ContainerValue]] | None
        ) = None
        self.call_args_list: list[
            tuple[tuple[t.ContainerValue, ...], dict[str, t.ContainerValue]]
        ] = []
        self.called: bool = False
        self._return_value = return_value
        self._side_effect = list(side_effect) if side_effect else None

    def __call__(
        self, *args: t.ContainerValue, **kwargs: t.ContainerValue
    ) -> t.ContainerValue:
        self.called = True
        self.call_count += 1
        self.call_args = (args, kwargs)
        self.call_args_list.append((args, kwargs))
        if self._side_effect:
            return self._side_effect.pop(0)
        return self._return_value

    @property
    def kwargs(self) -> dict[str, t.ContainerValue]:
        """Return kwargs from last call."""
        if self.call_args is None:
            return {}
        return self.call_args[1]

    @property
    def args(self) -> tuple[t.ContainerValue, ...]:
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
    issues: list[_CheckIssue] | None = None,
) -> _GateExecution:
    """Create a _GateExecution with defaults."""
    return _GateExecution(
        result=m.Infra.Check.GateResult(gate=gate, project=project, passed=passed),
        issues=issues or [],
    )


def make_issue(
    file: str = "a.py",
    line: int = 1,
    column: int = 1,
    code: str = "E1",
    message: str = "Error",
) -> _CheckIssue:
    """Create a _CheckIssue with defaults."""
    return _CheckIssue(file=file, line=line, column=column, code=code, message=message)


def make_project(
    name: str = "p",
    gates: dict[str, _GateExecution] | None = None,
) -> _ProjectResult:
    """Create a _ProjectResult with defaults."""
    return _ProjectResult(
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
