"""CLI-oriented test doubles for github test modules.

Stubs for PR management, syncing, linting, workspace orchestration,
and utility facades used by CLI integration tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from flext_core import r
from tests.infra.typings import t


class StubPrManager:
    """Stub for FlextInfraPrManager used in CLI tests."""

    def __init__(
        self,
        status_returns: Sequence[t.ContainerValue] | None = None,
        create_returns: Sequence[t.ContainerValue] | None = None,
        view_returns: Sequence[t.ContainerValue] | None = None,
        checks_returns: Sequence[t.ContainerValue] | None = None,
        merge_returns: Sequence[t.ContainerValue] | None = None,
        close_returns: Sequence[t.ContainerValue] | None = None,
    ) -> None:
        self._status = list(status_returns or [])
        self._create = list(create_returns or [])
        self._view = list(view_returns or [])
        self._checks = list(checks_returns or [])
        self._merge = list(merge_returns or [])
        self._close = list(close_returns or [])

    def _pop(self, returns: list[t.ContainerValue]) -> t.ContainerValue:
        if not returns:
            return r[bool].fail("no return configured")
        return returns[0] if len(returns) == 1 else returns.pop(0)

    def status(
        self, *_a: t.ContainerValue, **_kw: t.ContainerValue
    ) -> t.ContainerValue:
        return self._pop(self._status)

    def create(
        self, *_a: t.ContainerValue, **_kw: t.ContainerValue
    ) -> t.ContainerValue:
        return self._pop(self._create)

    def view(self, *_a: t.ContainerValue, **_kw: t.ContainerValue) -> t.ContainerValue:
        return self._pop(self._view)

    def checks(
        self, *_a: t.ContainerValue, **_kw: t.ContainerValue
    ) -> t.ContainerValue:
        return self._pop(self._checks)

    def merge(self, *_a: t.ContainerValue, **_kw: t.ContainerValue) -> t.ContainerValue:
        return self._pop(self._merge)

    def close(self, *_a: t.ContainerValue, **_kw: t.ContainerValue) -> t.ContainerValue:
        return self._pop(self._close)


class StubSyncer:
    """Stub for FlextInfraWorkflowSyncer used in CLI tests."""

    def __init__(self, sync_returns: object | None = None) -> None:
        self._sync_returns = (
            sync_returns
            if sync_returns is not None
            else r[list[t.ContainerValue]].ok([])
        )
        self.sync_workspace_calls: list[dict[str, t.ContainerValue]] = []

    def sync_workspace(self, **kwargs: t.ContainerValue) -> t.ContainerValue:
        self.sync_workspace_calls.append(kwargs)
        return self._sync_returns


class StubLinter:
    """Stub for FlextInfraWorkflowLinter used in CLI tests."""

    def __init__(self, lint_returns: object | None = None) -> None:
        self._lint_returns = (
            lint_returns if lint_returns is not None else r[bool].ok(True)
        )
        self.lint_calls: list[dict[str, t.ContainerValue]] = []

    def lint(self, **kwargs: t.ContainerValue) -> t.ContainerValue:
        self.lint_calls.append(kwargs)
        return self._lint_returns


class StubWorkspaceManager:
    """Stub for FlextInfraPrWorkspaceManager used in CLI tests."""

    def __init__(self, orchestrate_returns: object | None = None) -> None:
        self._orchestrate_returns = (
            orchestrate_returns if orchestrate_returns is not None else r[bool].ok(True)
        )
        self.orchestrate_calls: list[dict[str, t.ContainerValue]] = []

    def orchestrate(self, **kwargs: t.ContainerValue) -> t.ContainerValue:
        self.orchestrate_calls.append(kwargs)
        return self._orchestrate_returns


class StubUtilities:
    """Stub for u (FlextInfraUtilities) used in CLI tests."""

    class Infra:
        """Stub for u.Infra namespace."""

        _git_branch_returns: t.ContainerValue = None

        @classmethod
        def git_current_branch(
            cls, *_a: t.ContainerValue, **_kw: t.ContainerValue
        ) -> t.ContainerValue:
            return cls._git_branch_returns or r[str].ok("feature")

        @classmethod
        def git_has_changes(
            cls, *_a: t.ContainerValue, **_kw: t.ContainerValue
        ) -> t.ContainerValue:
            return r[bool].ok(True)

        @classmethod
        def git_checkout(
            cls, *_a: t.ContainerValue, **_kw: t.ContainerValue
        ) -> t.ContainerValue:
            return r[bool].ok(True)

        @classmethod
        def git_add(
            cls, *_a: t.ContainerValue, **_kw: t.ContainerValue
        ) -> t.ContainerValue:
            return r[bool].ok(True)

        @classmethod
        def git_commit(
            cls, *_a: t.ContainerValue, **_kw: t.ContainerValue
        ) -> t.ContainerValue:
            return r[bool].ok(True)


__all__ = [
    "StubLinter",
    "StubPrManager",
    "StubSyncer",
    "StubUtilities",
    "StubWorkspaceManager",
]
