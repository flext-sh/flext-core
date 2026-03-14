"""Lightweight test doubles for github test modules.

Replaces generic mock objects with typed, configurable stub classes
that satisfy flext_infra protocol contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, override

from pydantic import BaseModel, Field, JsonValue

from flext_core import FlextModels, r
from flext_infra import (
    FlextInfraUtilitiesIo,
    FlextInfraUtilitiesReporting,
    FlextInfraUtilitiesSelection,
    FlextInfraUtilitiesTemplates,
    FlextInfraUtilitiesVersioning,
    m,
)

from ._stubs_extra import (
    StubLinter,
    StubPrManager,
    StubSyncer,
    StubUtilities,
    StubWorkspaceManager,
)


class StubCommandOutput(FlextModels.ArbitraryTypesModel):
    """Stub command output matching p.Infra.CommandOutput protocol."""

    exit_code: Annotated[int, Field(default=0, description="Command exit code")] = 0
    stdout: Annotated[str, Field(default="", description="Captured stdout")] = ""
    stderr: Annotated[str, Field(default="", description="Captured stderr")] = ""


class StubRunner:
    """Configurable stub for p.Infra.CommandRunner protocol."""

    def __init__(
        self,
        run_returns: Sequence[r[m.Infra.Core.CommandOutput]] | None = None,
        capture_returns: Sequence[r[str]] | None = None,
        run_checked_returns: Sequence[r[bool]] | None = None,
        run_to_file_returns: Sequence[r[int]] | None = None,
    ) -> None:
        self._run_returns = list(run_returns or [])
        self._capture_returns = list(capture_returns or [])
        self._run_checked_returns = list(run_checked_returns or [])
        self._run_to_file_returns = list(run_to_file_returns or [])
        self.run_calls: list[list[str]] = []
        self.capture_calls: list[list[str]] = []
        self.run_checked_calls: list[list[str]] = []
        self.run_to_file_calls: list[list[str]] = []

    @staticmethod
    def _pop_run(
        returns: list[r[m.Infra.Core.CommandOutput]],
    ) -> r[m.Infra.Core.CommandOutput]:
        if not returns:
            return r[m.Infra.Core.CommandOutput].fail("no return value configured")
        return returns[0] if len(returns) == 1 else returns.pop(0)

    @staticmethod
    def _pop_capture(returns: list[r[str]]) -> r[str]:
        if not returns:
            return r[str].fail("no return value configured")
        return returns[0] if len(returns) == 1 else returns.pop(0)

    @staticmethod
    def _pop_checked(returns: list[r[bool]]) -> r[bool]:
        if not returns:
            return r[bool].fail("no return value configured")
        return returns[0] if len(returns) == 1 else returns.pop(0)

    @staticmethod
    def _pop_to_file(returns: list[r[int]]) -> r[int]:
        if not returns:
            return r[int].fail("no return value configured")
        return returns[0] if len(returns) == 1 else returns.pop(0)

    def run(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]:
        _ = cwd, timeout, env
        self.run_calls.append(list(cmd))
        return self._pop_run(self._run_returns)

    def capture(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[str]:
        _ = cwd, timeout, env
        self.capture_calls.append(list(cmd))
        return self._pop_capture(self._capture_returns)

    def run_checked(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[bool]:
        _ = cwd, timeout, env
        self.run_checked_calls.append(list(cmd))
        return self._pop_checked(self._run_checked_returns)

    def run_to_file(
        self,
        cmd: Sequence[str],
        output_file: Path,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[int]:
        _ = output_file, cwd, timeout, env
        self.run_to_file_calls.append(list(cmd))
        return self._pop_to_file(self._run_to_file_returns)

    def run_raw(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]:
        return self.run(cmd, cwd=cwd, timeout=timeout, env=env)


class StubJsonIo(FlextInfraUtilitiesIo):
    """Stub for FlextInfraUtilitiesIo (json_io dependency)."""

    write_json_returns: ClassVar[r[bool]] = r[bool].ok(True)
    write_json_calls: ClassVar[list[tuple[Path, object]]] = []

    def __init__(self, write_returns: r[bool] | None = None) -> None:
        StubJsonIo.write_json_returns = write_returns or r[bool].ok(True)
        StubJsonIo.write_json_calls = []

    @staticmethod
    @override
    def write_json(
        path: Path,
        payload: JsonValue | BaseModel | Mapping[str, JsonValue] | Sequence[JsonValue],
        *,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> r[bool]:
        _ = sort_keys, ensure_ascii, indent
        StubJsonIo.write_json_calls.append((path, payload))
        return StubJsonIo.write_json_returns


class StubVersioning(FlextInfraUtilitiesVersioning):
    """Stub for FlextInfraUtilitiesVersioning."""

    _release_tag_returns: ClassVar[r[str]] = r[str].fail("no tag")

    def __init__(self, release_tag_returns: r[str] | None = None) -> None:
        StubVersioning._release_tag_returns = release_tag_returns or r[str].fail(
            "no tag"
        )

    @staticmethod
    @override
    def release_tag_from_branch(branch: str) -> r[str]:
        _ = branch
        return StubVersioning._release_tag_returns


class StubSelector(FlextInfraUtilitiesSelection):
    """Stub for FlextInfraUtilitiesSelection."""

    _resolve_returns: ClassVar[r[list[m.Infra.Workspace.ProjectInfo]]] = r[
        list[m.Infra.Workspace.ProjectInfo]
    ].ok([])

    def __init__(
        self,
        resolve_returns: r[list[m.Infra.Workspace.ProjectInfo]] | None = None,
    ) -> None:
        StubSelector._resolve_returns = (
            resolve_returns
            if resolve_returns is not None
            else r[list[m.Infra.Workspace.ProjectInfo]].ok([])
        )

    @staticmethod
    @override
    def resolve_projects(
        workspace_root: Path,
        names: list[str],
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        _ = workspace_root, names
        return StubSelector._resolve_returns


class StubReporting(FlextInfraUtilitiesReporting):
    """Stub for FlextInfraUtilitiesReporting."""

    _report_dir: ClassVar[Path] = Path("/tmp/reports")

    def __init__(self, report_dir: Path | None = None) -> None:
        StubReporting._report_dir = report_dir or Path("/tmp/reports")

    @staticmethod
    @override
    def get_report_dir(
        root: Path | str,
        scope: str,
        verb: str,
    ) -> Path:
        _ = root, scope, verb
        return StubReporting._report_dir


class StubTemplates(FlextInfraUtilitiesTemplates):
    """Stub for FlextInfraUtilitiesTemplates."""

    GENERATED_SHELL_HEADER: ClassVar[str] = "# GENERATED by {source}\n"


class StubProjectInfo(m.Infra.Workspace.ProjectInfo):
    """Stub for p.Infra.ProjectInfo protocol."""

    name: Annotated[str, Field(default="test-project", description="Project name")] = (
        "test-project"
    )
    path: Annotated[
        Path, Field(default=Path("/tmp/test-project"), description="Project path")
    ] = Path("/tmp/test-project")
    stack: Annotated[
        str, Field(default="python", description="Primary technology stack")
    ] = "python"
    has_tests: Annotated[
        bool, Field(default=False, description="Project has tests")
    ] = False
    has_src: Annotated[bool, Field(default=True, description="Project has source")] = (
        True
    )


__all__ = [
    "StubCommandOutput",
    "StubJsonIo",
    "StubLinter",
    "StubPrManager",
    "StubProjectInfo",
    "StubReporting",
    "StubRunner",
    "StubSelector",
    "StubSyncer",
    "StubTemplates",
    "StubUtilities",
    "StubVersioning",
    "StubWorkspaceManager",
]
