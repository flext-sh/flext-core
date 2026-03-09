from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


class WorkspaceFactoryLike(Protocol):
    def create_minimal(self, tmp_path: Path, name: str = "test-proj") -> Path: ...

    def create_full(self, tmp_path: Path, name: str) -> Path: ...


@dataclass(config=ConfigDict(frozen=True))
class EmptyScenario:
    workspace_name: str = "workspace"

    def build(self, factory: WorkspaceFactoryLike, tmp_path: Path) -> Path:
        _ = factory
        root = tmp_path / self.workspace_name
        root.mkdir(parents=True, exist_ok=True)
        return root


@dataclass(config=ConfigDict(frozen=True))
class MinimalScenario:
    project_name: str = "test-proj"

    def build(self, factory: WorkspaceFactoryLike, tmp_path: Path) -> Path:
        return factory.create_minimal(tmp_path=tmp_path, name=self.project_name)


@dataclass(config=ConfigDict(frozen=True))
class FullScenario:
    project_name: str = "full-proj"

    def build(self, factory: WorkspaceFactoryLike, tmp_path: Path) -> Path:
        return factory.create_full(tmp_path=tmp_path, name=self.project_name)


@dataclass(config=ConfigDict(frozen=True))
class BrokenScenario:
    project_name: str = "broken-proj"

    def build(self, factory: WorkspaceFactoryLike, tmp_path: Path) -> Path:
        project_root = factory.create_minimal(tmp_path=tmp_path, name=self.project_name)
        (project_root / "Makefile").unlink(missing_ok=True)
        return project_root


__all__ = ["BrokenScenario", "EmptyScenario", "FullScenario", "MinimalScenario"]
