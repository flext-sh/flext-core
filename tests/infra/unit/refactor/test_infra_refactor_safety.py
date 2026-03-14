"""Unit tests for FlextInfraRefactorSafetyManager integration."""

from __future__ import annotations

from pathlib import Path
from typing import overload, override

import pytest

from flext_core import r

try:
    from flext_infra.refactor import (
        FlextInfraRefactorEngine,
    )
except ImportError as exc:
    pytest.skip(f"refactor package unavailable: {exc}", allow_module_level=True)
from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager
from flext_tests import tm


class EngineSafetyStub(FlextInfraRefactorSafetyManager):
    """Test double for safety manager lifecycle operations."""

    def __init__(self) -> None:
        """Initialize call capture state for assertions."""
        super().__init__()
        self.calls: list[str] = []

    @override
    def create_pre_transformation_stash(
        self,
        workspace_root: Path,
        *,
        label: str = "flext-refactor-pre-transform",
    ) -> r[str]:
        _ = workspace_root
        _ = label
        self.calls.append("stash")
        return r[str].ok("stash@{0}")

    @override
    def save_checkpoint_state(
        self,
        workspace_root: Path,
        *,
        status: str,
        stash_ref: str,
        processed_targets: list[str],
    ) -> r[bool]:
        _ = workspace_root
        _ = status
        _ = stash_ref
        _ = processed_targets
        self.calls.append("checkpoint")
        return r[bool].ok(True)

    @override
    def run_semantic_validation(self, workspace_root: Path) -> r[bool]:
        _ = workspace_root
        self.calls.append("validate")
        return r[bool].ok(True)

    @override
    def clear_checkpoint(self) -> r[bool]:
        self.calls.append("clear")
        return r[bool].ok(True)

    @override
    def request_emergency_stop(self, reason: str) -> None:
        _ = reason
        self.calls.append("stop")

    @overload
    def rollback(self, workspace_root: Path, stash_ref: str = "") -> r[bool]: ...

    @overload
    def rollback(self, workspace_root: str, /) -> None: ...

    @override
    def rollback(
        self,
        workspace_root: Path | str,
        stash_ref: str = "",
    ) -> r[bool] | None:
        _ = stash_ref
        self.calls.append("rollback")
        if isinstance(workspace_root, Path):
            return r[bool].ok(True)
        return None

    @override
    def is_emergency_stop_requested(self) -> bool:
        self.calls.append("is_stop")
        return False


def test_refactor_project_integrates_safety_manager(tmp_path: Path) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.yml").write_text(
        "\nrules:\n  - id: ensure-future-annotations\n    enabled: true\n    fix_action: ensure_future_annotations\n".strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text('refactor_engine:\n  project_scan_dirs: ["src"]\n')
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "sample.py").write_text("import os\n", encoding="utf-8")
    engine = FlextInfraRefactorEngine(config_path=config_path)
    stub = EngineSafetyStub()
    engine.safety_manager = stub
    loaded = engine.load_rules()
    tm.ok(loaded)
    results = engine.refactor_project(tmp_path, dry_run=False, apply_safety=True)
    assert results
    assert all(item.success for item in results)
    assert stub.calls == ["stash", "checkpoint", "validate", "clear"]
