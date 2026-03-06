"""Validation phase for migrate-to-mro."""

from __future__ import annotations

from pathlib import Path

from .scanner import FlextInfraRefactorMROScanner


class FlextInfraRefactorMROValidator:
    @classmethod
    def validate(cls, *, workspace_root: Path, target: str) -> tuple[int, int]:
        file_results, _ = FlextInfraRefactorMROScanner.scan_workspace(
            workspace_root=workspace_root,
            target=target,
        )
        remaining = sum(len(item.candidates) for item in file_results)
        return remaining, 0


__all__ = ["FlextInfraRefactorMROValidator"]
