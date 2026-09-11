"""Command payload mixin for flext-infra service bases."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraCommandPayloadMixin:
    """Private FLEXT payload builder for shared command services."""

    if TYPE_CHECKING:
        repository_root: Path
        apply_changes: bool
        check_only: bool
        dry_run: bool

        output_format: str
        project_filter: str | None
        report_path: Path | None
        output_dir: Path | None

    def command_payload(self) -> t.JsonMapping:
        """Return the normalized shared command payload once."""
        payload: t.MutableJsonMapping = {
            "repository_root": str(self.repository_root),
            "apply_changes": self.apply_changes,
            "check_only": self.check_only,
            "dry_run": self.dry_run,
            "output_format": self.output_format,
        }
        if self.project_filter is not None:
            payload["project_filter"] = self.project_filter
        if self.report_path is not None:
            payload["report_path"] = str(self.report_path)
        if self.output_dir is not None:
            payload["output_dir"] = str(self.output_dir)
        return payload


__all__: list[str] = ["FlextInfraCommandPayloadMixin"]
