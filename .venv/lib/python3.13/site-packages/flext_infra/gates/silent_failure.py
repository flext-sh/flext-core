"""Silent failure quality gate.

Enforces silent-failure detection across every Python project the workspace
discovers. Projects without Python sources provide an empty scan input; no
project-name allowlist exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import m
from flext_infra.detectors.silent_failure_detector import (
    FlextInfraSilentFailureDetector,
)
from flext_infra.gates.base_gate import FlextInfraGate, FlextInfraScannerGateMixin

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraSilentFailureGate(FlextInfraScannerGateMixin, FlextInfraGate):
    """Block silent failure sentinels in any Python project under the workspace."""

    gate_id: ClassVar[str] = "silent-failure"
    gate_name: ClassVar[str] = "Silent Failure"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = "Flext Silent Failure Detector"
    tool_url: ClassVar[str] = "internal://flext-infra/silent-failure"
    scan_error_message: ClassVar[str] = "silent-failure scan failed"

    @override
    def _detect_file_issues(
        self, file_path: Path, project_dir: Path, rope_project: t.Infra.RopeProject
    ) -> t.SequenceOf[m.Infra.Issue]:
        """Detect silent failure violations in a single file."""
        return FlextInfraSilentFailureDetector.detect_file(
            m.Infra.DetectorContext(
                file_path=file_path, project_root=project_dir, rope_project=rope_project
            )
        )


__all__: list[str] = ["FlextInfraSilentFailureGate"]
