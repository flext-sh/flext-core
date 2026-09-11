"""Deferred self-reference quality gate.

Blocks two structural defects that share one root cause — a nested model that
cannot name what it depends on while its own class body executes:

* ``default_factory=lambda: Outer.Sibling()`` postpones a name resolution the
  definition order should have satisfied, turning a structural defect into a
  runtime one that no static gate can see.
* a field annotated with its own owner cannot be instantiated while that
  owner is still incomplete.

The canonical repair is diamond-FLEXT composition: hoist the referenced model
into its own namespace class, inherit that namespace, and reference the model
as a resolved base-class attribute.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import m
from flext_infra.detectors.deferred_self_reference_detector import (
    FlextInfraDeferredSelfReferenceDetector,
)
from flext_infra.gates.base_gate import FlextInfraGate, FlextInfraScannerGateMixin

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraDeferredSelfReferenceGate(FlextInfraScannerGateMixin, FlextInfraGate):
    """Block deferred self-reference and recursive models in any Python project."""

    gate_id: ClassVar[str] = "deferred-self-reference"
    gate_name: ClassVar[str] = "Deferred Self Reference"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = "Flext Deferred Self Reference Detector"
    tool_url: ClassVar[str] = "internal://flext-infra/deferred-self-reference"
    scan_error_message: ClassVar[str] = "deferred-self-reference scan failed"

    @override
    def _detect_file_issues(
        self, file_path: Path, project_dir: Path, rope_project: t.Infra.RopeProject
    ) -> t.SequenceOf[m.Infra.Issue]:
        """Detect deferred self-reference violations in a single file."""
        return FlextInfraDeferredSelfReferenceDetector.detect_file(
            m.Infra.DetectorContext(
                file_path=file_path, project_root=project_dir, rope_project=rope_project
            )
        )


__all__: list[str] = ["FlextInfraDeferredSelfReferenceGate"]
