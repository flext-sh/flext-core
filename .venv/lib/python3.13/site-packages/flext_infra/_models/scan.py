"""Domain models for the shared utilities subpackage.

Scan violation and result models used by infrastructure scanning utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from flext_core import m
from flext_infra import t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm
from flext_infra._models.refactor_namespace_enforcer import (
    FlextInfraModelsNamespaceEnforcer,
)


class FlextInfraModelsScan:
    """Shared utility domain models for scanning and analysis."""

    class DetectorContext(m.ArbitraryTypesModel):
        """Bundles common parameters passed to every detect_file classmethod."""

        file_path: Path = m.Field(
            description="Filesystem path of the file being scanned."
        )
        rope_project: t.Infra.RopeProject = m.Field(
            description="Initialized Rope project for semantic metadata."
        )
        parse_failures: Annotated[
            (
                t.MutableSequenceOf[
                    FlextInfraModelsNamespaceEnforcer.ParseFailureViolation
                ]
                | None
            ),
            m.SkipValidation,
            m.Field(
                description="Shared parse-failure collector across detector passes."
            ),
        ] = None
        project_name: Annotated[
            str, m.Field(description="Optional project name for the scanned file.")
        ] = ""
        project_root: Annotated[
            Path | None,
            m.Field(description="Optional project root containing the scanned file."),
        ] = None

    class ScanViolation(mm.PositiveLineMixin, m.ContractModel):
        """A single violation found during file scanning."""

        message: Annotated[
            str, m.Field(description="Human-readable violation description")
        ]
        severity: Annotated[str, m.Field(description="Violation severity level")]
        rule_id: Annotated[
            str | None, m.Field(description="Optional rule identifier")
        ] = None

    class ScanResult(m.ArbitraryTypesModel):
        """Result of scanning a single file."""

        @staticmethod
        def _violations_default() -> list[FlextInfraModelsScan.ScanViolation]:
            """Violations default."""
            return []

        file_path: Annotated[Path, m.Field(description="Path to the scanned file")]
        violations: Annotated[
            list[FlextInfraModelsScan.ScanViolation],
            m.Field(
                default_factory=_violations_default,
                description="Violations found in the file",
            ),
        ]
        detector_name: Annotated[
            str, m.Field(description="Name of the detector that produced this result")
        ]


__all__: list[str] = ["FlextInfraModelsScan"]
