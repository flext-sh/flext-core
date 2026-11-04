# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/document
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._project_metadata import (
    FlextProtocolsProjectMetadataContract,
    FlextProtocolsProjectMetadataPyprojectIngress,
)

if TYPE_CHECKING:
    from flext_core._protocols._project_metadata import (
        FlextProtocolsProjectMetadataAggregates,
    )


class FlextProtocolsProjectMetadataDocument:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class PyprojectTool(
        FlextProtocolsProjectMetadataPyprojectIngress.FlextModelsPyprojectIngressContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def flext(self) -> FlextProtocolsProjectMetadataAggregates.ProjectToolFlext:
            """Validated FLEXT project policy."""
            ...

    @runtime_checkable
    class ProjectMetadata(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> Path:
            """Project root."""
            ...

        @property
        def package_name(self) -> str:
            """Import package."""
            ...

        @property
        def class_stem(self) -> str:
            """Class stem."""
            ...

        @property
        def project(self) -> FlextProtocolsProjectMetadataAggregates.Project:
            """Exact validated PEP 621 project object."""
            ...

        @property
        def flext(self) -> FlextProtocolsProjectMetadataAggregates.ProjectToolFlext:
            """Exact validated tool.flext object."""
            ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataDocument",
]
