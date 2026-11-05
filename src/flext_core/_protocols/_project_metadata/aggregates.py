# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/aggregates
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flext_core
from flext_core._protocols._project_metadata import (
    FlextProtocolsProjectMetadataContract,
    FlextProtocolsProjectMetadataPyprojectIngress,
)

if TYPE_CHECKING:
    from flext_core._protocols._project_metadata import (
        FlextProtocolsProjectMetadataFields,
    )


class FlextProtocolsProjectMetadataAggregates:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Project(
        FlextProtocolsProjectMetadataPyprojectIngress.FlextModelsPyprojectIngressContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def version(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def description(self) -> str:
            """Structural member projected from the owning model."""
            ...

        @property
        def authors(
            self,
        ) -> tuple[FlextProtocolsProjectMetadataFields.ProjectAuthor, ...]:
            """Project authors."""
            ...

        @property
        def urls(self) -> FlextProtocolsProjectMetadataFields.ProjectUrls:
            """Project URLs."""
            ...

        @property
        def requires_python(self) -> str:
            """Python constraint."""
            ...

        @property
        def dependencies(self) -> flext_core.t.StrTuple:
            """PEP 508 runtime dependency declarations."""
            ...

        @property
        def classifiers(self) -> flext_core.t.StrTuple:
            """Trove classifiers."""
            ...

        @property
        def keywords(self) -> flext_core.t.StrTuple:
            """Project search keywords."""
            ...

    @runtime_checkable
    class ProjectToolFlext(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def project(
            self,
        ) -> FlextProtocolsProjectMetadataFields.ProjectToolFlextProject:
            """Project naming policy."""
            ...

        @property
        def docs(self) -> FlextProtocolsProjectMetadataFields.ProjectToolFlextDocs:
            """Documentation policy."""
            ...

        @property
        def workspace(
            self,
        ) -> FlextProtocolsProjectMetadataFields.ProjectToolFlextWorkspace:
            """Workspace attachment policy."""
            ...

        @property
        def namespace(
            self,
        ) -> FlextProtocolsProjectMetadataFields.ProjectToolFlextNamespace:
            """Namespace enforcement policy."""
            ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataAggregates",
]
