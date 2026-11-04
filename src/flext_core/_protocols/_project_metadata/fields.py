# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/fields
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._project_metadata import (
    FlextProtocolsProjectMetadataContract,
    FlextProtocolsProjectMetadataPyprojectIngress,
)

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsProjectMetadataFields:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ProjectAuthor(
        FlextProtocolsProjectMetadataPyprojectIngress.FlextModelsPyprojectIngressContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Author display name."""
            ...

        @property
        def email(self) -> str:
            """Author email address."""
            ...

    @runtime_checkable
    class ProjectUrls(
        FlextProtocolsProjectMetadataPyprojectIngress.FlextModelsPyprojectIngressContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def homepage(self) -> str:
            """Project homepage URL."""
            ...

        @property
        def documentation(self) -> str:
            """Published documentation URL."""
            ...

        @property
        def repository(self) -> str:
            """Source repository URL."""
            ...

    @runtime_checkable
    class ProjectToolFlextProject(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def class_stem_override(self) -> str | None:
            """Explicit class stem override."""
            ...

        @property
        def budget(self) -> t.JsonMapping | None:
            """Structural member projected from the owning model.

            Per-gate resource budget table emitted by the flext-infra budget-gate
            projection into the managed ``[tool.flext.project.budget]`` section. The
            ingress owns the validated shape so the generator's own output always round-
            trips through this contract.
            """
            ...

        @property
        def duplication(self) -> t.JsonMapping | None:
            """Structural member projected from the owning model.

            Duplication-gate scope overrides table read by the flext-infra duplication
            gate from the managed ``[tool.flext.project.duplication]`` section. The
            ingress owns the validated shape so the generator's own output always round-
            trips through this contract.
            """
            ...

    @runtime_checkable
    class ProjectToolFlextReadmeSection(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def id(self) -> str:
            """Stable project README section identifier."""
            ...

        @property
        def title(self) -> str:
            """Project README section heading."""
            ...

        @property
        def content(self) -> str | None:
            """Inline Markdown content."""
            ...

        @property
        def include(self) -> PurePosixPath | None:
            """Portable repository-relative Markdown include path."""
            ...

    @runtime_checkable
    class ProjectToolFlextDocs(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def documentation_language(self) -> t.DocumentationLanguage:
            """Generated documentation language."""
            ...

        @property
        def package_name(self) -> str | None:
            """Explicit import package override."""
            ...

        @property
        def project_class(self) -> str:
            """Documentation project class."""
            ...

        @property
        def site_title(self) -> str | None:
            """Documentation site title override."""
            ...

        @property
        def readme_sections(
            self,
        ) -> tuple[
            FlextProtocolsProjectMetadataFields.ProjectToolFlextReadmeSection, ...
        ]:
            """Ordered project README sections."""
            ...

    @runtime_checkable
    class ProjectToolFlextWorkspace(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def attached(self) -> bool:
            """Attach project to its parent workspace."""
            ...

    @runtime_checkable
    class ProjectToolFlextNamespace(
        FlextProtocolsProjectMetadataContract.FlextModelsProjectMetadataContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def enabled(self) -> bool | None:
            """Explicit namespace-enforcement toggle."""
            ...

        @property
        def scan_dirs(self) -> t.StrTuple:
            """Explicit namespace scan directories."""
            ...

        @property
        def include_dynamic_dirs(self) -> bool | None:
            """Whether non-canonical tracked dirs join the scan."""
            ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataFields",
]
