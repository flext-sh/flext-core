# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/fields
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsProjectMetadataFields:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ProjectAuthor(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str: ...
        @property
        def email(self) -> str: ...

    @runtime_checkable
    class ProjectUrls(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def homepage(self) -> str: ...
        @property
        def documentation(self) -> str: ...
        @property
        def repository(self) -> str: ...

    @runtime_checkable
    class ProjectToolFlextProject(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def class_stem_override(self) -> str | None: ...
        @property
        def budget(self) -> t.JsonMapping | None: ...
        @property
        def duplication(self) -> t.JsonMapping | None: ...

    @runtime_checkable
    class ProjectToolFlextReadmeSection(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def id(self) -> str: ...
        @property
        def title(self) -> str: ...
        @property
        def content(self) -> str | None: ...
        @property
        def include(self) -> PurePosixPath | None: ...

    @runtime_checkable
    class ProjectToolFlextDocs(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def documentation_language(self) -> t.DocumentationLanguage: ...
        @property
        def package_name(self) -> str | None: ...
        @property
        def project_class(self) -> str: ...
        @property
        def site_title(self) -> str | None: ...
        @property
        def readme_sections(self) -> tuple[p.ProjectToolFlextReadmeSection, ...]: ...

    @runtime_checkable
    class ProjectToolFlextWorkspace(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def attached(self) -> bool: ...

    @runtime_checkable
    class ProjectToolFlextNamespace(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def enabled(self) -> bool | None: ...
        @property
        def scan_dirs(self) -> t.StrTuple: ...
        @property
        def include_dynamic_dirs(self) -> bool | None: ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataFields",
]
