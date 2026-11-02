# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/document
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core import p


class FlextProtocolsProjectMetadataDocument:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class PyprojectTool(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def flext(self) -> p.ProjectToolFlext: ...

    @runtime_checkable
    class ProjectMetadata(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def root(self) -> Path: ...
        @property
        def package_name(self) -> str: ...
        @property
        def class_stem(self) -> str: ...
        @property
        def project(self) -> p.Project: ...
        @property
        def flext(self) -> p.ProjectToolFlext: ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataDocument",
]
