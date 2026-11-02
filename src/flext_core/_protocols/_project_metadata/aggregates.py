# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
src/flext_core/_protocols/_project_metadata/aggregates
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import t

if TYPE_CHECKING:
    from flext_core.protocols import p


class FlextProtocolsProjectMetadataAggregates:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Project(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str: ...
        @property
        def version(self) -> str: ...
        @property
        def description(self) -> str: ...
        @property
        def authors(self) -> tuple[p.ProjectAuthor, ...]: ...
        @property
        def urls(self) -> p.ProjectUrls: ...
        @property
        def requires_python(self) -> str: ...
        @property
        def dependencies(self) -> t.StrTuple: ...
        @property
        def classifiers(self) -> t.StrTuple: ...
        @property
        def keywords(self) -> t.StrTuple: ...

    @runtime_checkable
    class ProjectToolFlext(Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def project(self) -> p.ProjectToolFlextProject: ...
        @property
        def docs(self) -> p.ProjectToolFlextDocs: ...
        @property
        def workspace(self) -> p.ProjectToolFlextWorkspace: ...
        @property
        def namespace(self) -> p.ProjectToolFlextNamespace: ...


__all__: list[str] = [
    "FlextProtocolsProjectMetadataAggregates",
]
