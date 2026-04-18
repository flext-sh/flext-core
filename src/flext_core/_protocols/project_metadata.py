"""FlextProtocolsProjectMetadata — Tier 1 protocols for metadata services.

Structural contracts for readers/derivers of project metadata. Inherited
flat into ``FlextProtocols`` so consumers access them as
``p.ProjectMetadataReader`` / ``p.ProjectClassStemDeriver`` /
``p.ProjectTierFacadeNamer``.

Architecture: Tier 1 — may import Tier 0 (_typings, _constants) only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


class FlextProtocolsProjectMetadata:
    """Protocols for project-metadata services (flat on ``p.*``)."""

    _flext_enforcement_exempt: ClassVar[bool] = True

    @runtime_checkable
    class ProjectMetadataReader(Protocol):
        """Reads project metadata from a project root directory.

        Return type is ``object`` to avoid Tier 1 → Tier 3 dependency;
        callers narrow to the concrete ``m.ProjectMetadata`` model at use site.
        """

        def read(self, root: Path) -> object: ...

    @runtime_checkable
    class ProjectClassStemDeriver(Protocol):
        """Derives a PascalCase class stem from a kebab-case project name."""

        def derive(self, project_name: str) -> str: ...

    @runtime_checkable
    class ProjectTierFacadeNamer(Protocol):
        """Builds tier-specific facade class names (src/tests/examples/...)."""

        def name_for(self, project_name: str, tier: str) -> str: ...
