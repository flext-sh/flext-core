"""FlextProtocolsProjectMetadata — Tier 1 protocols for metadata services.

Structural contracts for readers/derivers of project metadata. These
protocols let Tier 4 utilities, Tier 3 models, and external consumers
(flext-infra generators, flext-tests fixtures) type against a
structural surface without depending on concrete classes.

Architecture: Tier 1 — may import Tier 0 (_typings, _constants) only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


class FlextProtocolsProjectMetadata:
    """Namespace holder for project-metadata protocols SSOT.

    Protocols live under nested ``Project`` class so that ``FlextProtocols``
    can inherit this class via MRO and expose ``p.Project.MetadataReader``
    (sub-namespace access).
    """

    _flext_enforcement_exempt: ClassVar[bool] = True

    class Project:
        """Project-metadata protocols — SSOT (accessible as ``p.Project.*``)."""

        _flext_enforcement_exempt: ClassVar[bool] = True

        @runtime_checkable
        class MetadataReader(Protocol):
            """Reads project metadata from a project root directory.

            Return type is object to avoid Tier 1 → Tier 3 dependency; callers
            narrow to the concrete ``m.Project.Project`` model at use site.
            """

            def read(self, root: Path) -> object: ...

        @runtime_checkable
        class ClassStemDeriver(Protocol):
            """Derives a PascalCase class stem from a kebab-case project name."""

            def derive(self, project_name: str) -> str: ...

        @runtime_checkable
        class TierFacadeNamer(Protocol):
            """Builds tier-specific facade class names (src/tests/examples/...)."""

            def name_for(self, project_name: str, tier: str) -> str: ...
