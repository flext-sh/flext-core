"""Pydantic v2 project-metadata SSOT models (flat on ``m.*``).

Tier 3 domain models owned by flext-core. Consumed across the monorepo
as ``m.ProjectMetadata`` / ``m.ProjectNamespaceConfig`` /
``m.ProjectToolFlext`` / ``m.ProjectToolFlextProject`` / etc. Immutable
by construction (``frozen=True``, ``extra="forbid"``) so callers cannot
drift metadata after load.

Architecture: Tier 3 — imports Tier 0 (_constants/pydantic) only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
    FlextModelsProjectMetadata as FlextModelsProjectMetadataPartFinal,
)


class FlextModelsProjectMetadata(FlextModelsProjectMetadataPartFinal):
    """Public facade for FlextModelsProjectMetadata."""


__all__: list[str] = ["FlextModelsProjectMetadata"]
