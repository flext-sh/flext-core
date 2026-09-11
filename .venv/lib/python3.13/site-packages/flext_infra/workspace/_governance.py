"""Repository-local persistent-state policy composed into topology detection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import c, config
from flext_infra import m


class FlextInfraWorkspaceGovernanceMixin:
    """Project persistent-state ownership from repository-local policy."""

    @staticmethod
    def persistent_state_artifacts(
        make_profile: c.Infra.MakeProfile,
    ) -> tuple[m.Infra.CodegenArtifactSpec, ...]:
        """Project persistent-state artifacts owned by every governed repository."""
        del make_profile
        persistent = c.Infra.PERSISTENT_STATE_ARTIFACT_NAMES
        return tuple(
            artifact
            for artifact in config.Infra.codegen.artifacts
            if artifact.name in persistent
        )


__all__: list[str] = ["FlextInfraWorkspaceGovernanceMixin"]
