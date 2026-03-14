"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextModels
from flext_infra._utilities._models import FlextInfraUtilitiesModels
from flext_infra.basemk._models import FlextInfraBasemkModels
from flext_infra.check._models import FlextInfraCheckModels
from flext_infra.codegen._models import FlextInfraCodegenModels
from flext_infra.core._models import FlextInfraCoreModels
from flext_infra.deps._models import FlextInfraDepsModels
from flext_infra.docs._models import FlextInfraDocsModels
from flext_infra.github._models import FlextInfraGithubModels
from flext_infra.refactor._models import FlextInfraRefactorModels
from flext_infra.release._models import FlextInfraReleaseModels
from flext_infra.workspace._models import FlextInfraWorkspaceModels


class FlextInfraModels(FlextModels):
    """Pydantic model namespace for infrastructure services."""

    class Infra:
        """Infrastructure-domain models."""

        # -- Workspace domain (via MRO) -------------------------------------------

        class Workspace(FlextInfraWorkspaceModels):
            """Workspace domain models via MRO."""

        # -- Core domain (via MRO) ------------------------------------------------

        class Core(FlextInfraCoreModels):
            """Core domain models via MRO."""

        # -- Basemk domain (via MRO) ----------------------------------------------

        class Basemk(FlextInfraBasemkModels):
            """Basemk domain models via MRO."""

        # -- Release domain (via MRO) ---------------------------------------------

        class Release(FlextInfraReleaseModels):
            """Release domain models via MRO."""

        # -- GitHub domain (via MRO) ----------------------------------------------

        class Github(FlextInfraGithubModels):
            """GitHub domain models via MRO."""

        # -- Codegen domain (via MRO) ---------------------------------------------

        class Codegen(FlextInfraCodegenModels):
            """Codegen domain models via MRO."""

        # -- Deps domain (via MRO) ------------------------------------------------

        class Deps(FlextInfraDepsModels):
            """Deps domain models via MRO."""

        # -- Docs domain (via MRO) ------------------------------------------------

        class Docs(FlextInfraDocsModels):
            """Docs domain models via MRO."""

        # -- Check domain (via MRO) -----------------------------------------------

        class Check(FlextInfraCheckModels):
            """Check domain models via MRO."""

        # -- Refactor domain (via MRO) --------------------------------------------

        class Refactor(FlextInfraRefactorModels):
            """Refactor domain models via MRO."""

        # -- Shared utilities domain (via MRO) ------------------------------------

        class Utilities(FlextInfraUtilitiesModels):
            """Shared utilities domain models via MRO."""


m = FlextInfraModels

__all__ = ["FlextInfraModels", "m"]
