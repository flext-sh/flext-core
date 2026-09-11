"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import m
from flext_cli import m as cli_m
from flext_infra._models.base import FlextInfraModelsBase
from flext_infra._models.census import FlextInfraModelsCensus
from flext_infra._models.check import FlextInfraModelsCheck
from flext_infra._models.codegen import FlextInfraModelsCodegen
from flext_infra._models.config import FlextInfraConfigModels
from flext_infra._models.deps import FlextInfraModelsDeps
from flext_infra._models.docs import FlextInfraModelsDocs
from flext_infra._models.enforcement import FlextInfraModelsEnforcement
from flext_infra._models.gates import FlextInfraModelsGates
from flext_infra._models.git import FlextInfraModelsGit
from flext_infra._models.github import FlextInfraModelsGithub
from flext_infra._models.layout import FlextInfraModelsLayout
from flext_infra._models.mixins import FlextInfraModelsMixins
from flext_infra._models.refactor import FlextInfraModelsRefactor
from flext_infra._models.release import FlextInfraModelsRelease
from flext_infra._models.rope import FlextInfraModelsRope
from flext_infra._models.scan import FlextInfraModelsScan
from flext_infra._models.transformers import FlextInfraModelsTransformers
from flext_infra._models.validate import FlextInfraModelsCore
from flext_infra._models.workspace import FlextInfraModelsWorkspace
from flext_infra._models.worktree import FlextInfraModelsWorktree


class FlextInfraModels(m):
    """Merged model namespace for flext-infra domain objects."""

    # NOTE (multi-agent): keep CLI route contracts available as FlextInfraModels.Cli
    # for legacy facade usage from CLI service route declarations.
    Cli = cli_m.Cli

    class Infra(
        FlextInfraModelsCensus,
        FlextInfraModelsCheck,
        # NOTE (multi-agent, flext-wkii.17 / agent: codex): conform contracts are
        # isolated from the active detector work in _models/codegen.py while
        # remaining exposed through the single public m.Infra facade.
        FlextInfraConfigModels,
        FlextInfraModelsCodegen,
        FlextInfraModelsDeps,
        FlextInfraModelsDocs,
        # NOTE (multi-agent): enforcement/transformers model
        # facades added for the deep-FLEXT dataclass -> m.Infra migration.
        FlextInfraModelsEnforcement,
        FlextInfraModelsGates,
        FlextInfraModelsGithub,
        FlextInfraModelsLayout,
        FlextInfraModelsRefactor,
        FlextInfraModelsRelease,
        FlextInfraModelsMixins,
        FlextInfraModelsTransformers,
        FlextInfraModelsWorkspace,
        # flext-wkii.17.26 (codex): all fix/codegen mutations share one typed
        # worktree transaction report rather than command-local backup shapes.
        FlextInfraModelsWorktree,
        FlextInfraModelsGit,
        FlextInfraModelsRope,
        FlextInfraModelsScan,
        FlextInfraModelsCore,
        FlextInfraModelsBase,
    ):
        """Infrastructure-domain models - all classes exposed directly."""


m = FlextInfraModels

__all__: list[str] = ["FlextInfraModels", "m"]
