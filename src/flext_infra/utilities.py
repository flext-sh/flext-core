"""Utilities facade for flext-infra.

Re-exports flext_core utilities and adds infrastructure-specific
utility namespaces. All methods exposed directly via u.Infra.[method](...)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities
from flext_infra._utilities.cli import FlextInfraUtilitiesCli
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.formatting import FlextInfraUtilitiesFormatting
from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_infra._utilities.io import FlextInfraUtilitiesIo
from flext_infra._utilities.iteration import FlextInfraUtilitiesIteration
from flext_infra._utilities.output import FlextInfraUtilitiesOutput
from flext_infra._utilities.parsing import FlextInfraUtilitiesParsing
from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
from flext_infra._utilities.reporting import FlextInfraUtilitiesReporting
from flext_infra._utilities.safety import FlextInfraUtilitiesSafety
from flext_infra._utilities.scanning import FlextInfraUtilitiesScanning
from flext_infra._utilities.selection import FlextInfraUtilitiesSelection
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_infra._utilities.toml import FlextInfraUtilitiesToml
from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
from flext_infra._utilities.versioning import FlextInfraUtilitiesVersioning
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml
from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.refactor._utilities import FlextInfraUtilitiesRefactor


class FlextInfraUtilities(FlextUtilities):
    """Utility namespace for flext-infra; extends FlextUtilities.

    Usage::

        from flext_infra import u

        u.Infra.git_run_checked(["status"])
        u.Infra.read_json(path)
        u.Infra.discover_projects(workspace_root)
        u.Infra.parse_semver("1.2.3")
    """

    class Infra(
        FlextInfraUtilitiesCli,
        FlextInfraUtilitiesCodegen,
        FlextInfraUtilitiesDiscovery,
        FlextInfraUtilitiesFormatting,
        FlextInfraUtilitiesGit,
        FlextInfraUtilitiesIo,
        FlextInfraUtilitiesIteration,
        FlextInfraUtilitiesOutput,
        FlextInfraUtilitiesParsing,
        FlextInfraUtilitiesPaths,
        FlextInfraUtilitiesPatterns,
        FlextInfraUtilitiesRefactor,
        FlextInfraUtilitiesReporting,
        FlextInfraUtilitiesSafety,
        FlextInfraUtilitiesScanning,
        FlextInfraUtilitiesSelection,
        FlextInfraUtilitiesSubprocess,
        FlextInfraUtilitiesTemplates,
        FlextInfraUtilitiesTerminal,
        FlextInfraUtilitiesToml,
        FlextInfraUtilitiesTomlParse,
        FlextInfraUtilitiesVersioning,
        FlextInfraUtilitiesYaml,
    ):
        """Infrastructure-domain utilities - all methods exposed directly."""


u = FlextInfraUtilities
__all__ = ["FlextInfraUtilities", "u"]
