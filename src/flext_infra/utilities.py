"""Utilities facade for flext-infra.

Re-exports flext_core utilities and adds infrastructure-specific
utility namespaces for terminal, I/O, YAML, codegen, and refactor helpers.

All utility methods are exposed directly via u.Infra.[method](...)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities
from flext_infra._utilities.discover import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_infra._utilities.io import FlextInfraUtilitiesIo
from flext_infra._utilities.output import FlextInfraUtilitiesOutput
from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
from flext_infra._utilities.protocols import FlextInfraUtilitiesProtocols
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_infra._utilities.toml import FlextInfraUtilitiesToml
from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml
from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.refactor._utilities import FlextInfraUtilitiesRefactor


class FlextInfraUtilities(FlextUtilities):
    """Utility namespace for flext-infra; extends FlextUtilities.

    Usage::

        from flext_infra import u

        # Core utilities (inherited)
        u.generate()
        u.parse(value, int)

        # Infra-specific utilities - all methods exposed directly on u.Infra
        u.Infra.run_checked(["git", "status"])
        u.Infra.read_json(path)
        u.Infra.safe_load_yaml(path)
        u.Infra.infer_package(path)
        u.Infra.dotted_name(expr)
        u.Infra.as_toml_mapping(value)
        u.Infra.workspace_root()
    """

    class Infra(
        FlextInfraUtilitiesCodegen,
        FlextInfraUtilitiesDiscovery,
        FlextInfraUtilitiesGit,
        FlextInfraUtilitiesIo,
        FlextInfraUtilitiesOutput,
        FlextInfraUtilitiesPaths,
        FlextInfraUtilitiesPatterns,
        FlextInfraUtilitiesProtocols,
        FlextInfraUtilitiesRefactor,
        FlextInfraUtilitiesSubprocess,
        FlextInfraUtilitiesTemplates,
        FlextInfraUtilitiesTerminal,
        FlextInfraUtilitiesToml,
        FlextInfraUtilitiesTomlParse,
        FlextInfraUtilitiesYaml,
    ):
        """Infrastructure-domain utilities - all methods exposed directly."""


u = FlextInfraUtilities
__all__ = ["FlextInfraUtilities", "u"]
