"""Infrastructure utility modules for flext-infra.

Organizes helper functions into domain-specific namespaces, following the
same pattern as flext_core._utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
    from flext_infra._utilities.git import FlextInfraUtilitiesGit
    from flext_infra._utilities.io import FlextInfraUtilitiesIo
    from flext_infra._utilities.output import FlextInfraUtilitiesOutput
    from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
    from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
    from flext_infra._utilities.reporting import FlextInfraUtilitiesReporting
    from flext_infra._utilities.selection import FlextInfraUtilitiesSelection
    from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
    from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
    from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
    from flext_infra._utilities.toml import FlextInfraUtilitiesToml
    from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
    from flext_infra._utilities.versioning import FlextInfraUtilitiesVersioning
    from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml

__all__ = [
    "FlextInfraUtilitiesDiscovery",
    "FlextInfraUtilitiesGit",
    "FlextInfraUtilitiesIo",
    "FlextInfraUtilitiesOutput",
    "FlextInfraUtilitiesPaths",
    "FlextInfraUtilitiesPatterns",
    "FlextInfraUtilitiesReporting",
    "FlextInfraUtilitiesSelection",
    "FlextInfraUtilitiesSubprocess",
    "FlextInfraUtilitiesTemplates",
    "FlextInfraUtilitiesTerminal",
    "FlextInfraUtilitiesToml",
    "FlextInfraUtilitiesTomlParse",
    "FlextInfraUtilitiesVersioning",
    "FlextInfraUtilitiesYaml",
]


def __getattr__(name: str) -> object:
    """Lazy load utility modules on demand."""
    lazy_imports = {
        "FlextInfraUtilitiesDiscovery": (
            "flext_infra._utilities.discovery",
            "FlextInfraUtilitiesDiscovery",
        ),
        "FlextInfraUtilitiesGit": (
            "flext_infra._utilities.git",
            "FlextInfraUtilitiesGit",
        ),
        "FlextInfraUtilitiesIo": (
            "flext_infra._utilities.io",
            "FlextInfraUtilitiesIo",
        ),
        "FlextInfraUtilitiesOutput": (
            "flext_infra._utilities.output",
            "FlextInfraUtilitiesOutput",
        ),
        "FlextInfraUtilitiesPaths": (
            "flext_infra._utilities.paths",
            "FlextInfraUtilitiesPaths",
        ),
        "FlextInfraUtilitiesPatterns": (
            "flext_infra._utilities.patterns",
            "FlextInfraUtilitiesPatterns",
        ),
        "FlextInfraUtilitiesReporting": (
            "flext_infra._utilities.reporting",
            "FlextInfraUtilitiesReporting",
        ),
        "FlextInfraUtilitiesSelection": (
            "flext_infra._utilities.selection",
            "FlextInfraUtilitiesSelection",
        ),
        "FlextInfraUtilitiesSubprocess": (
            "flext_infra._utilities.subprocess",
            "FlextInfraUtilitiesSubprocess",
        ),
        "FlextInfraUtilitiesTemplates": (
            "flext_infra._utilities.templates",
            "FlextInfraUtilitiesTemplates",
        ),
        "FlextInfraUtilitiesTerminal": (
            "flext_infra._utilities.terminal",
            "FlextInfraUtilitiesTerminal",
        ),
        "FlextInfraUtilitiesToml": (
            "flext_infra._utilities.toml",
            "FlextInfraUtilitiesToml",
        ),
        "FlextInfraUtilitiesTomlParse": (
            "flext_infra._utilities.toml_parse",
            "FlextInfraUtilitiesTomlParse",
        ),
        "FlextInfraUtilitiesVersioning": (
            "flext_infra._utilities.versioning",
            "FlextInfraUtilitiesVersioning",
        ),
        "FlextInfraUtilitiesYaml": (
            "flext_infra._utilities.yaml",
            "FlextInfraUtilitiesYaml",
        ),
    }

    if name in lazy_imports:
        module_name, class_name = lazy_imports[name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
