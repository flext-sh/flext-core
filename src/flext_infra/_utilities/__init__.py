"""Infrastructure utility modules for flext-infra.

Organizes helper functions into domain-specific namespaces, following the
same pattern as flext_core._utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_infra._utilities.exceptions import FlextInfraUtilitiesExceptions
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

__all__ = [
    "FlextInfraUtilitiesExceptions",
    "FlextInfraUtilitiesIo",
    "FlextInfraUtilitiesOutput",
    "FlextInfraUtilitiesPaths",
    "FlextInfraUtilitiesPatterns",
    "FlextInfraUtilitiesProtocols",
    "FlextInfraUtilitiesSubprocess",
    "FlextInfraUtilitiesTemplates",
    "FlextInfraUtilitiesTerminal",
    "FlextInfraUtilitiesToml",
    "FlextInfraUtilitiesTomlParse",
    "FlextInfraUtilitiesYaml",
]


def __getattr__(name: str):
    """Lazy load utility modules on demand."""
    _LAZY_IMPORTS = {
        "FlextInfraUtilitiesExceptions": (
            "flext_infra._utilities.exceptions",
            "FlextInfraUtilitiesExceptions",
        ),
        "FlextInfraUtilitiesIo": ("flext_infra._utilities.io", "FlextInfraUtilitiesIo"),
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
        "FlextInfraUtilitiesProtocols": (
            "flext_infra._utilities.protocols",
            "FlextInfraUtilitiesProtocols",
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
        "FlextInfraUtilitiesYaml": (
            "flext_infra._utilities.yaml",
            "FlextInfraUtilitiesYaml",
        ),
    }

    if name in _LAZY_IMPORTS:
        module_name, class_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
