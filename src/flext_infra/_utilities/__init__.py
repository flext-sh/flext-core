# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Infrastructure utility modules for flext-infra.

Organizes helper functions into domain-specific namespaces, following the
same pattern as flext_core._utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra._utilities.exceptions import (
        FlextInfraExceptions,
        FlextInfraUtilitiesExceptions,
        e,
    )
    from flext_infra._utilities.io import FlextInfraUtilitiesIo
    from flext_infra._utilities.output import (
        BLUE,
        BOLD,
        GREEN,
        RED,
        RESET,
        SYM_ARROW,
        SYM_BULLET,
        SYM_FAIL,
        SYM_OK,
        SYM_SKIP,
        SYM_WARN,
        YELLOW,
        FlextInfraUtilitiesOutput,
        output,
    )
    from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
    from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
    from flext_infra._utilities.protocols import FlextInfraUtilitiesProtocols, p
    from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
    from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
    from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
    from flext_infra._utilities.toml import FlextInfraUtilitiesToml
    from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
    from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BLUE": ("flext_infra._utilities.output", "BLUE"),
    "BOLD": ("flext_infra._utilities.output", "BOLD"),
    "FlextInfraExceptions": ("flext_infra._utilities.exceptions", "FlextInfraExceptions"),
    "FlextInfraUtilitiesExceptions": ("flext_infra._utilities.exceptions", "FlextInfraUtilitiesExceptions"),
    "FlextInfraUtilitiesIo": ("flext_infra._utilities.io", "FlextInfraUtilitiesIo"),
    "FlextInfraUtilitiesOutput": ("flext_infra._utilities.output", "FlextInfraUtilitiesOutput"),
    "FlextInfraUtilitiesPaths": ("flext_infra._utilities.paths", "FlextInfraUtilitiesPaths"),
    "FlextInfraUtilitiesPatterns": ("flext_infra._utilities.patterns", "FlextInfraUtilitiesPatterns"),
    "FlextInfraUtilitiesProtocols": ("flext_infra._utilities.protocols", "FlextInfraUtilitiesProtocols"),
    "FlextInfraUtilitiesSubprocess": ("flext_infra._utilities.subprocess", "FlextInfraUtilitiesSubprocess"),
    "FlextInfraUtilitiesTemplates": ("flext_infra._utilities.templates", "FlextInfraUtilitiesTemplates"),
    "FlextInfraUtilitiesTerminal": ("flext_infra._utilities.terminal", "FlextInfraUtilitiesTerminal"),
    "FlextInfraUtilitiesToml": ("flext_infra._utilities.toml", "FlextInfraUtilitiesToml"),
    "FlextInfraUtilitiesTomlParse": ("flext_infra._utilities.toml_parse", "FlextInfraUtilitiesTomlParse"),
    "FlextInfraUtilitiesYaml": ("flext_infra._utilities.yaml", "FlextInfraUtilitiesYaml"),
    "GREEN": ("flext_infra._utilities.output", "GREEN"),
    "RED": ("flext_infra._utilities.output", "RED"),
    "RESET": ("flext_infra._utilities.output", "RESET"),
    "SYM_ARROW": ("flext_infra._utilities.output", "SYM_ARROW"),
    "SYM_BULLET": ("flext_infra._utilities.output", "SYM_BULLET"),
    "SYM_FAIL": ("flext_infra._utilities.output", "SYM_FAIL"),
    "SYM_OK": ("flext_infra._utilities.output", "SYM_OK"),
    "SYM_SKIP": ("flext_infra._utilities.output", "SYM_SKIP"),
    "SYM_WARN": ("flext_infra._utilities.output", "SYM_WARN"),
    "YELLOW": ("flext_infra._utilities.output", "YELLOW"),
    "e": ("flext_infra._utilities.exceptions", "e"),
    "output": ("flext_infra._utilities.output", "output"),
    "p": ("flext_infra._utilities.protocols", "p"),
}

__all__ = [
    "BLUE",
    "BOLD",
    "GREEN",
    "RED",
    "RESET",
    "SYM_ARROW",
    "SYM_BULLET",
    "SYM_FAIL",
    "SYM_OK",
    "SYM_SKIP",
    "SYM_WARN",
    "YELLOW",
    "FlextInfraExceptions",
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
    "e",
    "output",
    "p",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
