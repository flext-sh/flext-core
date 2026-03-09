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
    from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
    from flext_infra._utilities.git import FlextInfraUtilitiesGit
    from flext_infra._utilities.io import FlextInfraUtilitiesIo
    from flext_infra._utilities.output import (
        FlextInfraUtilitiesOutput,
        _OutputBackend,
        output,
    )
    from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
    from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
    from flext_infra._utilities.reporting import FlextInfraUtilitiesReporting
    from flext_infra._utilities.selection import FlextInfraUtilitiesSelection
    from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
    from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
    from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
    from flext_infra._utilities.toml import (
        FlextInfraUtilitiesToml,
        array,
        as_container_list,
        as_string_list,
        as_toml_mapping,
        ensure_table,
        normalize_container_value,
        read_doc,
        table_string_keys,
        toml_get,
        unwrap_item,
    )
    from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
    from flext_infra._utilities.versioning import FlextInfraUtilitiesVersioning
    from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraUtilitiesDiscovery": (
        "flext_infra._utilities.discovery",
        "FlextInfraUtilitiesDiscovery",
    ),
    "FlextInfraUtilitiesGit": ("flext_infra._utilities.git", "FlextInfraUtilitiesGit"),
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
    "_OutputBackend": ("flext_infra._utilities.output", "_OutputBackend"),
    "array": ("flext_infra._utilities.toml", "array"),
    "as_container_list": ("flext_infra._utilities.toml", "as_container_list"),
    "as_string_list": ("flext_infra._utilities.toml", "as_string_list"),
    "as_toml_mapping": ("flext_infra._utilities.toml", "as_toml_mapping"),
    "ensure_table": ("flext_infra._utilities.toml", "ensure_table"),
    "normalize_container_value": (
        "flext_infra._utilities.toml",
        "normalize_container_value",
    ),
    "output": ("flext_infra._utilities.output", "output"),
    "read_doc": ("flext_infra._utilities.toml", "read_doc"),
    "table_string_keys": ("flext_infra._utilities.toml", "table_string_keys"),
    "toml_get": ("flext_infra._utilities.toml", "toml_get"),
    "unwrap_item": ("flext_infra._utilities.toml", "unwrap_item"),
}

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
    "_OutputBackend",
    "array",
    "as_container_list",
    "as_string_list",
    "as_toml_mapping",
    "ensure_table",
    "normalize_container_value",
    "output",
    "read_doc",
    "table_string_keys",
    "toml_get",
    "unwrap_item",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
