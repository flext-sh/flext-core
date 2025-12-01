"""Dispatcher-friendly utility exports split from ``FlextUtilities``.

The module re-exports utility helpers that were formerly nested to keep
imports lightweight while preserving the dispatcher-safe defaults used by
handlers, services, and registry code across the package.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Import modules that don't create circular dependencies
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration, T_Model
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.type_checker import FlextUtilitiesTypeChecker
from flext_core._utilities.type_guards import FlextUtilitiesTypeGuards

# Modules that depend on result.py are imported via __getattr__ below to avoid cycles

__all__ = [
    "FlextUtilitiesArgs",
    "FlextUtilitiesCache",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesDataMapper",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnum",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesModel",
    "FlextUtilitiesPagination",
    "FlextUtilitiesReliability",
    "FlextUtilitiesStringParser",
    "FlextUtilitiesTextProcessor",
    "FlextUtilitiesTypeChecker",
    "FlextUtilitiesTypeGuards",
    "FlextUtilitiesValidation",
    "T_Model",
]


def __getattr__(name: str) -> object:
    """Lazy import for modules that depend on result.py to avoid circular imports."""
    lazy_imports: dict[str, str] = {
        "FlextUtilitiesDataMapper": "flext_core._utilities.data_mapper",
        "FlextUtilitiesEnum": "flext_core._utilities.enum",
        "FlextUtilitiesModel": "flext_core._utilities.model",
        "FlextUtilitiesPagination": "flext_core._utilities.pagination",
        "FlextUtilitiesReliability": "flext_core._utilities.reliability",
        "FlextUtilitiesStringParser": "flext_core._utilities.string_parser",
        "FlextUtilitiesTextProcessor": "flext_core._utilities.text_processor",
        "FlextUtilitiesValidation": "flext_core._utilities.validation",
    }
    if name in lazy_imports:
        module_path = lazy_imports[name]
        module = __import__(module_path, fromlist=[name], level=0)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
