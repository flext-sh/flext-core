"""Dispatcher-friendly utility exports split from ``FlextUtilities``.

The module re-exports utility helpers that were formerly nested to keep
imports lightweight while preserving the dispatcher-safe defaults used by
handlers, services, and registry code across the package.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Import all extracted classes
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.configuration import FlextUtilitiesConfiguration, T_Model
from flext_core._utilities.data_mapper import FlextUtilitiesDataMapper
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.string_parser import FlextUtilitiesStringParser
from flext_core._utilities.text_processor import FlextUtilitiesTextProcessor
from flext_core._utilities.type_checker import FlextUtilitiesTypeChecker
from flext_core._utilities.type_guards import FlextUtilitiesTypeGuards
from flext_core._utilities.validation import FlextUtilitiesValidation

__all__ = [
    "FlextUtilitiesCache",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesDataMapper",
    "FlextUtilitiesDomain",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesPagination",
    "FlextUtilitiesReliability",
    "FlextUtilitiesStringParser",
    "FlextUtilitiesTextProcessor",
    "FlextUtilitiesTypeChecker",
    "FlextUtilitiesTypeGuards",
    "FlextUtilitiesValidation",
    "T_Model",
]
