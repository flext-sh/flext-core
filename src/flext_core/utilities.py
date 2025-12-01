"""Utility façade for validation, parsing, and reliability helpers.

Expose ``FlextUtilities`` as a dispatcher-safe toolbox for caching, type
guards, string parsing, data mapping, and reliability patterns that all return
``FlextResult`` values. The underlying implementations live in ``_utilities``
but are surfaced here for stable imports across examples and services.

**CRITICAL ARCHITECTURE**: FlextUtilities is a THIN FACADE - pure delegation
to _utilities classes. No other module can import from _utilities directly.
All external code MUST use FlextUtilities as the single access point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from flext_core._utilities.data_mapper import FlextUtilitiesDataMapper
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enum import FlextUtilitiesEnum
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.string_parser import FlextUtilitiesStringParser
from flext_core._utilities.text_processor import FlextUtilitiesTextProcessor
from flext_core._utilities.type_checker import FlextUtilitiesTypeChecker
from flext_core._utilities.type_guards import FlextUtilitiesTypeGuards
from flext_core._utilities.validation import FlextUtilitiesValidation


class FlextUtilities:
    """Stable utility surface for dispatcher-friendly helpers.

    Provides enterprise-grade utility functions for common operations
    throughout the FLEXT ecosystem. This is a PURE FACADE that delegates
    to _utilities package implementations.

    Architecture: Tier 1.5 (Foundation Utilities)
    ==============================================
    - No nested class definitions (single class per module principle)
    - All attributes reference _utilities classes directly
    - External code uses FlextUtilities.XxxClass.method() pattern
    - No direct _utilities imports allowed outside this module

    Core Namespaces:
    - Enum: StrEnum utilities for type-safe enum handling
    - Collection: Collection conversion utilities
    - Args: Automatic args/kwargs parsing
    - Model: Pydantic model initialization utilities
    - Cache: Data normalization and cache key generation
    - Validation: Comprehensive input validation
    - Generators: ID, UUID, timestamp generation
    - TextProcessor: Text cleaning and processing
    - TypeGuards: Runtime type checking
    - Reliability: Timeout and retry patterns
    - TypeChecker: Runtime type introspection
    - Configuration: Parameter access/manipulation
    - DataMapper: Data mapping and transformation utilities
    - Domain: Domain-specific utilities
    - Pagination: API pagination utilities
    - StringParser: String parsing utilities

    Usage Pattern:
        from flext_core import FlextUtilities
        # No FlextUtilities.Enum class - direct reference instead
        result = FlextUtilities.Enum.parse(MyEnum, "value")
    """

    # ═══════════════════════════════════════════════════════════════════
    # CLASS-LEVEL ATTRIBUTES: Module References (NOT nested classes)
    # ═══════════════════════════════════════════════════════════════════
    # Each attribute points directly to _utilities class for pure delegation

    Enum = FlextUtilitiesEnum
    Collection = FlextUtilitiesCollection
    Args = FlextUtilitiesArgs
    Model = FlextUtilitiesModel
    Cache = FlextUtilitiesCache
    Validation = FlextUtilitiesValidation
    Generators = FlextUtilitiesGenerators
    TextProcessor = FlextUtilitiesTextProcessor
    TypeGuards = FlextUtilitiesTypeGuards
    Reliability = FlextUtilitiesReliability
    TypeChecker = FlextUtilitiesTypeChecker
    Configuration = FlextUtilitiesConfiguration
    DataMapper = FlextUtilitiesDataMapper
    Domain = FlextUtilitiesDomain
    Pagination = FlextUtilitiesPagination
    StringParser = FlextUtilitiesStringParser


__all__ = ["FlextUtilities"]
