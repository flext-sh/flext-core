"""Pydantic v2 constants, exceptions and configuration exported via FlextConstants.

Including: ConfigDict, SettingsConfigDict, ValidationError, sentinels, deprecations.

Architecture: Abstraction boundary - constants layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic_core import (
    MISSING,
    PydanticCustomError,
    PydanticKnownError,
    PydanticOmit,
    PydanticSerializationError,
    PydanticSerializationUnexpectedValue,
    PydanticUndefined,
    PydanticUndefinedType,
    PydanticUseDefault,
    SchemaError,
    ValidationError as CoreValidationError,
)

from pydantic import (
    VERSION,
    ConfigDict,
    PydanticDeprecatedSince20,
    PydanticDeprecatedSince26,
    PydanticDeprecatedSince29,
    PydanticDeprecatedSince210,
    PydanticDeprecatedSince211,
    PydanticDeprecatedSince212,
    PydanticDeprecationWarning,
    PydanticErrorCodes,
    PydanticExperimentalWarning,
    PydanticForbiddenQualifier,
    PydanticImportError,
    PydanticInvalidForJsonSchema,
    PydanticSchemaGenerationError,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    ValidationError,
)


class FlextConstantsPydantic:
    """Configuration, exceptions and constants: ConfigDict, ValidationError, sentinels.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use c.* instead.
    """

    # Configuration type
    ConfigDict = ConfigDict

    # Exceptions (pydantic v2)
    ValidationError = ValidationError
    PydanticImportError = PydanticImportError
    PydanticSchemaGenerationError = PydanticSchemaGenerationError
    PydanticUserError = PydanticUserError
    PydanticInvalidForJsonSchema = PydanticInvalidForJsonSchema
    PydanticUndefinedAnnotation = PydanticUndefinedAnnotation
    PydanticForbiddenQualifier = PydanticForbiddenQualifier

    # Warnings
    PydanticDeprecationWarning = PydanticDeprecationWarning
    PydanticExperimentalWarning = PydanticExperimentalWarning

    # Deprecation markers
    PydanticDeprecatedSince20 = PydanticDeprecatedSince20
    PydanticDeprecatedSince26 = PydanticDeprecatedSince26
    PydanticDeprecatedSince29 = PydanticDeprecatedSince29
    PydanticDeprecatedSince210 = PydanticDeprecatedSince210
    PydanticDeprecatedSince211 = PydanticDeprecatedSince211
    PydanticDeprecatedSince212 = PydanticDeprecatedSince212

    # Error information
    PydanticErrorCodes = PydanticErrorCodes

    # pydantic_core exceptions
    SchemaError = SchemaError
    PydanticCustomError = PydanticCustomError
    PydanticKnownError = PydanticKnownError
    PydanticSerializationError = PydanticSerializationError
    PydanticSerializationUnexpectedValue = PydanticSerializationUnexpectedValue
    CoreValidationError = CoreValidationError

    # pydantic_core sentinels and special values
    MISSING = MISSING
    PydanticUndefined = PydanticUndefined
    PydanticUndefinedType = PydanticUndefinedType
    PydanticUseDefault = PydanticUseDefault
    PydanticOmit = PydanticOmit

    # Version
    VERSION = VERSION
