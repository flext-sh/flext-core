"""Pydantic v2 constants and exceptions exported via FlextConstants.

Including: ValidationError, sentinels, deprecations. Model configuration types
live on the models facade (``m.ConfigDict``, ``m.SettingsConfigDict``).

Architecture: Abstraction boundary - constants layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    VERSION,
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


class FlextConstantsPydantic:
    """Exceptions and constants: ValidationError, warnings, sentinels.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use c.* instead.
    """

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
    PYDANTIC_UNDEFINED = PydanticUndefined
    PydanticUndefinedType = PydanticUndefinedType
    PydanticUseDefault = PydanticUseDefault
    PydanticOmit = PydanticOmit

    # Version
    VERSION = VERSION
