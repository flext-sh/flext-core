"""Pydantic v2 helper utilities exported via FlextUtilities.

This module provides public aliases for pydantic v2 field/validation helpers
that are used across the flext ecosystem. All projects consuming these
must import from flext_core.u.* instead of directly from pydantic.

Architecture: Abstraction boundary - utilities layer
Boundary: flext-core is sole owner of pydantic v2 integration. All other
projects receive pydantic functionality ONLY through public facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    Field,
    PrivateAttr,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)


class FlextUtilitiesPydantic:
    """Public field/validation helpers from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use these aliases via u.* instead: u.Field, u.computed_field, etc.

    Available helpers (accessible as u.HELPER_NAME):
        Field: Pydantic field definition with validation constraints
        PrivateAttr: Decorator for private model attributes
        computed_field: Decorator for derived fields from model state
        field_validator: Decorator for field-level validation
        field_serializer: Decorator for field-level serialization
        model_validator: Decorator for model-wide validation
        TypeAdapter: Runtime type validation and coercion
    """

    # Public Pydantic v2 field/validation APIs available via u.*
    Field = Field
    PrivateAttr = PrivateAttr
    computed_field = computed_field
    field_validator = field_validator
    field_serializer = field_serializer
    model_validator = model_validator
    TypeAdapter = TypeAdapter
