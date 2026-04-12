"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    create_model,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
    validate_call,
    with_config,
)
from pydantic_core import (
    from_json,
    to_json,
    to_jsonable_python,
)


class FlextUtilitiesPydantic:
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* instead.
    """

    # Field definition and private attributes
    Field = Field
    PrivateAttr = PrivateAttr

    # Field and model decorators
    computed_field = computed_field
    field_validator = field_validator
    field_serializer = field_serializer
    model_validator = model_validator
    model_serializer = model_serializer
    # root_validator = root_validator
    # validator = validator

    # Type adapters and model creation
    create_model = create_model
    validate_call = validate_call
    # parse_obj_as = parse_obj_as
    with_config = with_config

    # Schema and JSON utilities (from pydantic_core)
    from_json = from_json
    to_json = to_json
    to_jsonable_python = to_jsonable_python
