"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import (
    AfterValidator,
    Field,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    SkipValidation,
    WrapSerializer,
    WrapValidator,
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

from flext_core._models.pydantic import FlextModelsPydantic as mp


class FlextUtilitiesPydantic:
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    Field = Field
    PrivateAttr = PrivateAttr
    SkipValidation = SkipValidation

    computed_field = computed_field
    field_validator = field_validator
    field_serializer = field_serializer
    model_validator = model_validator
    model_serializer = model_serializer

    AfterValidator = AfterValidator
    BeforeValidator = mp.BeforeValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    ConfigDict = mp.ConfigDict
    FieldSerializationInfo = mp.FieldSerializationInfo
    TypeAdapter = mp.TypeAdapter
    create_model = create_model
    validate_call = validate_call
    with_config = with_config

    from_json = from_json
    to_json = to_json
    to_jsonable_python = to_jsonable_python
