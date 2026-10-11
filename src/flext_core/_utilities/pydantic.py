"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import (
    AfterValidator,
    PlainSerializer,
    PlainValidator,
    SkipValidation,
    WrapSerializer,
    WrapValidator,
    model_serializer,
    validate_call,
    with_config,
)
from pydantic_core import from_json, to_json, to_jsonable_python

from flext_core._models import FlextModelsPydantic
from flext_core._utilities.typeadapter_json import FlextUtilitiesTypingAdapterJson
from flext_core._utilities.typeadapter_scalars import FlextUtilitiesTypingAdapterScalars


class FlextUtilitiesPydantic(
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
):
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    # Wrap field specifiers in ``staticmethod`` so type checkers treat them as
    # static callables rather than instance-bound descriptors.
    # Why: a bare ``Field = FlextModelsPydantic.Field`` binds ``_field``'s
    # generic first parameter
    # (``default: DefaultT``) to ``self`` = ``FlextUtilities`` when accessed via the
    # facade, so ``u.Field(default_factory=...)`` was mis-inferred as returning
    # ``FlextUtilities`` (bogus reportAssignmentType). The model owner already
    # does this.
    Field = staticmethod(FlextModelsPydantic.Field)
    PrivateAttr = staticmethod(FlextModelsPydantic.PrivateAttr)
    # Canonical empty-collection default:
    # ``u.Field(default_factory=u.empty(TypeForm(<field annotation>)))``.
    empty = staticmethod(FlextModelsPydantic.empty)
    SkipValidation = SkipValidation

    # Keep the models-layer overload set intact; staticmethod's generic
    # constructor infers only its first arm. Runtime still uses the same factory.
    if TYPE_CHECKING:
        field_validator = FlextModelsPydantic.field_validator
        field_serializer = FlextModelsPydantic.field_serializer
        model_validator = FlextModelsPydantic.model_validator
    else:
        field_validator = staticmethod(FlextModelsPydantic.field_validator)
        field_serializer = staticmethod(FlextModelsPydantic.field_serializer)
        model_validator = staticmethod(FlextModelsPydantic.model_validator)

    computed_field = staticmethod(FlextModelsPydantic.computed_field)
    model_serializer = staticmethod(model_serializer)

    AfterValidator = AfterValidator
    BeforeValidator = FlextModelsPydantic.BeforeValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    if TYPE_CHECKING:
        validate_call = validate_call
    else:
        validate_call = staticmethod(validate_call)
    with_config = staticmethod(with_config)

    from_json = staticmethod(from_json)
    to_json = to_json
    to_jsonable_python = to_jsonable_python
