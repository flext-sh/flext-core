"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import p

from pydantic import (
    AfterValidator,
    PlainSerializer,
    PlainValidator,
    SkipValidation,
    TypeAdapter as PydanticTypeAdapter,
    WrapSerializer,
    WrapValidator,
    model_serializer,
    validate_call,
    with_config,
)
from pydantic_core import from_json, to_json, to_jsonable_python

from flext_core._models import FlextModelsPydantic as mp


class FlextUtilitiesPydantic:
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    # Wrap field specifiers in ``staticmethod`` so type checkers treat them as
    # static callables rather than instance-bound descriptors.
    # Why: a bare ``Field = mp.Field`` binds ``_field``'s generic first parameter
    # (``default: DefaultT``) to ``self`` = ``FlextUtilities`` when accessed via the
    # facade, so ``u.Field(default_factory=...)`` was mis-inferred as returning
    # ``FlextUtilities`` (bogus reportAssignmentType). ``mp.Field`` already does this.
    Field = staticmethod(mp.Field)
    PrivateAttr = staticmethod(mp.PrivateAttr)
    SkipValidation = SkipValidation

    # Keep the models-layer overload set intact; staticmethod's generic
    # constructor infers only its first arm. Runtime still uses the same factory.
    if TYPE_CHECKING:
        field_validator = mp.field_validator
        field_serializer = mp.field_serializer
        model_validator = mp.model_validator
    else:
        field_validator = staticmethod(mp.field_validator)
        field_serializer = staticmethod(mp.field_serializer)
        model_validator = staticmethod(mp.model_validator)

    computed_field = staticmethod(mp.computed_field)
    model_serializer = staticmethod(model_serializer)

    AfterValidator = AfterValidator
    BeforeValidator = mp.BeforeValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    if TYPE_CHECKING:
        validate_call: p.ValidateCall = validate_call
    else:
        validate_call = staticmethod(validate_call)
    with_config = staticmethod(with_config)

    from_json = staticmethod(from_json)
    to_json = to_json
    to_jsonable_python = to_jsonable_python

    # Adapter construction keeps pydantic's own constructor signature, which
    # accepts every type form (classes, unions, ``Annotated`` and PEP 695
    # aliases). ``m.TypeAdapter[T]`` is the annotation. ``TypeAdapter`` is the
    # public constructor on this facade, the same class pydantic exports.
    # ``type_adapter`` is that constructor under the function-shaped name
    # already used by migrated callers.
    TypeAdapter = PydanticTypeAdapter
    type_adapter = TypeAdapter
