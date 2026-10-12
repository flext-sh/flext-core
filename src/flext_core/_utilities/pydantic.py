"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import (
    AfterValidator,
    PlainSerializer,
    PlainValidator,
    SkipValidation,
    WrapSerializer,
    WrapValidator,
    validate_call,
    with_config,
)
from pydantic_core import from_json, to_json, to_jsonable_python

from flext_core._models import FlextModelsPydantic
from flext_core._protocols import FlextProtocolsPydantic
from flext_core._utilities import (
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
)


class FlextUtilitiesPydantic(
    FlextModelsPydantic,
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
):
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    # Inherit field/default factories without re-inferring generic descriptors.
    SkipValidation = SkipValidation

    # Keep the models-layer overload set intact; staticmethod's generic
    # constructor infers only its first arm. Runtime still uses the same factory.
    if TYPE_CHECKING:
        field_validator = FlextModelsPydantic.field_validator
    else:
        field_validator = staticmethod(FlextModelsPydantic.field_validator)
        model_validator = staticmethod(FlextModelsPydantic.model_validator)

    AfterValidator = AfterValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    validate_call: ClassVar[FlextProtocolsPydantic.ValidateCall] = validate_call
    with_config = staticmethod(with_config)

    from_json = staticmethod(from_json)
    to_json = to_json
    to_jsonable_python = to_jsonable_python


__all__: tuple[str, ...] = ("FlextUtilitiesPydantic",)
