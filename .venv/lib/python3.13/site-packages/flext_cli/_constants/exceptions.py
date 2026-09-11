"""Canonical CLI validation and serialization exceptions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from ruamel.yaml import YAMLError as RuamelYAMLError
from yaml import YAMLError

from flext_core import e


# NOTE (multi-agent): base classes come from the PUBLIC ``flext_core.e`` facade.
# ``e.ValidationError`` IS ``FlextExceptionsTypes.ValidationError`` (same class
# object via MRO) — verified, behavior-preserving. Do not re-import the private
# ``flext_core._exceptions.types`` module (ruff PLC2701).
class CliDefinitionError(e.ValidationError):
    """Located CLI definition-time failure (route/model/field).

    Raised while building/registering the automatic CLI so that consumers
    surface ``[CLI_DEFINITION_ERROR] command '<name>' field '<field>': ...``
    instead of a raw stacktrace. Inherits ``[code] message`` rendering and
    correlation metadata from ``flext_core`` ``e.ValidationError``.
    """

    _default_error_code: ClassVar[str] = "CLI_DEFINITION_ERROR"


class CliValidationError(e.ValidationError):
    """Located CLI runtime input failure (command/field).

    Raised when ``model_validate`` rejects user input so that consumers
    surface ``[VALIDATION_ERROR] command '<name>' field '<field>': ...``
    instead of pydantic's multi-line dump.
    """

    _default_error_code: ClassVar[str] = "VALIDATION_ERROR"


class FlextCliConstantsExceptions:
    """Canonical owned exception types for cross-project consumption."""

    # NOTE (multi-agent): local exception classes MUST keep their exact
    # ``type[<Class>]`` annotation. Widening to a generic exception hides
    # the real ``FlextBaseError.__init__`` (with **extra_kwargs) from pyrefly,
    # which then rejects ``command=``/``model=`` kwargs at raise sites
    # (validation.py).
    YamlParseError: ClassVar[type[Exception]] = YAMLError
    YamlRoundtripError: ClassVar[type[Exception]] = RuamelYAMLError
    CliDefinitionError: ClassVar[type[CliDefinitionError]] = CliDefinitionError
    CliValidationError: ClassVar[type[CliValidationError]] = CliValidationError


__all__: list[str] = [
    "CliDefinitionError",
    "CliValidationError",
    "FlextCliConstantsExceptions",
]
