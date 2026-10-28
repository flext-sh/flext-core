"""Validation-heavy exception families for boundary catches.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from typing import ClassVar

from pydantic import ValidationError as _PydanticValidationError

from flext_core._constants._errors.flextconstantserrors_part_02 import (
    FlextConstantsErrorsRuntimeExceptions as _RuntimeExceptions,
)


class FlextConstantsErrorsValidationExceptions:
    """Validation-heavy exception families for boundary catches."""

    _PYDANTIC: ClassVar[tuple[type[Exception], ...]] = (_PydanticValidationError,)

    EXC_VALIDATION_TYPE: ClassVar[tuple[type[Exception], ...]] = (TypeError, *_PYDANTIC)
    """Pydantic validation + type-validation catch for model-coercion boundaries."""

    EXC_VALIDATION_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        ValueError,
        *_PYDANTIC,
    )
    """Pydantic validation + value-validation catch for input-coercion boundaries."""

    EXC_VALIDATION_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        _PYDANTIC + _RuntimeExceptions.EXC_TYPE_VALIDATION
    )
    """Pydantic validation + typing + value catch for full validation boundaries."""

    EXC_ATTR_KEY_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        AttributeError,
        *_RuntimeExceptions.EXC_MAPPING_TYPE,
    )
    """Attribute + mapping + typing catch for object-state + dict boundaries."""

    EXC_FS_KEY_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        KeyError,
        *_RuntimeExceptions.EXC_OS_VALUE,
    )
    """Filesystem read + mapping + value-validation catch for config-file flows."""

    EXC_FS_FULL_DECODE: ClassVar[tuple[type[Exception], ...]] = (
        *_RuntimeExceptions.EXC_FS_DECODING,
        ValueError,
    )
    """Filesystem + permissions + decoding + value catch for full-text-file flows."""

    EXC_KEY_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        KeyError,
        *_RuntimeExceptions.EXC_OS_TYPE_VALUE,
    )
    """Mapping + filesystem + typing catch for IO-bound config flows."""

    EXC_OS_SYNTAX: ClassVar[tuple[type[Exception], ...]] = (OSError, SyntaxError)
    """Filesystem + syntax catch for source-parsing boundaries."""

    EXC_ATTR_RUNTIME_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        _RuntimeExceptions.EXC_ATTR_RUNTIME_TYPE + _PYDANTIC
    )
    """Pydantic validation + runtime + attr catch for full model boundaries."""

    EXC_OS_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        _RuntimeExceptions.EXC_OS_TYPE_VALUE + _PYDANTIC
    )
    """IO + validation + typing catch for config-file model boundaries."""

    EXC_ATTR_KEY_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        AttributeError,
        *EXC_KEY_OS_TYPE_VALUE,
    )
    """Object + mapping + filesystem + typing catch for full IO+state flows."""

    EXC_FS_TYPE_VALIDATION: ClassVar[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        *_RuntimeExceptions.EXC_TYPE_VALIDATION,
        *_PYDANTIC,
    )
    """Filesystem + typing + Pydantic validation catch for config-load flows."""


__all__: tuple[str, ...] = ("FlextConstantsErrorsValidationExceptions",)
