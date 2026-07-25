"""Validation exception tuple constants for FlextConstantsErrors."""

from __future__ import annotations

from typing import Final

from pydantic import ValidationError as _PydanticValidationError


class FlextConstantsErrorsValidationExceptions:
    """Validation-heavy exception families for boundary catches."""

    EXC_VALIDATION_TYPE: Final[tuple[type[Exception], ...]] = (
        TypeError,
        _PydanticValidationError,
    )
    """Pydantic validation + type-validation catch for model-coercion boundaries."""

    EXC_VALIDATION_VALUE: Final[tuple[type[Exception], ...]] = (
        ValueError,
        _PydanticValidationError,
    )
    """Pydantic validation + value-validation catch for input-coercion boundaries."""

    EXC_PYDANTIC_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        _PydanticValidationError,
        TypeError,
        ValueError,
    )
    """Pydantic + typing + value-validation catch for model-construction flows."""

    EXC_VALIDATION_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        TypeError,
        ValueError,
        _PydanticValidationError,
    )
    """Pydantic validation + typing + value catch for full validation boundaries."""

    EXC_ATTR_KEY_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    )
    """Attribute + mapping + typing catch for object-state + dict boundaries."""

    EXC_FS_KEY_VALUE: Final[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
    )
    """Filesystem read + mapping + value-validation catch for config-file flows."""

    EXC_FS_FULL_DECODE: Final[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        OSError,
        PermissionError,
        UnicodeDecodeError,
        ValueError,
    )
    """Filesystem + permissions + decoding + value catch for full-text-file flows."""

    EXC_KEY_OS_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        KeyError,
        OSError,
        TypeError,
        ValueError,
    )
    """Mapping + filesystem + typing catch for IO-bound config flows."""

    EXC_OS_SYNTAX: Final[tuple[type[Exception], ...]] = (
        OSError,
        SyntaxError,
    )
    """Filesystem + syntax catch for source-parsing boundaries."""

    EXC_ATTR_RUNTIME_VALIDATION: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        _PydanticValidationError,
    )
    """Pydantic validation + runtime + attr catch for full model boundaries."""

    EXC_OS_VALIDATION: Final[tuple[type[Exception], ...]] = (
        OSError,
        TypeError,
        ValueError,
        _PydanticValidationError,
    )
    """IO + validation + typing catch for config-file model boundaries."""

    EXC_ATTR_KEY_OS_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    )
    """Object + mapping + filesystem + typing catch for full IO+state flows."""

    EXC_FS_TYPE_VALIDATION: Final[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        TypeError,
        ValueError,
        _PydanticValidationError,
    )
    """Filesystem + typing + Pydantic validation catch for config-load flows."""


__all__ = ["FlextConstantsErrorsValidationExceptions"]
