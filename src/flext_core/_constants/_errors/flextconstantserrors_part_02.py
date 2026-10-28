"""Runtime and IO exception families for boundary catches.

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


class FlextConstantsErrorsRuntimeExceptions:
    """Runtime and IO exception families for boundary catches."""

    _ARITH: ClassVar[tuple[type[Exception], ...]] = (ArithmeticError,)
    _ATTR: ClassVar[tuple[type[Exception], ...]] = (AttributeError,)
    _CONN: ClassVar[tuple[type[Exception], ...]] = (ConnectionError,)
    _FS: ClassVar[tuple[type[Exception], ...]] = (FileNotFoundError,)
    _KEY: ClassVar[tuple[type[Exception], ...]] = (KeyError,)
    _OS: ClassVar[tuple[type[Exception], ...]] = (OSError,)
    _RUNTIME: ClassVar[tuple[type[Exception], ...]] = (RuntimeError,)
    _TYPE: ClassVar[tuple[type[Exception], ...]] = (TypeError,)
    _UNICODE_DECODE: ClassVar[tuple[type[Exception], ...]] = (UnicodeDecodeError,)
    _VALUE: ClassVar[tuple[type[Exception], ...]] = (ValueError,)

    EXC_TYPE_VALIDATION: ClassVar[tuple[type[Exception], ...]] = _TYPE + _VALUE
    """Minimal type-validation catch for value-coercion boundaries."""

    EXC_ATTR_TYPE: ClassVar[tuple[type[Exception], ...]] = _ATTR + _TYPE
    """Minimal attribute-access + type catch for object-shape boundaries."""

    EXC_OS_TYPE: ClassVar[tuple[type[Exception], ...]] = _OS + _TYPE
    """Filesystem + type-validation catch for path-handler boundaries."""

    EXC_OS_VALUE: ClassVar[tuple[type[Exception], ...]] = _OS + _VALUE
    """Filesystem + value-validation catch for path/IO boundaries."""

    EXC_OS_DECODING: ClassVar[tuple[type[Exception], ...]] = _OS + _UNICODE_DECODE
    """Filesystem read + unicode decoding catch for text-file boundaries."""

    EXC_OS_RUNTIME_VALUE: ClassVar[tuple[type[Exception], ...]] = (
        _OS + _RUNTIME + _VALUE
    )
    """Filesystem + runtime + value-validation catch for IO-bound flows."""

    EXC_FS_DECODING: ClassVar[tuple[type[Exception], ...]] = (
        _FS + _OS + (PermissionError,) + _UNICODE_DECODE
    )
    """Filesystem read + decoding catch for file-handler boundaries."""

    EXC_NETWORK_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        *_CONN,
        TimeoutError,
        *_VALUE,
    )
    """Network connectivity + type-validation catch for HTTP/RPC boundaries."""

    EXC_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Runtime + type-validation catch for basic adapter boundaries."""

    EXC_BASIC_TYPE: ClassVar[tuple[type[Exception], ...]] = _ATTR + EXC_TYPE_VALIDATION
    """Attribute + type-validation catch for object-shape adapter boundaries."""

    EXC_MAPPING_TYPE: ClassVar[tuple[type[Exception], ...]] = _KEY + EXC_TYPE_VALIDATION
    """Mapping access + type-validation catch for dict-shape boundaries."""

    EXC_OS_TYPE_VALUE: ClassVar[tuple[type[Exception], ...]] = _OS + EXC_TYPE_VALIDATION
    """Filesystem + typing catch for path/IO + value-validation boundaries."""

    EXC_ATTR_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _ATTR + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Attribute access + runtime + typing catch for object-state boundaries."""

    EXC_OS_RUNTIME_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Filesystem + runtime + typing catch for IO-bound boundary code."""

    EXC_BROAD_RUNTIME: ClassVar[tuple[type[Exception], ...]] = (
        _ARITH + _ATTR + _KEY + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Broad runtime catch for adapter-internal flows (no IO, no import)."""

    EXC_BROAD_RUNTIME_OS: ClassVar[tuple[type[Exception], ...]] = (
        _ATTR + _KEY + _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )
    """Broad runtime + filesystem boundary catch (no ImportError)."""

    EXC_BROAD_IO_TYPE: ClassVar[tuple[type[Exception], ...]] = (
        *_ATTR,
        ImportError,
        *_KEY,
        *_OS,
        *_RUNTIME,
        *EXC_TYPE_VALIDATION,
    )
    """Broad-spectrum boundary catch: filesystem, import, type, runtime errors."""

    CATCHABLE_RUNTIME_EXCEPTIONS: ClassVar[tuple[type[Exception], ...]] = (
        _ARITH + _ATTR + _KEY + (LookupError,) + _OS + _RUNTIME + EXC_TYPE_VALIDATION
    )

    EXC_HTTP_PROCESSING: ClassVar[tuple[type[Exception], ...]] = (
        _CONN + EXC_MAPPING_TYPE
    )
    """HTTP-shape boundary: connection + parsing + typing for request handlers."""


__all__: tuple[str, ...] = ("FlextConstantsErrorsRuntimeExceptions",)
