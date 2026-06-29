"""Runtime exception tuple constants for FlextConstantsErrors."""

from __future__ import annotations

from typing import Final


class FlextConstantsErrorsRuntimeExceptions:
    """Runtime and IO exception families for boundary catches."""

    CATCHABLE_RUNTIME_EXCEPTIONS: Final[tuple[type[Exception], ...]] = (
        ArithmeticError,
        AttributeError,
        KeyError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )

    EXC_BROAD_IO_TYPE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Broad-spectrum boundary catch: filesystem, import, type, runtime errors."""

    EXC_RUNTIME_TYPE: Final[tuple[type[Exception], ...]] = (
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Runtime + type-validation catch for basic adapter boundaries."""

    EXC_BASIC_TYPE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        TypeError,
        ValueError,
    )
    """Attribute + type-validation catch for object-shape adapter boundaries."""

    EXC_MAPPING_TYPE: Final[tuple[type[Exception], ...]] = (
        KeyError,
        TypeError,
        ValueError,
    )
    """Mapping access + type-validation catch for dict-shape boundaries."""

    EXC_TYPE_VALIDATION: Final[tuple[type[Exception], ...]] = (
        TypeError,
        ValueError,
    )
    """Minimal type-validation catch for value-coercion boundaries."""

    EXC_NETWORK_TYPE: Final[tuple[type[Exception], ...]] = (
        ConnectionError,
        TimeoutError,
        ValueError,
    )
    """Network connectivity + type-validation catch for HTTP/RPC boundaries."""

    EXC_HTTP_PROCESSING: Final[tuple[type[Exception], ...]] = (
        ConnectionError,
        KeyError,
        TypeError,
        ValueError,
    )
    """HTTP-shape boundary: connection + parsing + typing for request handlers."""

    EXC_BROAD_RUNTIME: Final[tuple[type[Exception], ...]] = (
        ArithmeticError,
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Broad runtime catch for adapter-internal flows (no IO, no import)."""

    EXC_OS_RUNTIME_TYPE: Final[tuple[type[Exception], ...]] = (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Filesystem + runtime + typing catch for IO-bound boundary code."""

    EXC_FS_DECODING: Final[tuple[type[Exception], ...]] = (
        FileNotFoundError,
        OSError,
        PermissionError,
        UnicodeDecodeError,
    )
    """Filesystem read + decoding catch for file-handler boundaries."""

    EXC_OS_VALUE: Final[tuple[type[Exception], ...]] = (
        OSError,
        ValueError,
    )
    """Filesystem + value-validation catch for path/IO boundaries."""

    EXC_OS_DECODING: Final[tuple[type[Exception], ...]] = (
        OSError,
        UnicodeDecodeError,
    )
    """Filesystem read + unicode decoding catch for text-file boundaries."""

    EXC_ATTR_RUNTIME_TYPE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Attribute access + runtime + typing catch for object-state boundaries."""

    EXC_OS_TYPE_VALUE: Final[tuple[type[Exception], ...]] = (
        OSError,
        TypeError,
        ValueError,
    )
    """Filesystem + typing catch for path/IO + value-validation boundaries."""

    EXC_ATTR_TYPE: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        TypeError,
    )
    """Minimal attribute-access + type catch for object-shape boundaries."""

    EXC_OS_TYPE: Final[tuple[type[Exception], ...]] = (
        OSError,
        TypeError,
    )
    """Filesystem + type-validation catch for path-handler boundaries."""

    EXC_BROAD_RUNTIME_OS: Final[tuple[type[Exception], ...]] = (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    )
    """Broad runtime + filesystem boundary catch (no ImportError)."""

    EXC_OS_RUNTIME_VALUE: Final[tuple[type[Exception], ...]] = (
        OSError,
        RuntimeError,
        ValueError,
    )
    """Filesystem + runtime + value-validation catch for IO-bound flows."""


__all__ = ["FlextConstantsErrorsRuntimeExceptions"]
