"""Exception handling utilities - standalone to avoid circular imports.

This module provides helpers for exception parameter preparation and metadata extraction.
It's a standalone module (not in _utilities/) to avoid circular dependency:
  exceptions → _utilities → result → exceptions

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final, cast

# Reserved keyword names used across all exception classes
EXCEPTION_RESERVED_KEYS: Final[frozenset[str]] = frozenset({
    "correlation_id",
    "metadata",
    "auto_log",
    "auto_correlation",
    "config",
})


def prepare_exception_kwargs(
    kwargs: dict[str, object],
    specific_params: dict[str, object] | None = None,
) -> tuple[
    str | None,
    dict[str, object] | None,
    bool,
    bool,
    object | None,
    dict[str, object],
]:
    """Prepare kwargs for exception initialization (DRY helper).

    Extracts common parameters and filters kwargs for BaseError.__init__().
    Eliminates 30-40 lines of duplicate code from each exception class.

    Args:
        kwargs: Raw kwargs from exception __init__
        specific_params: Dict of specific parameters to add to metadata

    Returns:
        Tuple of (correlation_id, metadata, auto_log, auto_correlation,
                  config, extra_kwargs)

    """
    # Add specific params to kwargs for metadata
    if specific_params:
        for key, value in specific_params.items():
            if value is not None:
                kwargs.setdefault(key, value)

    # Extract common parameters with proper type casting
    correlation_id = cast("str | None", kwargs.get("correlation_id"))
    metadata = cast("dict[str, object] | None", kwargs.get("metadata"))
    auto_log = bool(kwargs.get("auto_log"))
    auto_correlation = bool(kwargs.get("auto_correlation"))
    config = kwargs.get("config")

    # Filter out reserved keys
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in EXCEPTION_RESERVED_KEYS}

    return (
        correlation_id,
        metadata,
        auto_log,
        auto_correlation,
        config,
        extra_kwargs,
    )


def extract_common_kwargs(
    kwargs: dict[str, object],
) -> tuple[str | None, dict[str, object] | None]:
    """Extract correlation_id and metadata from kwargs.

    Used by exception factory methods (e.g., create()) to extract
    common parameters before passing to __init__().

    Args:
        kwargs: Raw kwargs containing correlation_id and/or metadata

    Returns:
        Tuple of (correlation_id, metadata)

    """
    correlation_id = cast("str | None", kwargs.get("correlation_id"))
    metadata = cast("dict[str, object] | None", kwargs.get("metadata"))
    return correlation_id, metadata


__all__ = [
    "EXCEPTION_RESERVED_KEYS",
    "extract_common_kwargs",
    "prepare_exception_kwargs",
]
