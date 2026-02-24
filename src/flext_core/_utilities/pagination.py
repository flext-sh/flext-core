"""Pagination helpers that stay compatible with dispatcher flows.

Provides comprehensive pagination functionality for API responses,
including parameter extraction, validation, data preparation, and
response building with ``r``-based error handling. Keep
metadata deterministic so dispatcher handlers can compose paginated
results without side effects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core._utilities.conversion import FlextUtilitiesConversion
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import FlextConstants as c
from flext_core.result import FlextResult as r
from flext_core.runtime import FlextRuntime
from flext_core.typings import T, FlextTypes as t

# Use centralized conversion from conversion.py
_to_general_value = FlextUtilitiesConversion.to_general_value_type


class FlextUtilitiesPagination:
    """Pagination utilities for API responses.

    Provides methods for extracting pagination parameters from requests,
    validating them, preparing paginated data, and building responses.
    All methods use r for consistent error handling.
    """

    @staticmethod
    def extract_page_params(
        query_params: Mapping[str, str],
        *,
        default_page: int = 1,
        default_page_size: int = c.Pagination.DEFAULT_PAGE_SIZE_EXAMPLE,
        max_page_size: int = c.Pagination.MAX_PAGE_SIZE_EXAMPLE,
    ) -> r[tuple[int, int]]:
        """Extract page and page_size from query parameters.

        Args:
            query_params: Dictionary of query parameters
            default_page: Default page number if not provided
            default_page_size: Default page size if not provided
            max_page_size: Maximum allowed page size

        Returns:
            r with (page, page_size) tuple or error

        """
        # StringDict values are always str, so isinstance is redundant
        page_str = str(default_page)
        if "page" in query_params:
            page_str = query_params["page"]

        page_size_str = str(default_page_size)
        if "page_size" in query_params:
            page_size_str = query_params["page_size"]

        try:
            page = int(page_str)
            page_size = int(page_size_str)

            if page < 1:
                return r[tuple[int, int]].fail("Page must be >= 1")
            if page_size < 1:
                return r[tuple[int, int]].fail("Page size must be >= 1")
            if page_size > max_page_size:
                return r[tuple[int, int]].fail(f"Page size must be <= {max_page_size}")

            return r[tuple[int, int]].ok((page, page_size))
        except ValueError as e:
            return r[tuple[int, int]].fail(f"Invalid page parameters: {e}")

    @staticmethod
    def validate_pagination_params(
        *,
        page: int,
        page_size: int | None,
        max_page_size: int,
    ) -> r[Mapping[str, int]]:
        """Validate pagination parameters.

        Args:
            page: Page number
            page_size: Page size or None for default
            max_page_size: Maximum allowed page size

        Returns:
            r with validated parameters or error

        """
        if page < 1:
            return r[Mapping[str, int]].fail("Page must be >= 1")

        effective_page_size = page_size if page_size is not None else 20

        if effective_page_size < 1:
            return r[Mapping[str, int]].fail("Page size must be >= 1")
        if effective_page_size > max_page_size:
            return r[Mapping[str, int]].fail(f"Page size must be <= {max_page_size}")

        return r[Mapping[str, int]].ok({"page": page, "page_size": effective_page_size})

    @staticmethod
    def prepare_pagination_data(
        data: Sequence[T] | None,
        total: int | None,
        *,
        page: int,
        page_size: int,
    ) -> r[
        Mapping[str, t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]]
    ]:
        """Prepare pagination data structure.

        Args:
            data: Sequence of items for current page
            total: Total number of items across all pages
            page: Current page number
            page_size: Page size

        Returns:
            r with pagination data dictionary or error

        """
        if data is None:
            data = []

        # Calculate pagination metadata
        total_count = total if total is not None else len(data)
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

        # Ensure page is within bounds
        if page > total_pages > 0:
            return r[
                Mapping[
                    str,
                    t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue],
                ]
            ].fail(
                f"Page {page} exceeds total pages {total_pages}",
            )

        has_next = page < total_pages
        has_prev = page > 1

        # Convert Sequence[T] to t.ConfigMapValue-compatible list
        # First convert T to PayloadValue, then normalize recursively
        data_list: list[t.ScalarValue] = []
        for item in data:
            general_value = _to_general_value(item)
            normalized = FlextRuntime.normalize_to_general_value(general_value)
            data_list.append(normalized)

        return r[
            Mapping[
                str, t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]
            ]
        ].ok({
            "data": data_list,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        })

    @staticmethod
    def build_pagination_response(
        pagination_data: Mapping[
            str,
            t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue],
        ],
        message: str | None = None,
    ) -> r[
        Mapping[str, t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]]
    ]:
        """Build paginated response dictionary.

        Args:
            pagination_data: Pagination data from prepare_pagination_data
            message: Optional response message

        Returns:
            r with response dictionary or error

        """
        data = pagination_data.get("data")
        pagination = pagination_data.get("pagination")

        if data is None or pagination is None:
            return r[
                Mapping[
                    str,
                    t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue],
                ]
            ].fail(
                "Invalid pagination data structure",
            )

        data_typed: t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]
        pagination_typed: (
            t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]
        )

        # Validate types match t.ConfigMapValue
        data_class = data.__class__
        if (
            data_class in (str, int, float, bool, None.__class__)
            or Sequence in data_class.__mro__
            or Mapping in data_class.__mro__
        ):
            data_typed = data
        else:
            data_typed = str(data)

        pagination_class = pagination.__class__
        if (
            pagination_class in (str, int, float, bool, None.__class__)
            or Sequence in pagination_class.__mro__
            or Mapping in pagination_class.__mro__
        ):
            pagination_typed = pagination
        else:
            pagination_typed = str(pagination)

        response: Mapping[
            str,
            t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue],
        ] = {
            "data": data_typed,
            "pagination": pagination_typed,
        }

        if message is not None:
            response = {**response, "message": message}

        return r[
            Mapping[
                str, t.ScalarValue | list[t.ScalarValue] | Mapping[str, t.ScalarValue]
            ]
        ].ok(response)

    @staticmethod
    def extract_pagination_config(
        config: object | None,
    ) -> Mapping[str, int]:
        """Extract pagination configuration values - no fallbacks.

        Args:
            config: Configuration object or None

        Returns:
            Dictionary with pagination config values

        """
        # Default values
        default_page_size = c.Pagination.DEFAULT_PAGE_SIZE_EXAMPLE
        max_page_size = c.Pagination.MAX_PAGE_SIZE_EXAMPLE

        if config is not None:
            # Use getattr to safely access attributes without type narrowing issues
            default_page_size_attr = getattr(config, "default_page_size", None)
            match default_page_size_attr:
                case int() as page_size if page_size > 0:
                    default_page_size = page_size

            max_page_size_attr = getattr(config, "max_page_size", None)
            match max_page_size_attr:
                case int() as page_size if page_size > 0:
                    max_page_size = page_size

        return {
            "default_page_size": default_page_size,
            "max_page_size": max_page_size,
        }


__all__ = [
    "FlextUtilitiesPagination",
]
