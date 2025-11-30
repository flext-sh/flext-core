"""FlextUtilitiesPagination - Pagination utilities for FLEXT ecosystem.

Provides comprehensive pagination functionality for API responses,
including parameter extraction, validation, data preparation, and
response building with FlextResult-based error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from flext_core.result import FlextResult

T = TypeVar("T")


class FlextUtilitiesPagination:
    """Pagination utilities for API responses.

    Provides methods for extracting pagination parameters from requests,
    validating them, preparing paginated data, and building responses.
    All methods use FlextResult for consistent error handling.
    """

    @staticmethod
    def extract_page_params(
        query_params: dict[str, str],
        *,
        default_page: int = 1,
        default_page_size: int = 20,
        max_page_size: int = 1000,
    ) -> FlextResult[tuple[int, int]]:
        """Extract page and page_size from query parameters.

        Args:
            query_params: Dictionary of query parameters
            default_page: Default page number if not provided
            default_page_size: Default page size if not provided
            max_page_size: Maximum allowed page size

        Returns:
            FlextResult with (page, page_size) tuple or error
        """
        page_str = str(default_page)
        if "page" in query_params:
            page_value = query_params["page"]
            if isinstance(page_value, str):
                page_str = page_value

        page_size_str = str(default_page_size)
        if "page_size" in query_params:
            page_size_value = query_params["page_size"]
            if isinstance(page_size_value, str):
                page_size_str = page_size_value

        try:
            page = int(page_str)
            page_size = int(page_size_str)

            if page < 1:
                return FlextResult.fail("Page must be >= 1")
            if page_size < 1:
                return FlextResult.fail("Page size must be >= 1")
            if page_size > max_page_size:
                return FlextResult.fail(f"Page size must be <= {max_page_size}")

            return FlextResult.ok((page, page_size))
        except ValueError as e:
            return FlextResult.fail(f"Invalid page parameters: {e}")

    @staticmethod
    def validate_pagination_params(
        *,
        page: int,
        page_size: int | None,
        max_page_size: int,
    ) -> FlextResult[dict[str, int]]:
        """Validate pagination parameters.

        Args:
            page: Page number
            page_size: Page size or None for default
            max_page_size: Maximum allowed page size

        Returns:
            FlextResult with validated parameters or error
        """
        if page < 1:
            return FlextResult.fail("Page must be >= 1")

        effective_page_size = page_size if page_size is not None else 20

        if effective_page_size < 1:
            return FlextResult.fail("Page size must be >= 1")
        if effective_page_size > max_page_size:
            return FlextResult.fail(f"Page size must be <= {max_page_size}")

        return FlextResult.ok({"page": page, "page_size": effective_page_size})

    @staticmethod
    def prepare_pagination_data(
        data: Sequence[T] | None,
        total: int | None,
        *,
        page: int,
        page_size: int,
    ) -> FlextResult[dict[str, object]]:
        """Prepare pagination data structure.

        Args:
            data: Sequence of items for current page
            total: Total number of items across all pages
            page: Current page number
            page_size: Page size

        Returns:
            FlextResult with pagination data dictionary or error
        """
        if data is None:
            data = []

        # Calculate pagination metadata
        total_count = total if total is not None else len(data)
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

        # Ensure page is within bounds
        if page > total_pages and total_pages > 0:
            return FlextResult.fail(f"Page {page} exceeds total pages {total_pages}")

        has_next = page < total_pages
        has_prev = page > 1

        return FlextResult.ok({
            "data": list(data),
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
        pagination_data: dict[str, object],
        message: str | None = None,
    ) -> FlextResult[dict[str, Any]]:
        """Build paginated response dictionary.

        Args:
            pagination_data: Pagination data from prepare_pagination_data
            message: Optional response message

        Returns:
            FlextResult with response dictionary or error
        """
        data = pagination_data.get("data")
        pagination = pagination_data.get("pagination")

        if data is None or pagination is None:
            return FlextResult.fail("Invalid pagination data structure")

        response: dict[str, Any] = {
            "data": data,
            "pagination": pagination,
        }

        if message is not None:
            response["message"] = message

        return FlextResult.ok(response)

    @staticmethod
    def extract_pagination_config(config: object | None) -> dict[str, int]:
        """Extract pagination configuration values - no fallbacks.

        Args:
            config: Configuration object or None

        Returns:
            Dictionary with pagination config values
        """
        # Default values
        default_page_size = 20
        max_page_size = 1000

        if config is not None:
            if hasattr(config, "default_page_size"):
                default_page_size_value = getattr(config, "default_page_size")
                if (
                    isinstance(default_page_size_value, int)
                    and default_page_size_value > 0
                ):
                    default_page_size = default_page_size_value

            if hasattr(config, "max_page_size"):
                max_page_size_value = getattr(config, "max_page_size")
                if isinstance(max_page_size_value, int) and max_page_size_value > 0:
                    max_page_size = max_page_size_value

        return {
            "default_page_size": default_page_size,
            "max_page_size": max_page_size,
        }
