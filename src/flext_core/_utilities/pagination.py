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

from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence

from pydantic import BaseModel

from flext_core import FlextRuntime, c, r, t


class FlextUtilitiesPagination:
    """Pagination utilities for API responses.

    Provides methods for extracting pagination parameters from requests,
    validating them, preparing paginated data, and building responses.
    All methods use r for consistent error handling.
    """

    @staticmethod
    def build_pagination_response(
        pagination_data: Mapping[
            str,
            Sequence[t.RuntimeAtomic] | t.FlatContainerMapping | str,
        ],
        message: str | None = None,
    ) -> r[t.StrMapping]:
        """Build paginated response dictionary.

        Args:
            pagination_data: Pagination data from prepare_pagination_data
            message: Optional response message

        Returns:
            r with response dictionary or error

        """
        data = pagination_data.get(c.FIELD_DATA)
        pagination = pagination_data.get("pagination")
        if data is None or pagination is None:
            return r[t.StrMapping].fail("Invalid pagination data structure")
        data_val: str = data if isinstance(data, str) else str(data)
        pagination_val: str = (
            pagination if isinstance(pagination, str) else str(pagination)
        )
        response: MutableMapping[str, str] = {
            c.FIELD_DATA: data_val,
            "pagination": pagination_val,
        }
        if message is not None:
            response["message"] = message
        return r[t.StrMapping].ok(response)

    @staticmethod
    def extract_page_params(
        query_params: t.StrMapping,
        *,
        default_page: int = 1,
        default_page_size: int = c.DEFAULT_PAGE_SIZE_EXAMPLE,
        max_page_size: int = c.MAX_PAGE_SIZE_EXAMPLE,
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
        page_str = str(default_page)
        if "page" in query_params:
            page_str = query_params["page"]
        page_size_str = str(default_page_size)
        if "page_size" in query_params:
            page_size_str = query_params["page_size"]
        page_result = r[int].ok(0).map(lambda _: int(page_str))
        if page_result.is_failure:
            return r[tuple[int, int]].fail(
                f"Invalid page parameters: {page_result.error}",
            )
        page_size_result = r[int].ok(0).map(lambda _: int(page_size_str))
        if page_size_result.is_failure:
            return r[tuple[int, int]].fail(
                f"Invalid page parameters: {page_size_result.error}",
            )
        page = page_result.value
        page_size = page_size_result.value
        if page < 1:
            return r[tuple[int, int]].fail("Page must be >= 1")
        if page_size < 1:
            return r[tuple[int, int]].fail("Page size must be >= 1")
        if page_size > max_page_size:
            return r[tuple[int, int]].fail(f"Page size must be <= {max_page_size}")
        return r[tuple[int, int]].ok((page, page_size))

    @staticmethod
    def extract_pagination_config(
        config: BaseModel | None,
    ) -> Mapping[str, int]:
        """Extract pagination configuration values - no fallbacks.

        Args:
            config: Configuration t.NormalizedValue or None

        Returns:
            Dictionary with pagination config values

        """
        default_page_size = c.DEFAULT_PAGE_SIZE_EXAMPLE
        max_page_size = c.MAX_PAGE_SIZE_EXAMPLE
        if config is not None:
            default_page_size_attr = getattr(config, "default_page_size", None)
            if default_page_size_attr is not None:
                match default_page_size_attr:
                    case int() as page_size if page_size > 0:
                        default_page_size = page_size
                    case _:
                        pass
            max_page_size_attr = getattr(config, "max_page_size", None)
            if max_page_size_attr is not None:
                match max_page_size_attr:
                    case int() as page_size if page_size > 0:
                        max_page_size = page_size
                    case _:
                        pass
        return {"default_page_size": default_page_size, "max_page_size": max_page_size}

    @staticmethod
    def prepare_pagination_data(
        data: t.FlatContainerList | None,
        total: int | None,
        *,
        page: int,
        page_size: int,
    ) -> r[Mapping[str, Sequence[t.RuntimeAtomic] | t.PaginationMeta]]:
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
            data: t.FlatContainerList = []
        total_count = total if total is not None else len(data)
        total_pages = (total_count + page_size - 1) // page_size
        if page > total_pages > 0:
            return r[Mapping[str, Sequence[t.RuntimeAtomic] | t.PaginationMeta]].fail(
                f"Page {page} exceeds total pages {total_pages}",
            )
        has_next: bool = page < total_pages
        has_prev: bool = page > 1
        data_list: MutableSequence[t.RuntimeAtomic] = []
        for item in data:
            normalized = FlextRuntime.normalize_to_container(item)
            data_list.append(normalized)
        pagination_meta: t.PaginationMeta = {
            "page": page,
            "page_size": page_size,
            "total": total_count,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
        }
        return r[Mapping[str, Sequence[t.RuntimeAtomic] | t.PaginationMeta]].ok({
            c.FIELD_DATA: data_list,
            "pagination": pagination_meta,
        })

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


__all__ = ["FlextUtilitiesPagination"]
