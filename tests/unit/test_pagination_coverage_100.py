"""Real tests to achieve 100% pagination coverage - no mocks.

Module: flext_core._utilities.pagination
Scope: FlextUtilitiesPagination - all methods and edge cases

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in _utilities/pagination.py.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from flext_core import u
from flext_core.typings import t


@dataclass(frozen=True, slots=True)
class ExtractPageParamsScenario:
    """Extract page params test scenario."""

    name: str
    query_params: dict[str, str]
    default_page: int
    default_page_size: int
    max_page_size: int
    expected_success: bool
    expected_page: int | None
    expected_page_size: int | None
    expected_error: str | None


@dataclass(frozen=True, slots=True)
class ValidatePaginationParamsScenario:
    """Validate pagination params test scenario."""

    name: str
    page: int
    page_size: int | None
    max_page_size: int
    expected_success: bool
    expected_page_size: int | None
    expected_error: str | None


@dataclass(frozen=True, slots=True)
class PreparePaginationDataScenario:
    """Prepare pagination data test scenario."""

    name: str
    data: list[object] | None
    total: int | None
    page: int
    page_size: int
    expected_success: bool
    expected_total: int | None
    expected_total_pages: int | None
    expected_error: str | None


class PaginationScenarios:
    """Centralized pagination test scenarios."""

    EXTRACT_PAGE_PARAMS: ClassVar[list[ExtractPageParamsScenario]] = [
        ExtractPageParamsScenario(
            name="default_values",
            query_params={},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=True,
            expected_page=1,
            expected_page_size=20,
            expected_error=None,
        ),
        ExtractPageParamsScenario(
            name="valid_page_and_size",
            query_params={"page": "2", "page_size": "50"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=True,
            expected_page=2,
            expected_page_size=50,
            expected_error=None,
        ),
        ExtractPageParamsScenario(
            name="page_zero",
            query_params={"page": "0"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Page must be >= 1",
        ),
        ExtractPageParamsScenario(
            name="page_size_zero",
            query_params={"page_size": "0"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Page size must be >= 1",
        ),
        ExtractPageParamsScenario(
            name="page_size_exceeds_max",
            query_params={"page_size": "2000"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Page size must be <= 1000",
        ),
        ExtractPageParamsScenario(
            name="invalid_page_string",
            query_params={"page": "abc"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Invalid page parameters",
        ),
        ExtractPageParamsScenario(
            name="invalid_page_size_string",
            query_params={"page_size": "xyz"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Invalid page parameters",
        ),
        ExtractPageParamsScenario(
            name="negative_page",
            query_params={"page": "-1"},
            default_page=1,
            default_page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page=None,
            expected_page_size=None,
            expected_error="Page must be >= 1",
        ),
        ExtractPageParamsScenario(
            name="custom_defaults",
            query_params={},
            default_page=5,
            default_page_size=100,
            max_page_size=500,
            expected_success=True,
            expected_page=5,
            expected_page_size=100,
            expected_error=None,
        ),
    ]

    VALIDATE_PAGINATION_PARAMS: ClassVar[list[ValidatePaginationParamsScenario]] = [
        ValidatePaginationParamsScenario(
            name="valid_with_page_size",
            page=2,
            page_size=50,
            max_page_size=1000,
            expected_success=True,
            expected_page_size=50,
            expected_error=None,
        ),
        ValidatePaginationParamsScenario(
            name="valid_without_page_size",
            page=1,
            page_size=None,
            max_page_size=1000,
            expected_success=True,
            expected_page_size=20,
            expected_error=None,
        ),
        ValidatePaginationParamsScenario(
            name="page_zero",
            page=0,
            page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page_size=None,
            expected_error="Page must be >= 1",
        ),
        ValidatePaginationParamsScenario(
            name="page_size_zero",
            page=1,
            page_size=0,
            max_page_size=1000,
            expected_success=False,
            expected_page_size=None,
            expected_error="Page size must be >= 1",
        ),
        ValidatePaginationParamsScenario(
            name="page_size_exceeds_max",
            page=1,
            page_size=2000,
            max_page_size=1000,
            expected_success=False,
            expected_page_size=None,
            expected_error="Page size must be <= 1000",
        ),
        ValidatePaginationParamsScenario(
            name="negative_page",
            page=-1,
            page_size=20,
            max_page_size=1000,
            expected_success=False,
            expected_page_size=None,
            expected_error="Page must be >= 1",
        ),
    ]

    PREPARE_PAGINATION_DATA: ClassVar[list[PreparePaginationDataScenario]] = [
        PreparePaginationDataScenario(
            name="with_data_and_total",
            data=["item1", "item2", "item3"],
            total=100,
            page=1,
            page_size=20,
            expected_success=True,
            expected_total=100,
            expected_total_pages=5,
            expected_error=None,
        ),
        PreparePaginationDataScenario(
            name="with_data_no_total",
            data=["item1", "item2"],
            total=None,
            page=1,
            page_size=20,
            expected_success=True,
            expected_total=2,
            expected_total_pages=1,
            expected_error=None,
        ),
        PreparePaginationDataScenario(
            name="no_data_no_total",
            data=None,
            total=None,
            page=1,
            page_size=20,
            expected_success=True,
            expected_total=0,
            expected_total_pages=0,
            expected_error=None,
        ),
        PreparePaginationDataScenario(
            name="page_exceeds_total_pages",
            data=["item1"],
            total=10,
            page=10,
            page_size=20,
            expected_success=False,
            expected_total=None,
            expected_total_pages=None,
            expected_error="Page 10 exceeds total pages 1",
        ),
        PreparePaginationDataScenario(
            name="last_page_exact",
            data=["item1", "item2"],
            total=22,
            page=2,
            page_size=20,
            expected_success=True,
            expected_total=22,
            expected_total_pages=2,
            expected_error=None,
        ),
        PreparePaginationDataScenario(
            name="empty_data_with_total",
            data=[],
            total=0,
            page=1,
            page_size=20,
            expected_success=True,
            expected_total=0,
            expected_total_pages=0,
            expected_error=None,
        ),
    ]


class TestuPaginationExtractPageParams:
    """Test FlextUtilitiesPagination.extract_page_params."""

    @pytest.mark.parametrize("scenario", PaginationScenarios.EXTRACT_PAGE_PARAMS)
    def test_extract_page_params(self, scenario: ExtractPageParamsScenario) -> None:
        """Test extract_page_params with various scenarios."""
        result = u.Pagination.extract_page_params(
            scenario.query_params,
            default_page=scenario.default_page,
            default_page_size=scenario.default_page_size,
            max_page_size=scenario.max_page_size,
        )

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            page, page_size = result.value
            assert page == scenario.expected_page
            assert page_size == scenario.expected_page_size
        else:
            assert result.is_failure
            assert (
                result.error is not None
                and scenario.expected_error is not None
                and scenario.expected_error in result.error
            )


class TestuPaginationValidatePaginationParams:
    """Test FlextUtilitiesPagination.validate_pagination_params."""

    @pytest.mark.parametrize("scenario", PaginationScenarios.VALIDATE_PAGINATION_PARAMS)
    def test_validate_pagination_params(
        self,
        scenario: ValidatePaginationParamsScenario,
    ) -> None:
        """Test validate_pagination_params with various scenarios."""
        result = u.Pagination.validate_pagination_params(
            page=scenario.page,
            page_size=scenario.page_size,
            max_page_size=scenario.max_page_size,
        )

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            params = result.value
            assert params["page"] == scenario.page
            assert params["page_size"] == scenario.expected_page_size
        else:
            assert result.is_failure
            assert (
                result.error is not None
                and scenario.expected_error is not None
                and scenario.expected_error in result.error
            )


class TestuPaginationPreparePaginationData:
    """Test FlextUtilitiesPagination.prepare_pagination_data."""

    @pytest.mark.parametrize("scenario", PaginationScenarios.PREPARE_PAGINATION_DATA)
    def test_prepare_pagination_data(
        self,
        scenario: PreparePaginationDataScenario,
    ) -> None:
        """Test prepare_pagination_data with various scenarios."""
        result = u.Pagination.prepare_pagination_data(
            scenario.data,
            scenario.total,
            page=scenario.page,
            page_size=scenario.page_size,
        )

        assert result.is_success == scenario.expected_success

        if scenario.expected_success:
            assert result.is_success
            data = result.value
            assert "data" in data
            assert "pagination" in data

            pagination = data["pagination"]
            assert isinstance(pagination, dict)
            assert pagination["page"] == scenario.page
            assert pagination["page_size"] == scenario.page_size
            assert pagination["total"] == scenario.expected_total
            assert pagination["total_pages"] == scenario.expected_total_pages

            # Check has_next and has_prev
            if scenario.expected_total_pages:
                assert pagination["has_next"] == (
                    scenario.page < scenario.expected_total_pages
                )
                assert pagination["has_prev"] == (scenario.page > 1)
        else:
            assert result.is_failure
            assert (
                result.error is not None
                and scenario.expected_error is not None
                and scenario.expected_error in result.error
            )


class TestuPaginationBuildPaginationResponse:
    """Test FlextUtilitiesPagination.build_pagination_response."""

    def test_build_pagination_response_success(self) -> None:
        """Test build_pagination_response with valid data."""
        pagination_data: dict[str, t.GeneralValueType] = {
            "data": ["item1", "item2"],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total": 100,
                "total_pages": 5,
                "has_next": True,
                "has_prev": False,
            },
        }

        result = u.Pagination.build_pagination_response(
            pagination_data,
            message="Success",
        )

        assert result.is_success
        response = result.value
        assert "data" in response
        assert "pagination" in response
        assert response["message"] == "Success"

    def test_build_pagination_response_no_message(self) -> None:
        """Test build_pagination_response without message."""
        pagination_data: dict[str, t.GeneralValueType] = {
            "data": ["item1"],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False,
            },
        }

        result = u.Pagination.build_pagination_response(pagination_data)

        assert result.is_success
        response = result.value
        assert "data" in response
        assert "pagination" in response
        assert "message" not in response

    def test_build_pagination_response_missing_data(self) -> None:
        """Test build_pagination_response with missing data."""
        pagination_data: dict[str, t.GeneralValueType] = {"pagination": {}}

        result = u.Pagination.build_pagination_response(pagination_data)

        assert result.is_failure
        assert (
            result.error is not None
            and "Invalid pagination data structure" in result.error
        )

    def test_build_pagination_response_missing_pagination(self) -> None:
        """Test build_pagination_response with missing pagination."""
        pagination_data: dict[str, t.GeneralValueType] = {"data": []}

        result = u.Pagination.build_pagination_response(pagination_data)

        assert result.is_failure
        assert (
            result.error is not None
            and "Invalid pagination data structure" in result.error
        )

    def test_build_pagination_response_with_non_sequence_data(self) -> None:
        """Test build_pagination_response with non-sequence data."""
        pagination_data: dict[str, t.GeneralValueType] = {
            "data": {"key": "value"},  # dict instead of list
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False,
            },
        }

        result = u.Pagination.build_pagination_response(pagination_data)

        # Should still succeed - dict is valid GeneralValueType
        assert result.is_success
        response = result.value
        assert "data" in response


class TestuPaginationExtractPaginationConfig:
    """Test FlextUtilitiesPagination.extract_pagination_config."""

    def test_extract_pagination_config_none(self) -> None:
        """Test extract_pagination_config with None."""
        result = u.Pagination.extract_pagination_config(None)

        assert result["default_page_size"] == 20
        assert result["max_page_size"] == 1000

    def test_extract_pagination_config_with_attributes(self) -> None:
        """Test extract_pagination_config with object attributes."""

        class Config:
            default_page_size = 50
            max_page_size = 500

        config = Config()
        result = u.Pagination.extract_pagination_config(config)  # type: ignore[arg-type]  # Config is compatible with GeneralValueType at runtime

        assert result["default_page_size"] == 50
        assert result["max_page_size"] == 500

    def test_extract_pagination_config_partial_attributes(self) -> None:
        """Test extract_pagination_config with partial attributes."""

        class Config:
            default_page_size = 30

        config = Config()
        result = u.Pagination.extract_pagination_config(config)  # type: ignore[arg-type]  # Config is compatible with GeneralValueType at runtime

        assert result["default_page_size"] == 30
        assert result["max_page_size"] == 1000  # Default

    def test_extract_pagination_config_invalid_values(self) -> None:
        """Test extract_pagination_config with invalid values."""

        class Config:
            default_page_size = -10
            max_page_size = 0

        config = Config()
        result = u.Pagination.extract_pagination_config(config)  # type: ignore[arg-type]  # Config is compatible with GeneralValueType at runtime

        # Invalid values should be ignored, defaults used
        assert result["default_page_size"] == 20
        assert result["max_page_size"] == 1000

    def test_extract_pagination_config_with_dict(self) -> None:
        """Test extract_pagination_config with dict-like object."""

        class Config:
            def __init__(self) -> None:
                self.default_page_size = 40
                self.max_page_size = 600

        config = Config()
        result = u.Pagination.extract_pagination_config(config)  # type: ignore[arg-type]  # Config is compatible with GeneralValueType at runtime

        assert result["default_page_size"] == 40
        assert result["max_page_size"] == 600
