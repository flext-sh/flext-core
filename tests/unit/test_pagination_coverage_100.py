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

from typing import Annotated, ClassVar

import pytest
from flext_tests import tm, u
from pydantic import BaseModel, ConfigDict, Field


class TestPaginationCoverage100:
    class ExtractPageParamsScenario(BaseModel):
        """Extract page params test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Extract page params scenario name")]
        query_params: Annotated[
            dict[str, str], Field(description="Input query parameters")
        ]
        default_page: Annotated[int, Field(description="Default page value")]
        default_page_size: Annotated[int, Field(description="Default page size value")]
        max_page_size: Annotated[int, Field(description="Maximum allowed page size")]
        expected_success: Annotated[
            bool, Field(description="Expected operation success")
        ]
        expected_page: Annotated[
            int | None, Field(description="Expected resolved page")
        ]
        expected_page_size: Annotated[
            int | None, Field(description="Expected resolved page size")
        ]
        expected_error: Annotated[
            str | None, Field(description="Expected error message")
        ]

    class ValidatePaginationParamsScenario(BaseModel):
        """Validate pagination params test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Validate pagination scenario name")]
        page: Annotated[int, Field(description="Input page number")]
        page_size: Annotated[int | None, Field(description="Input page size")]
        max_page_size: Annotated[int, Field(description="Maximum allowed page size")]
        expected_success: Annotated[
            bool, Field(description="Expected validation success")
        ]
        expected_page_size: Annotated[
            int | None, Field(description="Expected validated page size")
        ]
        expected_error: Annotated[
            str | None, Field(description="Expected validation error")
        ]

    class PreparePaginationDataScenario(BaseModel):
        """Prepare pagination data test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Prepare pagination data scenario name")]
        data: Annotated[list[str] | None, Field(description="Input page data")]
        total: Annotated[int | None, Field(description="Input total count")]
        page: Annotated[int, Field(description="Requested page")]
        page_size: Annotated[int, Field(description="Requested page size")]
        expected_success: Annotated[
            bool, Field(description="Expected preparation success")
        ]
        expected_total: Annotated[
            int | None, Field(description="Expected total in output")
        ]
        expected_total_pages: Annotated[
            int | None, Field(description="Expected total pages in output")
        ]
        expected_error: Annotated[
            str | None, Field(description="Expected preparation error")
        ]

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

    @pytest.mark.parametrize("scenario", PaginationScenarios.EXTRACT_PAGE_PARAMS)
    def test_extract_page_params(self, scenario: ExtractPageParamsScenario) -> None:
        """Test extract_page_params with various scenarios."""
        result = u.extract_page_params(
            scenario.query_params,
            default_page=scenario.default_page,
            default_page_size=scenario.default_page_size,
            max_page_size=scenario.max_page_size,
        )
        if scenario.expected_success:
            _ = u.Tests.Result.assert_success(result)
            page, page_size = result.value
            tm.that(page, eq=scenario.expected_page)
            tm.that(page_size, eq=scenario.expected_page_size)
        else:
            u.Tests.Result.assert_failure_with_error(
                result,
                expected_error=scenario.expected_error,
            )

    @pytest.mark.parametrize(
        "scenario",
        PaginationScenarios.VALIDATE_PAGINATION_PARAMS,
    )
    def test_validate_pagination_params(
        self,
        scenario: ValidatePaginationParamsScenario,
    ) -> None:
        """Test validate_pagination_params with various scenarios."""
        result = u.validate_pagination_params(
            page=scenario.page,
            page_size=scenario.page_size,
            max_page_size=scenario.max_page_size,
        )
        if scenario.expected_success:
            _ = u.Tests.Result.assert_success(result)
            params = result.value
            assert params["page"] == scenario.page
            assert params["page_size"] == scenario.expected_page_size
        else:
            u.Tests.Result.assert_failure_with_error(
                result,
                expected_error=scenario.expected_error,
            )

    @pytest.mark.parametrize(
        "scenario",
        PaginationScenarios.PREPARE_PAGINATION_DATA,
    )
    def test_prepare_pagination_data(
        self,
        scenario: PreparePaginationDataScenario,
    ) -> None:
        """Test prepare_pagination_data with various scenarios."""
        result = u.prepare_pagination_data(
            scenario.data,
            scenario.total,
            page=scenario.page,
            page_size=scenario.page_size,
        )
        if scenario.expected_success:
            _ = u.Tests.Result.assert_success(result)
            data = result.value
            tm.that(data, contains="data")
            tm.that(data, contains="pagination")
            pagination = data["pagination"]
            assert isinstance(pagination, dict)
            assert pagination["page"] == scenario.page
            assert pagination["page_size"] == scenario.page_size
            assert pagination["total"] == scenario.expected_total
            assert pagination["total_pages"] == scenario.expected_total_pages
            if scenario.expected_total_pages:
                assert pagination["has_next"] == (
                    scenario.page < scenario.expected_total_pages
                )
                assert pagination["has_prev"] == (scenario.page > 1)
        else:
            u.Tests.Result.assert_failure_with_error(
                result,
                expected_error=scenario.expected_error,
            )

    def test_build_pagination_response_success(self) -> None:
        """Test build_pagination_response with valid data."""
        pagination_data: dict[str, list[str] | dict[str, int | bool]] = {
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
        result = u.build_pagination_response(
            pagination_data,
            message="Success",
        )
        _ = u.Tests.Result.assert_success(result)
        response = result.value
        tm.that(response, contains="data")
        tm.that(response, contains="pagination")
        assert response["message"] == "Success"

    def test_build_pagination_response_no_message(self) -> None:
        """Test build_pagination_response without message."""
        pagination_data: dict[str, list[str] | dict[str, int | bool]] = {
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
        result = u.build_pagination_response(pagination_data)
        _ = u.Tests.Result.assert_success(result)
        response = result.value
        tm.that(response, contains="data")
        tm.that(response, contains="pagination")
        assert "message" not in response

    def test_build_pagination_response_missing_data(self) -> None:
        """Test build_pagination_response with missing data."""
        pagination_data: dict[str, dict[str, int | bool]] = {"pagination": {}}
        result = u.build_pagination_response(pagination_data)
        u.Tests.Result.assert_failure_with_error(
            result,
            expected_error="Invalid pagination data structure",
        )

    def test_build_pagination_response_missing_pagination(self) -> None:
        """Test build_pagination_response with missing pagination."""
        pagination_data: dict[str, list[str]] = {"data": []}
        result = u.build_pagination_response(pagination_data)
        u.Tests.Result.assert_failure_with_error(
            result,
            expected_error="Invalid pagination data structure",
        )

    def test_build_pagination_response_with_non_sequence_data(self) -> None:
        """Test build_pagination_response with non-sequence data."""
        pagination_data: dict[str, dict[str, str] | dict[str, int | bool]] = {
            "data": {"key": "value"},
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False,
            },
        }
        result = u.build_pagination_response(pagination_data)
        _ = u.Tests.Result.assert_success(result)
        response = result.value
        tm.that(response, contains="data")

    def test_extract_pagination_config_none(self) -> None:
        """Test extract_pagination_config with None."""
        result = u.extract_pagination_config(None)
        assert result["default_page_size"] == 20
        assert result["max_page_size"] == 1000

    def test_extract_pagination_config_with_attributes(self) -> None:
        """Test extract_pagination_config with object attributes."""

        class Config(BaseModel):
            default_page_size: int = 50
            max_page_size: int = 500

        result = u.extract_pagination_config(Config())
        assert result["default_page_size"] == 50
        assert result["max_page_size"] == 500

    def test_extract_pagination_config_partial_attributes(self) -> None:
        """Test extract_pagination_config with partial attributes."""

        class Config(BaseModel):
            default_page_size: int = 30

        result = u.extract_pagination_config(Config())
        assert result["default_page_size"] == 30
        assert result["max_page_size"] == 1000

    def test_extract_pagination_config_invalid_values(self) -> None:
        """Test extract_pagination_config with invalid values."""

        class Config(BaseModel):
            default_page_size: int = -10
            max_page_size: int = 0

        result = u.extract_pagination_config(Config())
        assert result["default_page_size"] == 20
        assert result["max_page_size"] == 1000

    def test_extract_pagination_config_with_dict(self) -> None:
        """Test extract_pagination_config with dict-like object."""

        class Config(BaseModel):
            default_page_size: int = 40
            max_page_size: int = 600

        result = u.extract_pagination_config(Config())
        assert result["default_page_size"] == 40
        assert result["max_page_size"] == 600


ExtractPageParamsScenario = TestPaginationCoverage100.ExtractPageParamsScenario
ValidatePaginationParamsScenario = ValidatePaginationParamsScenario
PreparePaginationDataScenario = TestPaginationCoverage100.PreparePaginationDataScenario
PaginationScenarios = TestPaginationCoverage100.PaginationScenarios
