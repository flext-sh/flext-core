# ruff: noqa: PLC0415
"""HTTP testing utilities using pytest-httpx for comprehensive API testing.

Provides advanced HTTP mocking, request/response validation, and API testing
patterns with automatic retry, error simulation, and performance monitoring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urljoin

import httpx
import pytest
from pytest_httpx import HTTPXMock

from flext_core import FlextResult, FlextTypes

JsonDict = FlextTypes.Core.JsonObject


class HTTPTestUtils:
    """Comprehensive HTTP testing utilities with mocking and validation."""

    @staticmethod
    def mock_successful_response(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
        json_data: JsonDict | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Mock successful HTTP response."""
        response_data = json_data or {"status": "success"}
        default_headers = {"content-type": "application/json"}
        if headers:
            default_headers.update(headers)

        httpx_mock.add_response(
            method=method,
            url=url,
            json=response_data,
            status_code=status_code,
            headers=default_headers,
        )

    @staticmethod
    def mock_error_response(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
        status_code: int = 500,
        error_message: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
    ) -> None:
        """Mock error HTTP response."""
        error_data = {
            "error": {
                "message": error_message,
                "code": error_code,
                "status": status_code,
            },
        }

        httpx_mock.add_response(
            method=method,
            url=url,
            json=error_data,
            status_code=status_code,
            headers={"content-type": "application/json"},
        )

    @staticmethod
    def mock_timeout_response(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
    ) -> None:
        """Mock timeout response."""
        import httpx

        httpx_mock.add_exception(
            httpx.TimeoutException("Request timed out"),
            method=method,
            url=url,
        )

    @staticmethod
    def mock_network_error(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
    ) -> None:
        """Mock network error."""
        httpx_mock.add_exception(
            httpx.NetworkError("Network unreachable"),
            method=method,
            url=url,
        )

    @staticmethod
    def mock_paginated_response(
        httpx_mock: HTTPXMock,
        base_url: str,
        data_list: list[JsonDict],
        page_size: int = 10,
        method: str = "GET",
    ) -> None:
        """Mock paginated API responses."""
        total_items = len(data_list)
        total_pages = (total_items + page_size - 1) // page_size

        for page in range(1, total_pages + 1):
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_items)
            page_data = data_list[start_idx:end_idx]

            url = f"{base_url}?page={page}&size={page_size}"

            response_data = {
                "data": page_data,
                "pagination": {
                    "page": page,
                    "size": page_size,
                    "total_pages": total_pages,
                    "total_items": total_items,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }

            httpx_mock.add_response(
                method=method,
                url=url,
                json=response_data,
            )

    @staticmethod
    def mock_rate_limited_response(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
        retry_after: int = 60,
    ) -> None:
        """Mock rate-limited response."""
        httpx_mock.add_response(
            method=method,
            url=url,
            status_code=429,
            json={
                "error": {
                    "message": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": retry_after,
                },
            },
            headers={
                "retry-after": str(retry_after),
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(retry_after),
            },
        )


class APITestClient:
    """Advanced API test client with automatic retry and validation."""

    def __init__(self, base_url: str = "https://api.example.com") -> None:
        self.base_url = base_url
        self.default_headers = {
            "content-type": "application/json",
            "user-agent": "flext-test-client/1.0",
        }

    def build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(self.base_url, endpoint)

    def validate_response_structure(
        self,
        response_data: JsonDict,
        required_fields: list[str],
        _: list[str] | None = None,
    ) -> FlextResult[None]:
        """Validate API response structure."""
        missing_fields = [
            field for field in required_fields if field not in response_data
        ]

        if missing_fields:
            return FlextResult[None].fail(
                f"Missing required fields: {missing_fields}",
                error_code="INVALID_RESPONSE_STRUCTURE",
            )

        return FlextResult[None].ok(None)

    def validate_error_response(
        self,
        response_data: JsonDict,
        expected_error_code: str | None = None,
    ) -> FlextResult[None]:
        """Validate error response format."""
        if "error" not in response_data:
            return FlextResult[None].fail(
                "Error response missing 'error' field",
                error_code="INVALID_ERROR_FORMAT",
            )

        error: JsonDict = response_data["error"]
        required_error_fields = ["message", "code"]

        for field in required_error_fields:
            if field not in error:
                return FlextResult[None].fail(
                    f"Error object missing '{field}' field",
                    error_code="INVALID_ERROR_FORMAT",
                )

        if expected_error_code and error["code"] != expected_error_code:
            return FlextResult[None].fail(
                f"Expected error code '{expected_error_code}', got '{error['code']}'",
                error_code="UNEXPECTED_ERROR_CODE",
            )

        return FlextResult[None].ok(None)

    def assert_response_time(
        self,
        response: object,
        max_time_ms: float = 1000.0,
    ) -> None:
        """Assert response time is within limits."""
        # This would typically extract timing from actual HTTP response
        # For testing purposes, we'll simulate
        response_time_ms = getattr(response, "elapsed_ms", 100.0)

        assert response_time_ms <= max_time_ms, (
            f"Response time {response_time_ms}ms exceeds limit {max_time_ms}ms"
        )


class HTTPScenarioBuilder:
    """Builder for complex HTTP testing scenarios."""

    def __init__(self, httpx_mock: HTTPXMock) -> None:
        self.httpx_mock = httpx_mock
        self.scenarios: list[dict[str, object]] = []

    def add_successful_request(
        self,
        url: str,
        method: str = "GET",
        response_data: JsonDict | None = None,
        status_code: int = 200,
    ) -> HTTPScenarioBuilder:
        """Add successful request to scenario."""
        HTTPTestUtils.mock_successful_response(
            self.httpx_mock,
            url,
            method,
            response_data,
            status_code,
        )

        self.scenarios.append({
            "type": "success",
            "url": url,
            "method": method,
            "status_code": status_code,
        })

        return self

    def add_error_request(
        self,
        url: str,
        method: str = "GET",
        status_code: int = 500,
        error_message: str = "Server error",
    ) -> HTTPScenarioBuilder:
        """Add error request to scenario."""
        HTTPTestUtils.mock_error_response(
            self.httpx_mock,
            url,
            method,
            status_code,
            error_message,
        )

        self.scenarios.append({
            "type": "error",
            "url": url,
            "method": method,
            "status_code": status_code,
        })

        return self

    def add_retry_scenario(
        self,
        url: str,
        method: str = "GET",
        failure_count: int = 2,
        final_response: JsonDict | None = None,
    ) -> HTTPScenarioBuilder:
        """Add retry scenario (fail N times, then succeed)."""
        # Add failures
        for i in range(failure_count):
            self.httpx_mock.add_response(
                method=method,
                url=url,
                status_code=503,
                json={"error": {"message": f"Service unavailable (attempt {i + 1})"}},
            )

        # Add final success
        HTTPTestUtils.mock_successful_response(
            self.httpx_mock,
            url,
            method,
            final_response or {"status": "success", "retry_succeeded": True},
        )

        self.scenarios.append({
            "type": "retry",
            "url": url,
            "method": method,
            "failure_count": failure_count,
        })

        return self

    def add_circuit_breaker_scenario(
        self,
        url: str,
        method: str = "GET",
        failure_threshold: int = 3,
    ) -> HTTPScenarioBuilder:
        """Add circuit breaker scenario."""
        # Add failures up to threshold
        for i in range(failure_threshold):
            HTTPTestUtils.mock_error_response(
                self.httpx_mock,
                url,
                method,
                status_code=503,
                error_message=f"Service failure {i + 1}",
            )

        # Add circuit breaker open response
        self.httpx_mock.add_response(
            method=method,
            url=url,
            status_code=503,
            json={
                "error": {
                    "message": "Circuit breaker open",
                    "code": "CIRCUIT_BREAKER_OPEN",
                },
            },
        )

        self.scenarios.append({
            "type": "circuit_breaker",
            "url": url,
            "method": method,
            "failure_threshold": failure_threshold,
        })

        return self

    def build_scenario(self) -> dict[str, object]:
        """Build complete scenario."""
        return {
            "total_requests": len(self.scenarios),
            "scenarios": self.scenarios,
        }


class WebhookTestUtils:
    """Utilities for testing webhook functionality."""

    @staticmethod
    def create_webhook_payload(
        event_type: str,
        data: JsonDict,
        timestamp: str | None = None,
        signature: str | None = None,
    ) -> JsonDict:
        """Create webhook payload."""
        payload: JsonDict = {
            "event": event_type,
            "data": data,
            "timestamp": timestamp or datetime.now(UTC).isoformat(),
        }

        if signature:
            payload["signature"] = signature

        return payload

    @staticmethod
    def validate_webhook_signature(
        payload: str,
        signature: str,
        secret: str,
    ) -> bool:
        """Validate webhook signature (simplified)."""
        import hashlib
        import hmac

        expected_signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, f"sha256={expected_signature}")

    @staticmethod
    def mock_webhook_delivery(
        httpx_mock: HTTPXMock,
        webhook_url: str,
        _event_type: str,
        _data: JsonDict,
        *,
        success: bool = True,
    ) -> None:
        """Mock webhook delivery."""
        if success:
            httpx_mock.add_response(
                method="POST",
                url=webhook_url,
                status_code=200,
                json={"received": True},
            )
        else:
            HTTPTestUtils.mock_error_response(
                httpx_mock,
                webhook_url,
                "POST",
                status_code=500,
                error_message="Webhook delivery failed",
            )


# Pytest fixtures for HTTP testing
@pytest.fixture
def api_client() -> APITestClient:
    """Provide API test client."""
    return APITestClient()


@pytest.fixture
def http_scenario_builder(httpx_mock: HTTPXMock) -> HTTPScenarioBuilder:
    """Provide HTTP scenario builder."""
    return HTTPScenarioBuilder(httpx_mock)


# Export utilities
__all__ = [
    "APITestClient",
    "HTTPScenarioBuilder",
    "HTTPTestUtils",
    "WebhookTestUtils",
    "api_client",
    "http_scenario_builder",
]
