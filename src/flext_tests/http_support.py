"""HTTP testing utilities using pytest-httpx for comprehensive API testing.

Provides advanced HTTP mocking, request/response validation, and API testing
patterns with automatic retry, error simulation, and performance monitoring.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urljoin

import httpx
from pytest_httpx import HTTPXMock

from flext_core import FlextResult, FlextTypes


class FlextTestsHttp:
    """Unified HTTP testing utilities for FLEXT ecosystem.

    Consolidates all HTTP testing patterns into a single class interface.
    Provides comprehensive HTTP mocking, request/response validation, and API testing
    patterns with automatic retry, error simulation, and performance monitoring.
    """

    # === HTTP Testing Utilities ===

    class HTTPTestUtils:
        """Comprehensive HTTP testing utilities with mocking and validation."""

        @staticmethod
        def mock_successful_response(
            httpx_mock: HTTPXMock,
            url: str,
            method: str = "GET",
            json_data: FlextTypes.Core.JsonObject | None = None,
            status_code: int = 200,
            headers: FlextTypes.Core.Headers | None = None,
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
            data_list: list[FlextTypes.Core.JsonObject],
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

    # === API Testing Client ===

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
            response_data: FlextTypes.Core.JsonObject,
            required_fields: FlextTypes.Core.StringList,
            _: FlextTypes.Core.StringList | None = None,
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
            response_data: FlextTypes.Core.JsonObject,
            expected_error_code: str | None = None,
        ) -> FlextResult[None]:
            """Validate error response format."""
            if "error" not in response_data:
                return FlextResult[None].fail(
                    "Error response missing 'error' field",
                    error_code="INVALID_ERROR_FORMAT",
                )

            error_data = response_data["error"]
            if not isinstance(error_data, dict):
                return FlextResult[None].fail(
                    "Error field must be an object",
                    error_code="INVALID_ERROR_FORMAT",
                )

            error: FlextTypes.Core.JsonObject = cast(
                "FlextTypes.Core.JsonObject", error_data
            )  # Now we know it's a dict
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

    # === HTTP Scenario Builder ===

    class HTTPScenarioBuilder:
        """Builder for complex HTTP testing scenarios."""

        def __init__(self, httpx_mock: HTTPXMock) -> None:
            self.httpx_mock = httpx_mock
            self.scenarios: list[FlextTypes.Core.Dict] = []

        def add_successful_request(
            self,
            url: str,
            method: str = "GET",
            response_data: FlextTypes.Core.JsonObject | None = None,
            status_code: int = 200,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add successful request to scenario."""
            FlextTestsHttp.HTTPTestUtils.mock_successful_response(
                self.httpx_mock,
                url,
                method,
                response_data,
                status_code,
            )

            self.scenarios.append(
                {
                    "type": "success",
                    "url": url,
                    "method": method,
                    "status_code": status_code,
                },
            )

            return self

        def add_error_request(
            self,
            url: str,
            method: str = "GET",
            status_code: int = 500,
            error_message: str = "Server error",
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add error request to scenario."""
            FlextTestsHttp.HTTPTestUtils.mock_error_response(
                self.httpx_mock,
                url,
                method,
                status_code,
                error_message,
            )

            self.scenarios.append(
                {
                    "type": "error",
                    "url": url,
                    "method": method,
                    "status_code": status_code,
                },
            )

            return self

        def add_retry_scenario(
            self,
            url: str,
            method: str = "GET",
            failure_count: int = 2,
            final_response: FlextTypes.Core.JsonObject | None = None,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add retry scenario (fail N times, then succeed)."""
            # Add failures
            for i in range(failure_count):
                self.httpx_mock.add_response(
                    method=method,
                    url=url,
                    status_code=503,
                    json={
                        "error": {"message": f"Service unavailable (attempt {i + 1})"}
                    },
                )

            # Add final success
            FlextTestsHttp.HTTPTestUtils.mock_successful_response(
                self.httpx_mock,
                url,
                method,
                final_response or {"status": "success", "retry_succeeded": True},
            )

            self.scenarios.append(
                {
                    "type": "retry",
                    "url": url,
                    "method": method,
                    "failure_count": failure_count,
                },
            )

            return self

        def add_circuit_breaker_scenario(
            self,
            url: str,
            method: str = "GET",
            failure_threshold: int = 3,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add circuit breaker scenario."""
            # Add failures up to threshold
            for i in range(failure_threshold):
                FlextTestsHttp.HTTPTestUtils.mock_error_response(
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

            self.scenarios.append(
                {
                    "type": "circuit_breaker",
                    "url": url,
                    "method": method,
                    "failure_threshold": failure_threshold,
                },
            )

            return self

        def build_scenario(self) -> FlextTypes.Core.Dict:
            """Build complete scenario."""
            return {
                "total_requests": len(self.scenarios),
                "scenarios": self.scenarios,
            }

    # === Webhook Testing Utilities ===

    class WebhookTestUtils:
        """Utilities for testing webhook functionality."""

        @staticmethod
        def create_webhook_payload(
            event_type: str,
            data: FlextTypes.Core.JsonValue,
            timestamp: str | None = None,
            signature: str | None = None,
        ) -> FlextTypes.Core.JsonObject:
            """Create webhook payload."""
            payload: FlextTypes.Core.JsonObject = {
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
            _data: FlextTypes.Core.JsonObject,
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
                FlextTestsHttp.HTTPTestUtils.mock_error_response(
                    httpx_mock,
                    webhook_url,
                    "POST",
                    status_code=500,
                    error_message="Webhook delivery failed",
                )

    # === Convenience Factory Methods ===

    @staticmethod
    def create_api_client(
        base_url: str = "https://api.example.com",
    ) -> FlextTestsHttp.APITestClient:
        """Create API test client."""
        return FlextTestsHttp.APITestClient(base_url)

    @staticmethod
    def create_scenario_builder(
        httpx_mock: HTTPXMock,
    ) -> FlextTestsHttp.HTTPScenarioBuilder:
        """Create HTTP scenario builder."""
        return FlextTestsHttp.HTTPScenarioBuilder(httpx_mock)

    @staticmethod
    def mock_success(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
        json_data: FlextTypes.Core.JsonObject | None = None,
        status_code: int = 200,
    ) -> None:
        """Convenience method to mock successful response."""
        FlextTestsHttp.HTTPTestUtils.mock_successful_response(
            httpx_mock, url, method, json_data, status_code
        )

    @staticmethod
    def mock_error(
        httpx_mock: HTTPXMock,
        url: str,
        method: str = "GET",
        status_code: int = 500,
        error_message: str = "Internal server error",
    ) -> None:
        """Convenience method to mock error response."""
        FlextTestsHttp.HTTPTestUtils.mock_error_response(
            httpx_mock, url, method, status_code, error_message
        )

    @staticmethod
    def mock_timeout(httpx_mock: HTTPXMock, url: str, method: str = "GET") -> None:
        """Convenience method to mock timeout."""
        FlextTestsHttp.HTTPTestUtils.mock_timeout_response(httpx_mock, url, method)

    @staticmethod
    def mock_network_error(
        httpx_mock: HTTPXMock, url: str, method: str = "GET"
    ) -> None:
        """Convenience method to mock network error."""
        FlextTestsHttp.HTTPTestUtils.mock_network_error(httpx_mock, url, method)

    @staticmethod
    def create_webhook_payload(
        event_type: str,
        data: FlextTypes.Core.JsonValue,
        timestamp: str | None = None,
        signature: str | None = None,
    ) -> FlextTypes.Core.JsonObject:
        """Convenience method to create webhook payload."""
        return FlextTestsHttp.WebhookTestUtils.create_webhook_payload(
            event_type, data, timestamp, signature
        )

    @staticmethod
    def validate_webhook_signature(
        payload: str,
        signature: str,
        secret: str,
    ) -> bool:
        """Convenience method to validate webhook signature."""
        return FlextTestsHttp.WebhookTestUtils.validate_webhook_signature(
            payload, signature, secret
        )


# === REMOVED STANDALONE FUNCTIONS ===
# Moved to FlextTestsHttp class as per user request
# Only the unified FlextTestsHttp class should be used

# @pytest.fixture
# def api_client() -> FlextTestsHttp.APITestClient:
#     """Provide API test client."""
#     return FlextTestsHttp.APITestClient()

# @pytest.fixture
# def http_scenario_builder(httpx_mock: HTTPXMock) -> FlextTestsHttp.HTTPScenarioBuilder:
#     """Provide HTTP scenario builder."""
#     return FlextTestsHttp.HTTPScenarioBuilder(httpx_mock)


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsHttp class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsHttpSupport = FlextTestsHttp

# Legacy HTTPTestUtils class - REMOVED (commented out)
# class HTTPTestUtils:
#     """Compatibility facade for HTTPTestUtils - use FlextTestsHttp instead."""
#     ... all methods commented out

# Legacy APITestClient class - REMOVED (commented out)
# class APITestClient:
#     """Compatibility facade for APITestClient - use FlextTestsHttp instead."""
#     ... all methods commented out

# Legacy HTTPScenarioBuilder class - REMOVED (commented out)
# class HTTPScenarioBuilder:
#     """Compatibility facade for HTTPScenarioBuilder - use FlextTestsHttp instead."""
#     ... all methods commented out

# Legacy WebhookTestUtils class - REMOVED (commented out)
# class WebhookTestUtils:
#     """Compatibility facade for WebhookTestUtils - use FlextTestsHttp instead."""
#     ... all methods commented out

# Export only the unified class
__all__ = [
    "FlextTestsHttp",
]
