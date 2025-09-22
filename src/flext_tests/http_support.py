"""HTTP testing utilities using pytest-httpx and pytest-mock.

Provides HTTP testing patterns, API client testing,
webhook testing, and HTTP scenario building for robust testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from urllib.parse import urljoin

from pydantic import BaseModel
from pytest_httpx import HTTPXMock

from flext_core import FlextLogger, FlextResult, FlextTypes

logger = FlextLogger(__name__)


class FlextTestsHttp:
    """Unified HTTP testing utilities for FLEXT ecosystem.

    Consolidates all HTTP testing patterns, API client testing,
    webhook testing, and HTTP scenario building into a single unified class.
    """

    # === API Test Client ===

    class APITestClient:
        """API test client for HTTP testing."""

        def __init__(self, base_url: str = "https://api.example.com") -> None:
            """Initialize API test client."""
            self.base_url = base_url
            self.default_headers = {
                "content-type": "application/json",
                "user-agent": "flext-test-client/1.0",
            }

        def build_url(self, endpoint: str) -> str:
            """Build complete URL from endpoint.

            Returns:
                str: Complete URL built from base URL and endpoint

            """
            return urljoin(self.base_url, endpoint)

        def validate_response_structure(
            self,
            response_data: FlextTypes.Core.JsonObject,
            required_fields: FlextTypes.Core.StringList,
            _: FlextTypes.Core.StringList | None = None,
        ) -> FlextResult[None]:
            """Validate API response structure.

            Returns:
                FlextResult[None]: Success if structure is valid, failure with error message if invalid

            """
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
            """Validate error response format.

            Returns:
                FlextResult[None]: Success if error format is valid, failure with error message if invalid

            """
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

            if expected_error_code and error_data.get("code") != expected_error_code:
                return FlextResult[None].fail(
                    f"Expected error code {expected_error_code}, got {error_data.get('code')}",
                    error_code="UNEXPECTED_ERROR_CODE",
                )

            return FlextResult[None].ok(None)

        def create_test_request(
            self,
            method: str,
            endpoint: str,
            data: object | None = None,
            headers: FlextTypes.Core.Headers | None = None,
        ) -> FlextTypes.Core.Dict:
            """Create test request data using ``FlextTypes.Core.Dict``.

            Returns:
                FlextTypes.Core.Dict: Test request payload expressed via the official alias

            """
            request_headers: FlextTypes.Core.Headers = self.default_headers.copy()
            if headers:
                request_headers.update(headers)

            request_data: FlextTypes.Core.Dict = {
                "method": method.upper(),
                "url": self.build_url(endpoint),
                "headers": request_headers,
            }

            if data is not None:
                if isinstance(data, (dict, list)):
                    request_data["json"] = data
                else:
                    request_data["data"] = str(data)

            return request_data

    # === HTTP Scenario Builder ===

    class HTTPScenarioBuilder:
        """Builder for complex HTTP testing scenarios."""

        def __init__(self, httpx_mock: HTTPXMock) -> None:
            """Initialize HTTP scenario builder."""
            self.httpx_mock = httpx_mock
            self.scenarios: list[FlextTypes.Core.Dict] = []

        def add_successful_request(
            self,
            url: str,
            method: str = "GET",
            response_data: object | None = None,
            status_code: int = 200,
            headers: FlextTypes.Core.Headers | None = None,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add successful request scenario.

            Returns:
                FlextTestsHttp.HTTPScenarioBuilder: Self for method chaining

            """
            response_headers: FlextTypes.Core.Headers = {
                "content-type": "application/json"
            }
            if headers:
                response_headers.update(headers)

            response_json = response_data or {"status": "success", "data": "test_data"}

            self.httpx_mock.add_response(
                method=method,
                url=url,
                json=response_json,
                status_code=status_code,
                headers=response_headers,
            )

            self.scenarios.append(
                {
                    "type": "success",
                    "url": url,
                    "method": method,
                    "status_code": status_code,
                    "response_data": response_json,
                },
            )

            return self

        def add_error_request(
            self,
            url: str,
            method: str = "GET",
            error_code: str = "INTERNAL_ERROR",
            status_code: int = 500,
            error_message: str = "Internal server error",
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add error request scenario.

            Returns:
                FlextTestsHttp.HTTPScenarioBuilder: Self for method chaining

            """
            error_response = {
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "timestamp": "2025-01-01T00:00:00Z",
                },
            }

            self.httpx_mock.add_response(
                method=method,
                url=url,
                json=error_response,
                status_code=status_code,
                headers={"content-type": "application/json"},
            )

            self.scenarios.append(
                {
                    "type": "error",
                    "url": url,
                    "method": method,
                    "status_code": status_code,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )

            return self

        def add_retry_scenario(
            self,
            url: str,
            method: str = "GET",
            max_retries: int = 3,
            success_after_retries: int = 2,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add retry scenario with failures followed by success.

            Returns:
                FlextTestsHttp.HTTPScenarioBuilder: Self for method chaining

            """
            # Add failures
            for _ in range(success_after_retries):
                self.httpx_mock.add_response(
                    method=method,
                    url=url,
                    json={
                        "error": {"code": "TEMPORARY_ERROR", "message": "Retry needed"},
                    },
                    status_code=503,
                    headers={"content-type": "application/json"},
                )

            # Add success
            self.httpx_mock.add_response(
                method=method,
                url=url,
                json={"status": "success", "retry_count": success_after_retries},
                status_code=200,
                headers={"content-type": "application/json"},
            )

            self.scenarios.append(
                {
                    "type": "retry",
                    "url": url,
                    "method": method,
                    "max_retries": max_retries,
                    "success_after_retries": success_after_retries,
                },
            )

            return self

        def add_circuit_breaker_scenario(
            self,
            url: str,
            method: str = "GET",
            failure_threshold: int = 5,
            recovery_timeout: int = 60,
        ) -> FlextTestsHttp.HTTPScenarioBuilder:
            """Add circuit breaker scenario.

            Returns:
                FlextTestsHttp.HTTPScenarioBuilder: Self for method chaining

            """
            # Add failures to trigger circuit breaker
            for _ in range(failure_threshold):
                self.httpx_mock.add_response(
                    method=method,
                    url=url,
                    json={"error": {"code": "CIRCUIT_BREAKER_TRIGGERED"}},
                    status_code=503,
                    headers={"content-type": "application/json"},
                )

            self.scenarios.append(
                {
                    "type": "circuit_breaker",
                    "url": url,
                    "method": method,
                    "failure_threshold": failure_threshold,
                    "recovery_timeout": recovery_timeout,
                },
            )

            return self

        def build_scenario(self) -> FlextTypes.Core.Dict:
            """Build test scenario.

            Returns:
                FlextTypes.Core.Dict: Test scenario configuration dictionary

            """
            return {
                "scenarios": self.scenarios,
                "total_scenarios": len(self.scenarios),
                "scenario_types": list({s["type"] for s in self.scenarios}),
            }

    # === Webhook Testing Utilities ===

    class WebhookTestUtils:
        """Utilities for testing webhook functionality."""

        @staticmethod
        def create_webhook_payload(
            event_type: str,
            data: FlextTypes.Core.Dict,
            webhook_id: str | None = None,
        ) -> FlextTypes.Core.Dict:
            """Create webhook payload.

            Returns:
                FlextTypes.Core.Dict: Webhook payload dictionary

            """
            return {
                "id": webhook_id or "test_webhook_123",
                "event": event_type,
                "data": data,
                "timestamp": "2025-01-01T00:00:00Z",
                "version": "1.0",
            }

        @staticmethod
        def validate_webhook_signature(
            payload: str,
            signature: str,
            secret: str,
        ) -> FlextResult[bool]:
            """Validate webhook signature.

            Returns:
                FlextResult[bool]: Success with validation result if valid, failure with error if invalid

            """
            # Simple validation for testing
            expected_signature = f"sha256={hash(payload + secret)}"
            is_valid = signature == expected_signature

            if is_valid:
                return FlextResult[bool].ok(True)
            return FlextResult[bool].fail(
                "Invalid webhook signature",
                error_code="INVALID_SIGNATURE",
            )

        @staticmethod
        def create_webhook_response(
            status: str = "received",
            message: str = "Webhook processed successfully",
        ) -> FlextTypes.Core.Dict:
            """Create webhook response.

            Returns:
                FlextTypes.Core.Dict: Webhook response dictionary

            """
            return {
                "status": status,
                "message": message,
                "processed_at": "2025-01-01T00:00:00Z",
            }

    # === HTTP Test Models ===

    class HTTPTestRequest(BaseModel):
        """HTTP test request model."""

        method: str
        url: str
        headers: FlextTypes.Core.Dict | None = None
        data: object | None = None
        timeout: float = 30.0

    class HTTPTestResponse(BaseModel):
        """HTTP test response model."""

        status_code: int
        headers: FlextTypes.Core.Dict
        data: object | None = None
        json_data: FlextTypes.Core.JsonObject | None = None

    # === Convenience Factory Methods ===

    @staticmethod
    def create_api_client(
        base_url: str = "https://api.example.com",
    ) -> FlextTestsHttp.APITestClient:
        """Create API test client.

        Returns:
            FlextTestsHttp.APITestClient: API test client instance

        """
        return FlextTestsHttp.APITestClient(base_url)

    @staticmethod
    def create_scenario_builder(
        httpx_mock: HTTPXMock,
    ) -> FlextTestsHttp.HTTPScenarioBuilder:
        """Create HTTP scenario builder.

        Returns:
            FlextTestsHttp.HTTPScenarioBuilder: HTTP scenario builder instance

        """
        return FlextTestsHttp.HTTPScenarioBuilder(httpx_mock)

    @staticmethod
    def create_test_request(
        method: str,
        url: str,
        data: object | None = None,
        headers: FlextTypes.Core.Dict | None = None,
    ) -> FlextTestsHttp.HTTPTestRequest:
        """Create test request.

        Returns:
            FlextTestsHttp.HTTPTestRequest: HTTP test request instance

        """
        return FlextTestsHttp.HTTPTestRequest(
            method=method,
            url=url,
            data=data,
            headers=headers,
        )

    @staticmethod
    def create_test_response(
        status_code: int,
        headers: FlextTypes.Core.Dict | None = None,
        data: object | None = None,
        json_data: FlextTypes.Core.JsonObject | None = None,
    ) -> FlextTestsHttp.HTTPTestResponse:
        """Create test response.

        Returns:
            FlextTestsHttp.HTTPTestResponse: HTTP test response instance

        """
        return FlextTestsHttp.HTTPTestResponse(
            status_code=status_code,
            headers=headers or {},
            data=data,
            json_data=json_data,
        )

    @staticmethod
    def mock_httpx_response(
        httpx_mock: HTTPXMock,
        url: str,
        *,
        method: str = "GET",
        json_data: object | None = None,
        status_code: int = 200,
        headers: FlextTypes.Core.Dict | None = None,
    ) -> None:
        """Mock HTTPX response."""
        response_headers: FlextTypes.Core.Headers = {
            "content-type": "application/json"
        }
        if headers:
            # Convert headers dict values to strings
            for key, value in headers.items():
                response_headers[key] = (
                    str(value) if not isinstance(value, str) else value
                )

        httpx_mock.add_response(
            method=method,
            url=url,
            json=json_data,
            status_code=status_code,
            headers=response_headers,
        )


# Export only the unified class
__all__ = [
    "FlextTestsHttp",
]
