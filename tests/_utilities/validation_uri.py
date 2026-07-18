"""URI validation scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests import m, p

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesValidationUriScenarios:
    """URI validation scenarios."""

    URI_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="uri_valid_http",
            validator_type="network",
            input_value="http://example.com",
            should_succeed=True,
            expected_value="http://example.com",
            description="Valid HTTP URI",
        ),
        m.Tests.ValidationScenario(
            name="uri_valid_https",
            validator_type="network",
            input_value="https://example.com",
            should_succeed=True,
            expected_value="https://example.com",
            description="Valid HTTPS URI",
        ),
        m.Tests.ValidationScenario(
            name="uri_with_port",
            validator_type="network",
            input_value="https://example.com:8080",
            should_succeed=True,
            expected_value="https://example.com:8080",
            description="URI with custom port",
        ),
        m.Tests.ValidationScenario(
            name="uri_with_path",
            validator_type="network",
            input_value="https://example.com/path/to/resource",
            should_succeed=True,
            expected_value="https://example.com/path/to/resource",
            description="URI with path",
        ),
        m.Tests.ValidationScenario(
            name="uri_with_query",
            validator_type="network",
            input_value="https://example.com/path?key=value",
            should_succeed=True,
            expected_value="https://example.com/path?key=value",
            description="URI with query parameters",
        ),
        m.Tests.ValidationScenario(
            name="uri_with_fragment",
            validator_type="network",
            input_value="https://example.com/path#section",
            should_succeed=True,
            expected_value="https://example.com/path#section",
            description="URI with fragment",
        ),
        m.Tests.ValidationScenario(
            name="uri_none",
            validator_type="network",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None URI rejection",
        ),
        m.Tests.ValidationScenario(
            name="uri_empty",
            validator_type="network",
            input_value="",
            should_succeed=False,
            expected_error_contains="cannot be empty",
            description="Empty URI rejection",
        ),
        m.Tests.ValidationScenario(
            name="uri_invalid_scheme",
            validator_type="network",
            input_value="ftp://example.com",
            should_succeed=False,
            expected_error_contains="not in allowed",
            description="Invalid URI scheme",
        ),
        m.Tests.ValidationScenario(
            name="uri_malformed",
            validator_type="network",
            input_value="not a valid uri",
            should_succeed=False,
            expected_error_contains="not a valid",
            description="Malformed URI",
        ),
        m.Tests.ValidationScenario(
            name="uri_whitespace",
            validator_type="network",
            input_value="   ",
            should_succeed=False,
            expected_error_contains="cannot be empty",
            description="Whitespace-only URI",
        ),
        m.Tests.ValidationScenario(
            name="uri_custom_scheme",
            validator_type="network",
            input_value="custom://example.com",
            input_params={"allowed_schemes": ["http", "https", "custom"]},
            should_succeed=True,
            expected_value="custom://example.com",
            description="Custom scheme with allowlist",
        ),
    ]


__all__: list[str] = ["TestsFlextUtilitiesValidationUriScenarios"]
