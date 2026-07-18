"""Network validation scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests import m, p

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestsFlextUtilitiesValidationNetworkScenarios:
    """Network validation scenarios."""

    PORT_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="port_valid_80",
            validator_type="network",
            input_value=80,
            should_succeed=True,
            expected_value=80,
            description="Valid port 80 (HTTP)",
        ),
        m.Tests.ValidationScenario(
            name="port_valid_443",
            validator_type="network",
            input_value=443,
            should_succeed=True,
            expected_value=443,
            description="Valid port 443 (HTTPS)",
        ),
        m.Tests.ValidationScenario(
            name="port_valid_8080",
            validator_type="network",
            input_value=8080,
            should_succeed=True,
            expected_value=8080,
            description="Valid port 8080",
        ),
        m.Tests.ValidationScenario(
            name="port_valid_1",
            validator_type="network",
            input_value=1,
            should_succeed=True,
            expected_value=1,
            description="Valid port 1 (minimum)",
        ),
        m.Tests.ValidationScenario(
            name="port_valid_65535",
            validator_type="network",
            input_value=65535,
            should_succeed=True,
            expected_value=65535,
            description="Valid port 65535 (maximum)",
        ),
        m.Tests.ValidationScenario(
            name="port_none",
            validator_type="network",
            input_value=None,
            should_succeed=False,
            expected_error_contains="cannot be None",
            description="None port rejection",
        ),
        m.Tests.ValidationScenario(
            name="port_zero",
            validator_type="network",
            input_value=0,
            should_succeed=False,
            expected_error_contains="must be between",
            description="Port zero rejection",
        ),
        m.Tests.ValidationScenario(
            name="port_negative",
            validator_type="network",
            input_value=-1,
            should_succeed=False,
            expected_error_contains="must be between",
            description="Negative port rejection",
        ),
        m.Tests.ValidationScenario(
            name="port_above_max",
            validator_type="network",
            input_value=65536,
            should_succeed=False,
            expected_error_contains="at most",
            description="Port above maximum",
        ),
    ]
    HOSTNAME_SCENARIOS: ClassVar[Sequence[p.Tests.ValidationScenario]] = [
        m.Tests.ValidationScenario(
            name="hostname_simple",
            validator_type="network",
            input_value="example",
            should_succeed=True,
            expected_value="example",
            description="Simple hostname",
        ),
        m.Tests.ValidationScenario(
            name="hostname_fqdn",
            validator_type="network",
            input_value="example.com",
            should_succeed=True,
            expected_value="example.com",
            description="Fully qualified domain name",
        ),
        m.Tests.ValidationScenario(
            name="hostname_subdomain",
            validator_type="network",
            input_value="sub.example.com",
            should_succeed=True,
            expected_value="sub.example.com",
            description="Hostname with subdomain",
        ),
        m.Tests.ValidationScenario(
            name="hostname_hyphen",
            validator_type="network",
            input_value="my-host.example.com",
            should_succeed=True,
            expected_value="my-host.example.com",
            description="Hostname with hyphen",
        ),
    ]


__all__: list[str] = ["TestsFlextUtilitiesValidationNetworkScenarios"]
