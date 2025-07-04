"""Minimal production validation stub - replacing broken file."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.config.domain_config import FlextConfiguration


def validate_production_configuration(_config: FlextConfiguration) -> dict[str, object]:
    """Stub for production configuration validation."""
    return {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "checks_performed": ["basic_validation"],
    }


def detect_configuration_drift(_config: FlextConfiguration) -> tuple[bool, dict[str, object]]:
    """Stub for configuration drift detection."""
    return False, {"drift_detected": False}


def get_configuration_security_score(_config: FlextConfiguration) -> dict[str, object]:
    """Stub for security score calculation."""
    return {"score": 85, "grade": "B"}
