"""Helper classes for reducing complexity.

Implements Single Responsibility Principle to reduce repetitive patterns
and cognitive complexity in demonstration code.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .validation_utilities import is_email, is_int, is_non_empty_string

if TYPE_CHECKING:
    from flext_core import TErrorMessage, TUserData

# =============================================================================
# COMPLEXITY REDUCTION HELPERS - SOLID SRP: Modular utility demonstrations
# =============================================================================


class DemonstrationSectionHelper:
    """Helper to reduce repetitive demonstration patterns - SOLID SRP."""

    @staticmethod
    def print_section_header(example_num: int, title: str) -> None:
        """DRY Helper: Print standardized section headers."""
        "\n" + "=" * 60

    @staticmethod
    def log_operation(operation: str, result: object) -> None:
        """DRY Helper: Log operation results consistently."""

    @staticmethod
    def print_separator() -> None:
        """DRY Helper: Print visual separator."""

    @staticmethod
    def log_success(message: str) -> None:
        """DRY Helper: Log success messages consistently."""

    @staticmethod
    def log_error(message: str) -> None:
        """DRY Helper: Log error messages consistently."""


class ValidationHelper:
    """Helper to reduce repetitive validation patterns - SOLID SRP."""

    @staticmethod
    def validate_user_data(user_data: TUserData) -> list[TErrorMessage]:
        """DRY Helper: Validate user data with consistent rules."""
        validation_errors: list[TErrorMessage] = []

        if not is_non_empty_string(user_data.get("name")):
            validation_errors.append("Name is required")

        email_value = user_data.get("email", "")
        if not isinstance(email_value, str) or not is_email(email_value):
            validation_errors.append("Valid email is required")

        age_value = user_data.get("age")
        if not is_int(age_value):
            validation_errors.append("Age must be a number")
        elif isinstance(age_value, int) and age_value < 0:
            validation_errors.append("Age must be positive")

        return validation_errors

    @staticmethod
    def report_validation_result(validation_errors: list[TErrorMessage]) -> None:
        """DRY Helper: Report validation results consistently."""
        if validation_errors:
            pass

    @staticmethod
    def validate_config_data(config: dict[str, object]) -> list[TErrorMessage]:
        """DRY Helper: Validate configuration data."""
        validation_errors: list[TErrorMessage] = []

        required_keys = ["database_url", "api_key", "timeout"]

        for key in required_keys:
            if key not in config:
                validation_errors.append(f"Missing required config key: {key}")
            elif not is_non_empty_string(config[key]):
                validation_errors.append(f"Config key '{key}' must be non-empty string")

        return validation_errors


class FormattingHelper:
    """Helper for consistent data formatting - SOLID SRP."""

    @staticmethod
    def format_currency(amount: float, currency: str = "USD") -> str:
        """Format currency consistently."""
        return f"{currency} {amount:.2f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage consistently."""
        return f"{value:.1f}%"

    @staticmethod
    def format_file_size(bytes_count: int) -> str:
        """Format file size in human-readable format."""
        kb = 1024
        mb = kb * kb
        gb = mb * kb

        if bytes_count < kb:
            return f"{bytes_count} B"
        if bytes_count < mb:
            return f"{bytes_count / kb:.1f} KB"
        if bytes_count < gb:
            return f"{bytes_count / mb:.1f} MB"
        return f"{bytes_count / gb:.1f} GB"
