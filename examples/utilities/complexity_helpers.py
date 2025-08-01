"""Complexity Reduction Helpers Following SOLID Principles.

Helper classes that implement Single Responsibility Principle (SRP) to reduce
repetitive patterns and cognitive complexity in demonstration code.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import TErrorMessage, TLogMessage, TUserData

from .validation_utilities import is_email, is_int, is_non_empty_string

# =============================================================================
# COMPLEXITY REDUCTION HELPERS - SOLID SRP: Modular utility demonstrations
# =============================================================================


class DemonstrationSectionHelper:
    """Helper to reduce repetitive demonstration patterns - SOLID SRP."""

    @staticmethod
    def print_section_header(example_num: int, title: str) -> None:
        """DRY Helper: Print standardized section headers."""
        log_message: TLogMessage = "\n" + "=" * 60
        print(log_message)
        print(f"📋 EXAMPLE {example_num}: {title}")
        print("=" * 60)

    @staticmethod
    def log_operation(operation: str, result: object) -> None:
        """DRY Helper: Log operation results consistently."""
        log_message: TLogMessage = f"🔧 {operation}: {result}"
        print(log_message)

    @staticmethod
    def print_separator() -> None:
        """DRY Helper: Print visual separator."""
        print("-" * 40)

    @staticmethod
    def log_success(message: str) -> None:
        """DRY Helper: Log success messages consistently."""
        log_message: TLogMessage = f"✅ {message}"
        print(log_message)

    @staticmethod
    def log_error(message: str) -> None:
        """DRY Helper: Log error messages consistently."""
        log_message: TLogMessage = f"❌ {message}"
        print(log_message)


class ValidationHelper:
    """Helper to reduce repetitive validation patterns - SOLID SRP."""

    @staticmethod
    def validate_user_data(user_data: TUserData) -> list[TErrorMessage]:
        """DRY Helper: Validate user data with consistent rules."""
        validation_errors: list[TErrorMessage] = []

        if not is_non_empty_string(user_data.get("name")):
            validation_errors.append("Name is required")

        if not is_email(user_data.get("email", "")):
            validation_errors.append("Valid email is required")

        if not is_int(user_data.get("age")):
            validation_errors.append("Age must be a number")
        elif user_data.get("age", 0) < 0:
            validation_errors.append("Age must be positive")

        return validation_errors

    @staticmethod
    def report_validation_result(validation_errors: list[TErrorMessage]) -> None:
        """DRY Helper: Report validation results consistently."""
        if validation_errors:
            log_message: TLogMessage = f"❌ Validation failed: {validation_errors}"
            print(log_message)
        else:
            log_message = "✅ Data validation passed"
            print(log_message)

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
        if bytes_count < 1024:
            return f"{bytes_count} B"
        elif bytes_count < 1024 * 1024:
            return f"{bytes_count / 1024:.1f} KB"
        elif bytes_count < 1024 * 1024 * 1024:
            return f"{bytes_count / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_count / (1024 * 1024 * 1024):.1f} GB"
