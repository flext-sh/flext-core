"""Oracle-specific configuration and validation patterns.

This module provides base classes and common patterns for Oracle database
integration across all FLEXT projects. It eliminates duplication of Oracle
validation logic and provides consistent patterns.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any

from flext_core.domain.types import ServiceResult

logger = logging.getLogger(__name__)


class OracleValidationError(Exception):
    """Oracle validation error."""


class OracleCriticalValidationError(Exception):
    """Critical Oracle validation error that requires immediate abort."""


class BaseOracleValidator(ABC):
    """Base class for Oracle validation patterns.

    Provides common validation patterns used across Oracle projects:
    - Environment variable validation
    - Data format validation
    - Business rule validation
    - WMS-specific patterns
    """

    def __init__(self, project_prefix: str) -> None:
        """Initialize Oracle validator.

        Args:
            project_prefix: Environment variable prefix (e.g., "TAP_ORACLE_WMS")

        """
        self.project_prefix = project_prefix
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def get_critical_env_vars(self) -> dict[str, Any]:
        """Get critical environment variables for validation.

        Returns:
            Dict mapping env var names to validation requirements

        """

    def enforce_critical_environment_variables(self) -> ServiceResult[None]:
        """Enforce critical environment variables.

        Returns:
            ServiceResult indicating success or failure with error details

        """
        try:
            critical_vars = self.get_critical_env_vars()
            errors = []

            for var_name, requirements in critical_vars.items():
                env_var_name = f"{self.project_prefix}_{var_name}"
                value = os.getenv(env_var_name, "").lower()

                # Check required value
                if "required_value" in requirements:
                    required = str(requirements["required_value"]).lower()
                    if value != required:
                        error_msg = (
                            f"❌ CRITICAL FAILURE: {env_var_name} must be '{required}' "
                            f"but got '{value}'. This is a NON-NEGOTIABLE requirement."
                        )
                        errors.append(error_msg)
                        self.logger.error(error_msg)

                # Check required format
                if "pattern" in requirements:
                    pattern = requirements["pattern"]
                    if value and not re.match(pattern, value):
                        error_msg = (
                            f"❌ CRITICAL FAILURE: {env_var_name} value '{value}' "
                            f"does not match required pattern '{pattern}'"
                        )
                        errors.append(error_msg)
                        self.logger.error(error_msg)

                # Check numeric constraints
                if "numeric_constraints" in requirements:
                    try:
                        numeric_value = int(value) if value else -1
                        constraints = requirements["numeric_constraints"]

                        if (
                            "exact" in constraints
                            and numeric_value != constraints["exact"]
                        ):
                            error_msg = (
                                f"❌ CRITICAL FAILURE: {env_var_name} must be exactly "
                                f"'{constraints['exact']}' but got '{numeric_value}'"
                            )
                            errors.append(error_msg)
                            self.logger.error(error_msg)

                    except (ValueError, TypeError):
                        if requirements.get("required", False):
                            error_msg = f"❌ CRITICAL FAILURE: {env_var_name} must be a valid number"
                            errors.append(error_msg)
                            self.logger.exception(error_msg)

            if errors:
                return ServiceResult.fail("; ".join(errors))

            self.logger.info(
                "🚨 CRITICAL VALIDATION PASSED: All mandatory environment variables validated",
            )
            return ServiceResult.ok(None)

        except Exception as e:
            error_msg = f"Environment validation failed: {e}"
            self.logger.exception(error_msg)
            return ServiceResult.fail(error_msg)

    def validate_oracle_connection_config(
        self,
        config: dict[str, Any],
    ) -> ServiceResult[None]:
        """Validate Oracle connection configuration.

        Args:
            config: Oracle connection configuration

        Returns:
            ServiceResult indicating validation success or failure

        """
        try:
            required_fields = ["host", "port", "user", "password"]

            missing_fields = [
                field
                for field in required_fields
                if field not in config or not config[field]
            ]

            if missing_fields:
                return ServiceResult.fail(
                    f"Missing required Oracle connection fields: {missing_fields}",
                )

            # Validate port range
            port = config.get("port", 1521)
            if not isinstance(port, int) or port < 1 or port > 65535:
                return ServiceResult.fail(f"Invalid Oracle port: {port}")

            # Validate service_name or sid is provided
            service_name = config.get("service_name")
            sid = config.get("sid")
            if not service_name and not sid:
                return ServiceResult.fail("Either service_name or sid must be provided")

            return ServiceResult.ok(None)

        except (KeyError, ValueError, TypeError) as e:
            return ServiceResult.fail(f"Oracle connection validation failed: {e}")


class OracleWMSValidator(BaseOracleValidator):
    """Oracle WMS-specific validation patterns.

    Provides WMS-specific validation rules common across WMS projects.
    """

    def __init__(self, project_prefix: str = "TAP_ORACLE_WMS") -> None:
        """Initialize Oracle WMS validator."""
        super().__init__(project_prefix)

        # WMS specific field constraints
        self.wms_field_constraints: dict[str, dict[str, Any]] = {
            "company_code": {"max_length": 25, "pattern": r"^[A-Z0-9_]+$"},
            "facility_code": {"max_length": 30, "pattern": r"^[A-Z0-9_]+$"},
            "item_code": {"max_length": 50, "pattern": r"^[A-Z0-9_\-\.]+$"},
            "location_barcode": {"max_length": 30, "pattern": r"^[A-Z0-9]+$"},
            "order_nbr": {"max_length": 50},
            "batch_nbr": {"max_length": 50},
            "serial_nbr": {"max_length": 50},
            "lpn": {"max_length": 30, "pattern": r"^[A-Z0-9]+$"},
        }

        # Date/time fields for WMS
        self.datetime_fields = [
            "create_ts",
            "mod_ts",
            "last_upd_ts",
            "ship_date",
            "receipt_date",
            "expiry_date",
            "manufactured_date",
        ]

        # Numeric precision requirements for WMS
        self.decimal_fields = {
            "quantity": {"precision": 15, "scale": 4},
            "weight": {"precision": 15, "scale": 4},
            "volume": {"precision": 15, "scale": 4},
            "price": {"precision": 15, "scale": 2},
            "cost": {"precision": 15, "scale": 2},
        }

    def get_critical_env_vars(self) -> dict[str, Any]:
        """Get WMS-specific critical environment variables."""
        return {
            "USE_METADATA_ONLY": {
                "required_value": "true",
                "description": "Force metadata-only discovery for WMS",
            },
            "DISCOVERY_SAMPLE_SIZE": {
                "numeric_constraints": {"exact": 0},
                "description": "Disable sample-based discovery for WMS",
            },
        }

    def validate_wms_record(self, record: dict[str, Any]) -> ServiceResult[list[str]]:
        """Validate a WMS record against business rules.

        Args:
            record: WMS record to validate

        Returns:
            ServiceResult containing list of validation errors (empty if valid)

        """
        try:
            errors = []

            # Validate WMS field constraints
            wms_errors = self._validate_wms_constraints(record)
            errors.extend(wms_errors)

            # Validate data types
            type_errors = self._validate_wms_data_types(record)
            errors.extend(type_errors)

            # Validate WMS business logic
            logic_errors = self._validate_wms_business_logic(record)
            errors.extend(logic_errors)

            return ServiceResult.ok(errors)

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return ServiceResult.fail(f"WMS record validation failed: {e}")

    def _validate_wms_constraints(self, record: dict[str, Any]) -> list[str]:
        """Validate WMS-specific field constraints."""
        errors = []

        for field_name, constraints in self.wms_field_constraints.items():
            if field_name in record:
                value = record[field_name]

                # Check max length
                if (
                    "max_length" in constraints
                    and isinstance(value, str)
                    and len(value) > constraints["max_length"]
                ):
                    errors.append(
                        f"Field '{field_name}' exceeds maximum length "
                        f"{constraints['max_length']}: {len(value)}",
                    )

                # Check pattern
                if (
                    "pattern" in constraints
                    and isinstance(value, str)
                    and not re.match(constraints["pattern"], value)
                ):
                    errors.append(
                        "Field '{}' does not match required pattern '{}': {}".format(
                            field_name,
                            constraints["pattern"],
                            value,
                        ),
                    )

        return errors

    def _validate_wms_data_types(self, record: dict[str, Any]) -> list[str]:
        """Validate WMS data types."""
        errors = []

        # Validate datetime fields
        for field_name in self.datetime_fields:
            if field_name in record:
                value = record[field_name]
                if value is not None:
                    try:
                        if isinstance(value, str):
                            datetime.fromisoformat(value)
                    except ValueError:
                        errors.append(
                            f"Invalid datetime format for '{field_name}': {value}",
                        )

        # Validate decimal precision
        for field_name, precision_info in self.decimal_fields.items():
            if field_name in record:
                value = record[field_name]
                if value is not None and isinstance(value, int | float):
                    # Convert to string to check precision
                    str_value = str(float(value))
                    if "." in str_value:
                        integer_part, decimal_part = str_value.split(".")
                        total_digits = len(integer_part) + len(decimal_part)

                        if total_digits > precision_info["precision"]:
                            errors.append(
                                f"Field '{field_name}' exceeds precision "
                                f"{precision_info['precision']}: {value}",
                            )

                        if len(decimal_part) > precision_info["scale"]:
                            errors.append(
                                f"Field '{field_name}' exceeds scale "
                                f"{precision_info['scale']}: {value}",
                            )

        return errors

    def _validate_wms_business_logic(self, record: dict[str, Any]) -> list[str]:
        """Validate WMS business logic rules."""
        errors = []

        # Quantity validations
        if "quantity" in record:
            quantity = record["quantity"]
            if (
                quantity is not None
                and isinstance(quantity, int | float)
                and quantity < 0
            ):
                errors.append(f"Quantity cannot be negative: {quantity}")

        # Date validations
        if "ship_date" in record and "receipt_date" in record:
            ship_date = record["ship_date"]
            receipt_date = record["receipt_date"]

            if ship_date and receipt_date:
                try:
                    ship_dt = datetime.fromisoformat(ship_date)
                    receipt_dt = datetime.fromisoformat(receipt_date)

                    if receipt_dt > ship_dt:
                        errors.append("Receipt date cannot be after ship date")

                except ValueError:
                    pass  # Date format errors handled elsewhere

        return errors


class OracleTapValidator(BaseOracleValidator):
    """Oracle Tap-specific validation patterns."""

    def get_critical_env_vars(self) -> dict[str, Any]:
        """Get Tap-specific critical environment variables."""
        return {
            "USE_METADATA_ONLY": {
                "required_value": "true",
                "description": "Force metadata-only discovery",
            },
            "DISCOVERY_SAMPLE_SIZE": {
                "numeric_constraints": {"exact": 0},
                "description": "Disable sample-based discovery",
            },
        }


class OracleTargetValidator(BaseOracleValidator):
    """Oracle Target-specific validation patterns."""

    def get_critical_env_vars(self) -> dict[str, Any]:
        """Get Target-specific critical environment variables."""
        return {
            "BATCH_SIZE": {
                "numeric_constraints": {"min": 1, "max": 10000},
                "description": "Valid batch size range",
            },
            "ENABLE_BULK_LOADING": {
                "required_value": "true",
                "description": "Enable bulk loading for performance",
            },
        }
