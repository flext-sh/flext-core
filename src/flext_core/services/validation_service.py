"""Validation service for FLEXT Core."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class ValidationLevel(Enum):
    """Validation levels."""

    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class ValidationResult:
    """Validation result."""

    def __init__(
        self,
        is_valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)


class ValidationService(ABC):
    """Abstract validation service."""

    @abstractmethod
    async def validate_pipeline_config(
        self, config: dict[str, Any], level: ValidationLevel = ValidationLevel.BASIC
    ) -> ValidationResult:
        """Validate pipeline configuration."""

    @abstractmethod
    async def validate_plugin_config(
        self,
        config: dict[str, Any],
        plugin_name: str,
        level: ValidationLevel = ValidationLevel.BASIC,
    ) -> ValidationResult:
        """Validate plugin configuration."""

    @abstractmethod
    async def validate_execution_parameters(
        self, parameters: dict[str, Any], pipeline_id: str
    ) -> ValidationResult:
        """Validate execution parameters."""

    @abstractmethod
    async def validate_schema(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> ValidationResult:
        """Validate data against schema."""


class DefaultValidationService(ValidationService):
    """Default implementation of validation service."""

    async def validate_pipeline_config(
        self, config: dict[str, Any], level: ValidationLevel = ValidationLevel.BASIC
    ) -> ValidationResult:
        """Validate pipeline configuration."""
        result = ValidationResult(is_valid=True)

        # Basic validation
        if not config.get("name"):
            result.add_error("Pipeline name is required")

        if not config.get("steps"):
            result.add_error("Pipeline must have at least one step")

        # Strict validation
        if level in {ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE}:
            if not config.get("description"):
                result.add_warning("Pipeline description is recommended")

            steps = config.get("steps", [])
            for i, step in enumerate(steps):
                if not step.get("name"):
                    result.add_error(f"Step {i} is missing name")
                if not step.get("type"):
                    result.add_error(f"Step {i} is missing type")

        # Comprehensive validation
        if level == ValidationLevel.COMPREHENSIVE:
            # Would implement comprehensive validation here
            pass

        return result

    async def validate_plugin_config(
        self,
        config: dict[str, Any],
        plugin_name: str,
        level: ValidationLevel = ValidationLevel.BASIC,
    ) -> ValidationResult:
        """Validate plugin configuration."""
        result = ValidationResult(is_valid=True)

        if not config:
            result.add_error("Plugin configuration cannot be empty")
            return result

        # Basic validation for common plugin fields
        if "version" not in config:
            result.add_warning("Plugin version not specified")

        return result

    async def validate_execution_parameters(
        self, parameters: Any, pipeline_id: str
    ) -> ValidationResult:
        """Validate execution parameters."""
        result = ValidationResult(is_valid=True)

        # Basic parameter validation
        if not isinstance(parameters, dict):
            result.add_error("Execution parameters must be a dictionary")
            return result

        # Additional validations for valid dict
        if not parameters:
            result.add_warning("Execution parameters are empty")

        return result

    async def validate_schema(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> ValidationResult:
        """Validate data against schema."""
        result = ValidationResult(is_valid=True)

        # Basic schema validation - would implement JSON schema validation here
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                result.add_error(f"Required field '{field}' is missing")

        return result
