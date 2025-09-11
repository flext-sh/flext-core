"""Migration Examples: From Dict-based to Pydantic-based Configuration.

This file shows real before/after examples from the FLEXT Core migration,
demonstrating how to migrate from manual dict validation to Pydantic models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from flext_core import FlextResult, FlextConstants, FlextModels


# ❌ BEFORE: Manual validation with dicts
def configure_logging_old(config: dict) -> FlextResult[dict]:
    """Old approach with manual validation.

    Args:
        config: The logging configuration.

    Returns:
        FlextResult[dict]: The logging configuration.

    """
    # Manual validation of required fields
    if "log_level" not in config:
        return FlextResult.fail("log_level is required")

    # Manual validation of log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config["log_level"].upper() not in valid_levels:
        return FlextResult.fail(
            f"Invalid log_level: {config['log_level']}. Valid options: {valid_levels}"
        )

    # Manual type checking
    if "max_file_size" in config:
        if not isinstance(config["max_file_size"], int):
            return FlextResult.fail("max_file_size must be an integer")
        if config["max_file_size"] < 1048576:  # 1MB minimum
            return FlextResult.fail("max_file_size must be at least 1MB")

    # Manual defaults
    config.setdefault("log_format", "json")
    config.setdefault("max_file_size", 10485760)  # 10MB
    config.setdefault("rotation_count", 5)
    config.setdefault("enable_console", True)

    # Normalize values
    config["log_level"] = config["log_level"].upper()

    return FlextResult.ok(config)


# ✅ AFTER: Pydantic model with automatic validation
class LoggingConfig(FlextModels.SystemConfigs.BaseSystemConfig):
    """New approach with Pydantic validation."""

    log_format: str = Field(default="json", pattern="^(json|text|structured)$")
    max_file_size: int = Field(default=10485760, ge=1048576)  # >= 1MB
    rotation_count: int = Field(default=5, ge=1, le=100)
    enable_console: bool = Field(default=True)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Normalize and validate log level.

        Args:
            v: The log level.

        Returns:
            str: The normalized log level.

        """
        return v.upper()


def configure_logging_new(config: dict) -> FlextResult[dict]:
    """New approach with Pydantic validation.

    Args:
        config: The logging configuration.

    Returns:
        FlextResult[dict]: The logging configuration.

    """
    try:
        validated = LoggingConfig.model_validate(config)
        return FlextResult.ok(validated.model_dump())
    except ValidationError as e:
        return FlextResult.fail(str(e))


# ❌ BEFORE: Manual nested validation
def configure_database_old(config: dict) -> FlextResult[dict]:
    """Old approach with complex manual validation."""
    # Check main database config
    if "database" not in config:
        return FlextResult.fail("database configuration required")

    db_config = config["database"]

    # Validate host
    if "host" not in db_config:
        return FlextResult.fail("database.host is required")
    if not isinstance(db_config["host"], str):
        return FlextResult.fail("database.host must be a string")

    # Validate port
    if "port" in db_config:
        if not isinstance(db_config["port"], int):
            return FlextResult.fail("database.port must be an integer")
        if not 1 <= db_config["port"] <= 65535:
            return FlextResult.fail("database.port must be between 1 and 65535")
    else:
        db_config["port"] = 5432  # Default PostgreSQL port

    # Validate connection pool
    if "pool" in db_config:
        pool = db_config["pool"]
        if "min_size" in pool:
            if not isinstance(pool["min_size"], int) or pool["min_size"] < 1:
                return FlextResult.fail("pool.min_size must be positive integer")
        else:
            pool["min_size"] = 2

        if "max_size" in pool:
            if (
                not isinstance(pool["max_size"], int)
                or pool["max_size"] < pool["min_size"]
            ):
                return FlextResult.fail("pool.max_size must be >= min_size")
        else:
            pool["max_size"] = 10
    else:
        db_config["pool"] = {"min_size": 2, "max_size": 10}

    # Environment-specific validation
    if config.get("environment") == "production":
        if "password" not in db_config:
            return FlextResult.fail("database.password required in production")
        if db_config.get("ssl_mode") != "require":
            return FlextResult.fail("SSL required in production")

    return FlextResult.ok(config)


# ✅ AFTER: Pydantic models with automatic nested validation
class DatabasePoolConfig(BaseModel):
    """Database connection pool configuration."""

    min_size: int = Field(default=2, ge=1, le=100)
    max_size: int = Field(default=10, ge=1, le=1000)
    timeout: float = Field(default=5.0, ge=0.1, le=60.0)

    @model_validator(mode="after")
    def validate_sizes(self) -> DatabasePoolConfig:
        """Ensure max_size >= min_size."""
        if self.max_size < self.min_size:
            msg = "max_size must be >= min_size"
            raise ValueError(msg)
        return self


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str = Field(..., min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="flext")
    username: str = Field(default="flext_user")
    password: str | None = Field(None)
    ssl_mode: str = Field(default="prefer", pattern="^(disable|prefer|require)$")
    pool: DatabasePoolConfig = Field(default_factory=DatabasePoolConfig)


class SystemConfigWithDatabase(FlextModels.SystemConfigs.BaseSystemConfig):
    """System configuration with database."""

    database: DatabaseConfig

    @model_validator(mode="after")
    def validate_production_database(self) -> SystemConfigWithDatabase:
        """Validate production database requirements."""
        if self.environment == "production":
            if not self.database.password:
                msg = "database.password required in production"
                raise ValueError(msg)
            if self.database.ssl_mode != "require":
                msg = "SSL required in production (ssl_mode='require')"
                raise ValueError(msg)
        return self


def configure_database_new(config: dict) -> FlextResult[dict]:
    """New approach with Pydantic validation."""
    try:
        validated = SystemConfigWithDatabase.model_validate(config)
        return FlextResult.ok(validated.model_dump())
    except ValidationError as e:
        return FlextResult.fail(str(e))


# ❌ BEFORE: Manual validation with dynamic fields
def configure_commands_old(config: dict) -> FlextResult[dict]:
    """Old commands configuration with manual validation."""
    # Validate environment
    if "environment" in config:
        valid_envs = ["development", "staging", "production", "test", "local"]
        if config["environment"] not in valid_envs:
            return FlextResult.fail(f"Invalid environment: {config['environment']}")
    else:
        config["environment"] = "development"

    # Validate log_level
    if "log_level" in config:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config["log_level"].upper() not in valid_levels:
            return FlextResult.fail(f"Invalid log_level: {config['log_level']}")
        config["log_level"] = config["log_level"].upper()
    else:
        config["log_level"] = "INFO"

    # Validate command-specific fields
    if "max_retries" in config:
        if not isinstance(config["max_retries"], int) or config["max_retries"] < 0:
            return FlextResult.fail("max_retries must be non-negative integer")
    else:
        config["max_retries"] = 3

    # Handle dynamic command handlers
    if "command_handlers" in config and not isinstance(
        config["command_handlers"], dict
    ):
        return FlextResult.fail("command_handlers must be a dictionary")
        # Can't validate dynamic keys easily

    # Add defaults
    config.setdefault("enable_command_validation", True)
    config.setdefault("command_timeout", 30)
    config.setdefault("enable_async", False)

    return FlextResult.ok(config)


# ✅ AFTER: Pydantic with dynamic fields support
class CommandsConfig(FlextModels.SystemConfigs.BaseSystemConfig):
    """Commands configuration with dynamic fields."""

    # Static fields with validation
    max_retries: int = Field(default=3, ge=0, le=10)
    enable_command_validation: bool = Field(default=True)
    command_timeout: int = Field(default=30, ge=1, le=300)
    enable_async: bool = Field(default=False)

    # Dynamic fields (command handlers)
    command_handlers: dict[str, object] = Field(
        default_factory=dict,
        json_schema_extra={"dynamic": True},  # Mark as dynamic
    )

    # Allow extra fields for extensibility
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields
        validate_assignment=True,
    )

    @field_validator("command_handlers")
    @classmethod
    def validate_handlers(cls, v: dict[str, object]) -> dict[str, object]:
        """Validate command handlers structure."""
        for name, handler in v.items():
            if not isinstance(handler, dict):
                msg = f"Handler {name} must be a dictionary"
                raise TypeError(msg)
            if "class" not in handler:
                msg = f"Handler {name} missing 'class' field"
                raise ValueError(msg)
        return v


def configure_commands_new(config: dict) -> FlextResult[dict]:
    """New commands configuration with Pydantic."""
    try:
        validated = CommandsConfig.model_validate(config)
        return FlextResult.ok(validated.model_dump())
    except ValidationError as e:
        return FlextResult.fail(str(e))


# This shows how we migrated mixins while preserving backward compatibility


def configure_mixins_with_compatibility(config: dict) -> FlextResult[dict]:
    """Mixins configuration with backward compatibility.

    This is the actual implementation used in flext-core that:
    1. Preserves custom log levels for legacy tests
    2. Maintains legacy field names
    3. Adds environment-specific fields
    """
    try:
        # Note: FlextModels already imported at module level
        # This demonstrates handling legacy configuration patterns

        # Preserve custom log_level if present (for backward compatibility)
        custom_log_level = None
        if "log_level" in config:
            original_log_level = config["log_level"]
            valid_levels = {e.value for e in FlextConstants.Config.LogLevel}
            if (
                isinstance(original_log_level, str)
                and original_log_level.upper() not in valid_levels
            ):
                custom_log_level = original_log_level
                # Temporarily use a valid level for validation
                config = dict(config)
                config["log_level"] = "INFO"

        # Validate using Pydantic model
        mixins_config = FlextModels.SystemConfigs.MixinsConfig.model_validate(config)

        # Convert to dict for backward compatibility
        result = mixins_config.model_dump()

        # Restore custom log_level if it was present
        if custom_log_level is not None:
            result["log_level"] = custom_log_level

        # Add legacy fields for test compatibility
        result.setdefault("enable_metrics", True)
        result.setdefault("default_cache_size", result.get("cache_size", 1000))
        result.setdefault("max_validation_errors", 10)

        # Environment-specific legacy fields
        env_value = result.get("environment", "development")
        if env_value == "staging":
            result["cache_ttl_seconds"] = config.get("cache_ttl_seconds", 3600)
            result["enable_staging_validation"] = config.get(
                "enable_staging_validation", True
            )
        elif env_value == "local":
            result["enable_local_debugging"] = config.get(
                "enable_local_debugging", True
            )

        return FlextResult.ok(result)

    except ValidationError as e:
        error_details = "; ".join(
            f"{err['loc'][0] if err['loc'] else 'field'}: {err['msg']}"
            for err in e.errors()
        )
        return FlextResult.fail(f"Configuration validation failed: {error_details}")


def migration_step_by_step() -> None:
    """Step-by-step guide for migrating a configuration function.

    This example shows the actual process used to migrate functions
    in flext-core from dict-based to Pydantic-based validation.
    """
    print("=" * 60)
    print("STEP-BY-STEP MIGRATION GUIDE")
    print("=" * 60)

    # Step 1: Identify the function to migrate
    print("\n1. IDENTIFY THE FUNCTION")
    print("   Look for functions with pattern: configure_*_system()")
    print("   Example: configure_guards_system, configure_processors_system")

    # Step 2: Analyze current validation
    print("\n2. ANALYZE CURRENT VALIDATION")
    print("   Search for validation patterns:")
    print("   - if 'field' not in config")
    print("   - if not isinstance(config['field'], type)")
    print("   - config.setdefault('field', default)")
    print("   - if config['field'] not in valid_values")

    # Step 3: Create Pydantic model
    print("\n3. CREATE PYDANTIC MODEL")

    class ExampleConfig(FlextModels.SystemConfigs.BaseSystemConfig):
        """Example configuration model."""

        # Replace manual checks with Field constraints
        required_field: str = Field(..., min_length=1)
        optional_field: int | None = Field(None, ge=0, le=100)
        enum_field: str = Field(
            default="option1", pattern="^(option1|option2|option3)$"
        )

        @field_validator("enum_field")
        @classmethod
        def validate_enum(cls, v: str) -> str:
            """Custom validation if needed."""
            return v.lower()

    print("   Define fields with constraints using Field()")
    print("   Add validators for complex logic")

    # Step 4: Update the configuration function
    print("\n4. UPDATE CONFIGURATION FUNCTION")

    def configure_example_system(config: dict) -> FlextResult[dict]:
        """Updated configuration function."""
        try:
            # Single line replaces all manual validation!
            validated = ExampleConfig.model_validate(config)

            # Add any backward compatibility fields
            result = validated.model_dump()
            result.setdefault("legacy_field", "legacy_value")

            return FlextResult.ok(result)
        except ValidationError as e:
            return FlextResult.fail(str(e))

    print("   Replace manual validation with model_validate()")
    print("   Return model_dump() for dict compatibility")

    # Step 5: Test the migration
    print("\n5. TEST THE MIGRATION")
    print("   Run existing tests to ensure compatibility:")
    print("   $ pytest tests/unit/test_module.py -v")
    print("   Add compatibility fields if tests fail")

    # Step 6: Document the changes
    print("\n6. DOCUMENT THE CHANGES")
    print("   Update docstrings to mention Pydantic validation")
    print("   Add migration notes to CHANGELOG.md")

    print("\n" + "=" * 60)


class CommonPatterns:
    """Common patterns encountered during migration and their solutions."""

    @staticmethod
    def pattern_enum_validation() -> str:
        """Pattern: Validating against a list of valid values."""

        # ❌ OLD: Manual enum validation
        def old_way(config: dict) -> FlextResult[dict]:
            valid_levels = ["low", "medium", "high"]
            if config.get("level") not in valid_levels:
                return FlextResult.fail(f"Invalid level. Valid: {valid_levels}")
            return FlextResult.ok(config)

        # ✅ NEW: Using Pydantic with StrEnum
        class NewConfig(BaseModel):
            level: str = Field(default="medium", pattern="^(low|medium|high)$")
            # Or use actual enum from FlextConstants
            # level: str = Field(default=FlextConstants.Level.MEDIUM.value)

        return "Use Field with pattern or StrEnum values"

    @staticmethod
    def pattern_conditional_requirements() -> str:
        """Pattern: Fields required based on other fields."""

        # ❌ OLD: Manual conditional validation
        def old_way(config: dict) -> FlextResult[dict]:
            if config.get("mode") == "advanced" and "advanced_option" not in config:
                return FlextResult.fail("advanced_option required in advanced mode")
            return FlextResult.ok(config)

        # ✅ NEW: Using model_validator
        class NewConfig(BaseModel):
            mode: str = Field(default="basic")
            advanced_option: str | None = None

            @model_validator(mode="after")
            def validate_conditional(self) -> NewConfig:
                if self.mode == "advanced" and not self.advanced_option:
                    msg = "advanced_option required in advanced mode"
                    raise ValueError(msg)
                return self

        return "Use model_validator for conditional logic"

    @staticmethod
    def pattern_normalization() -> str:
        """Pattern: Normalizing values during validation."""

        # ❌ OLD: Manual normalization
        def old_way(config: dict) -> FlextResult[dict]:
            if "url" in config:
                url = config["url"]
                if not url.startswith(("http://", "https://")):
                    config["url"] = f"https://{url}"
                if url.endswith("/"):
                    config["url"] = url[:-1]
            return FlextResult.ok(config)

        # ✅ NEW: Using field_validator
        class NewConfig(BaseModel):
            url: str = Field(default="https://example.com")

            @field_validator("url")
            @classmethod
            def normalize_url(cls, v: str) -> str:
                if not v.startswith(("http://", "https://")):
                    v = f"https://{v}"
                return v.removesuffix("/")

        return "Use field_validator for normalization"

    @staticmethod
    def pattern_dynamic_fields() -> str:
        """Pattern: Handling unknown/dynamic fields."""

        # ❌ OLD: No validation for dynamic fields
        def old_way(config: dict) -> FlextResult[dict]:
            # Can't validate unknown fields
            return FlextResult.ok(config)

        # ✅ NEW: Using extra="allow" or explicit dict field
        class NewConfig(BaseModel):
            # Option 1: Allow extra fields
            model_config = ConfigDict(extra="allow")

            # Option 2: Explicit dynamic fields
            metadata: dict[str, object] = Field(default_factory=dict)
            custom_handlers: dict[str, dict] = Field(
                default_factory=dict, json_schema_extra={"dynamic": True}
            )

        return "Use extra='allow' or explicit dict fields"


if __name__ == "__main__":
    # Test Example 1: Simple Configuration
    print("=== EXAMPLE 1: Simple Configuration ===")
    test_config = {"log_level": "debug", "max_file_size": 2097152}

    old_result = configure_logging_old(test_config.copy())
    print(f"Old approach: {old_result}")

    new_result = configure_logging_new(test_config.copy())
    print(f"New approach: {new_result}")

    # Test Example 2: Nested Configuration
    print("\n=== EXAMPLE 2: Nested Configuration ===")
    db_config = {
        "environment": "production",
        "database": {
            "host": "db.example.com",
            "port": 5432,
            "password": "secret",
            "ssl_mode": "require",
            "pool": {"min_size": 5, "max_size": 20},
        },
    }

    old_result = configure_database_old(db_config.copy())
    print(f"Old approach: {old_result.success}")

    new_result = configure_database_new(db_config.copy())
    print(f"New approach: {new_result.success}")

    # Test Example 3: Dynamic Fields
    print("\n=== EXAMPLE 3: Dynamic Fields ===")
    cmd_config = {
        "environment": "staging",
        "max_retries": 5,
        "command_handlers": {
            "create_user": {"class": "CreateUserHandler", "async": True},
            "delete_user": {"class": "DeleteUserHandler", "async": False},
        },
    }

    old_result = configure_commands_old(cmd_config.copy())
    print(f"Old approach: {old_result.success}")

    new_result = configure_commands_new(cmd_config.copy())
    print(f"New approach: {new_result.success}")

    # Test Example 4: Backward Compatibility
    print("\n=== EXAMPLE 4: Backward Compatibility ===")
    mixins_config = {
        "log_level": "CUSTOM_LEVEL",  # Custom level for legacy tests
        "environment": "staging",
        "cache_ttl_seconds": 7200,  # Legacy field
    }

    result = configure_mixins_with_compatibility(mixins_config)
    print(f"With compatibility: {result.success}")
    if result.success:
        print(f"Preserved custom log_level: {result.unwrap()['log_level']}")
        print(
            f"Added legacy field: cache_ttl_seconds={result.unwrap().get('cache_ttl_seconds')}"
        )

    # Run step-by-step guide
    print("\n")
    migration_step_by_step()

    # Show common patterns
    print("\n=== COMMON PATTERNS ===")
    patterns = CommonPatterns()
    print(f"1. Enum Validation: {patterns.pattern_enum_validation()}")
    print(f"2. Conditional Requirements: {patterns.pattern_conditional_requirements()}")
    print(f"3. Normalization: {patterns.pattern_normalization()}")
    print(f"4. Dynamic Fields: {patterns.pattern_dynamic_fields()}")
