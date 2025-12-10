"""Constants for FLEXT tests.

Provides FlextTestsConstants, extending FlextConstants with test-specific constants
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Literal

from flext_core.constants import FlextConstants

# Import t at runtime - no circular dependency since typings.py doesn't import constants
from flext_tests.typings import t


class FlextTestsConstants(FlextConstants):
    """Constants for FLEXT tests - extends FlextConstants.

    Architecture: Extends FlextConstants with test-specific constants.
    All base constants from FlextConstants are available through inheritance.
    Uses StrEnum and Literals for type-safe constants following Python 3.13+ patterns.
    """

    class Tests:
        """Test-specific constants namespace.

        All test-specific constants are organized under this namespace to clearly
        distinguish them from base FlextConstants. Access via c.Tests.*
        """

        class Docker:
            """Docker test infrastructure constants for test infrastructure."""

            # Test-specific Docker constants (not in FlextConstants)
            # Use c helper for accessing base constants
            DEFAULT_LOG_TAIL: Final[int] = 100
            DEFAULT_CONTAINER_CHOICES: Final[tuple[str, ...]] = (
                "postgres",
                "redis",
                "mongodb",
                "elasticsearch",
            )
            SHARED_CONTAINERS: Final[Mapping[str, t.Types.ContainerConfigDict]] = {}

            # Test-specific Docker constants
            DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
            MAX_TIMEOUT_SECONDS: Final[int] = 300
            DEFAULT_HEALTH_CHECK_INTERVAL: Final[int] = 2
            DEFAULT_HEALTH_CHECK_RETRIES: Final[int] = 10
            DEFAULT_STARTUP_WAIT_SECONDS: Final[int] = 5

            class ContainerStatus(StrEnum):
                """Container status enumeration for test infrastructure."""

                RUNNING = "running"
                STOPPED = "stopped"
                NOT_FOUND = "not_found"
                ERROR = "error"
                STARTING = "starting"
                STOPPING = "stopping"
                RESTARTING = "restarting"

            class Operation(StrEnum):
                """Docker operation types."""

                START = "start"
                STOP = "stop"
                RESTART = "restart"
                REMOVE = "remove"
                BUILD = "build"
                PULL = "pull"
                LOGS = "logs"
                EXEC = "exec"

            # Literal types for type-safe operations
            type OperationLiteral = Literal[
                "start",
                "stop",
                "restart",
                "remove",
                "build",
                "pull",
                "logs",
                "exec",
            ]
            """Type-safe literal for Docker operations."""

            type ContainerStatusLiteral = Literal[
                "running",
                "stopped",
                "not_found",
                "error",
                "starting",
                "stopping",
                "restarting",
            ]
            """Type-safe literal for container status."""

            # Error messages
            ERROR_CONTAINER_NOT_FOUND: Final[str] = "Container not found"
            ERROR_CONTAINER_ALREADY_RUNNING: Final[str] = "Container already running"
            ERROR_CONTAINER_NOT_RUNNING: Final[str] = "Container not running"
            ERROR_DOCKER_NOT_AVAILABLE: Final[str] = "Docker not available"
            ERROR_COMPOSE_FILE_NOT_FOUND: Final[str] = "Docker compose file not found"
            ERROR_OPERATION_TIMEOUT: Final[str] = "Docker operation timed out"

        class Matcher:
            """Matcher constants for test assertions (tm.* methods).

            Provides error message templates with .format() support.
            Use c.Tests.Matcher.* for access.

            Usage:
                c.Tests.Matcher.ERR_EXPECTED_SUCCESS.format(error="err")
                c.Tests.Matcher.ERR_LENGTH_MISMATCH.format(expected=5, actual=3)
            """

            # Result assertion messages
            ERR_EXPECTED_SUCCESS: Final[str] = (
                "Expected success but got failure: {error}"
            )
            ERR_EXPECTED_FAILURE: Final[str] = (
                "Expected failure but got success: {value}"
            )
            ERR_ERROR_CONTAINS: Final[str] = (
                "Expected error containing '{expected}' but got: {actual}"
            )
            ERR_VALUE_NONE: Final[str] = "Expected error but got None"
            ERR_CHAIN_NO_VALUE: Final[str] = "Cannot get value from failed result"
            ERR_CHAIN_NO_ERROR: Final[str] = "Cannot get error from successful result"

            # Equality/containment messages
            ERR_EXPECTED_VALUE: Final[str] = "Expected {expected!r}, got {actual!r}"
            ERR_KEY_NOT_FOUND: Final[str] = "Key '{key}' not found in dict"
            ERR_KEY_VALUE_MISMATCH: Final[str] = (
                "Key '{key}': expected {expected!r}, got {actual!r}"
            )
            ERR_NOT_IN_STRING: Final[str] = "Expected '{item}' in '{container}'"
            ERR_NOT_IN_SEQUENCE: Final[str] = "Expected {item!r} in sequence"

            # Length messages
            ERR_LENGTH_EXACT: Final[str] = "Expected length {expected}, got {actual}"
            ERR_LENGTH_GT: Final[str] = "Expected length > {min}, got {actual}"
            ERR_LENGTH_GTE: Final[str] = "Expected length >= {min}, got {actual}"
            ERR_LENGTH_LT: Final[str] = "Expected length < {max}, got {actual}"
            ERR_LENGTH_LTE: Final[str] = "Expected length <= {max}, got {actual}"
            ERR_EMPTY_SEQUENCE: Final[str] = "Expected non-empty sequence"

            # Type/None messages
            ERR_EXPECTED_NONE: Final[str] = "Expected None, got {value!r}"
            ERR_EXPECTED_NOT_NONE: Final[str] = "Expected not None, got None"
            ERR_TYPE_MISMATCH: Final[str] = "Expected {expected}, got {actual}"

            # String messages
            ERR_NOT_CONTAINS: Final[str] = "Expected '{substring}' in '{text}'"
            ERR_NOT_STARTSWITH: Final[str] = (
                "Expected '{text}' to start with '{prefix}'"
            )
            ERR_NOT_ENDSWITH: Final[str] = "Expected '{text}' to end with '{suffix}'"
            ERR_NOT_MATCHES: Final[str] = (
                "Expected '{text}' to match pattern '{pattern}'"
            )
            ERR_SHOULD_NOT_CONTAIN: Final[str] = "Expected '{excluded}' NOT in '{text}'"

            # Enhanced matcher error messages
            ERR_OK_FAILED: Final[str] = "Expected success but got failure: {error}"
            ERR_FAIL_EXPECTED: Final[str] = (
                "Expected failure but got success with value: {value!r}"
            )
            ERR_EQ_FAILED: Final[str] = "Expected {expected!r} but got {actual!r}"
            ERR_NE_FAILED: Final[str] = "Expected value different from {value!r}"
            ERR_TYPE_FAILED: Final[str] = "Expected type {expected} but got {actual}"
            ERR_CONTAINS_FAILED: Final[str] = (
                "Expected {container!r} to contain {item!r}"
            )
            ERR_LACKS_FAILED: Final[str] = (
                "Expected {container!r} to NOT contain {item!r}"
            )
            ERR_LEN_EXACT_FAILED: Final[str] = (
                "Expected length {expected} but got {actual}"
            )
            ERR_LEN_RANGE_FAILED: Final[str] = (
                "Expected length in range [{min}, {max}] but got {actual}"
            )
            ERR_DEEP_PATH_FAILED: Final[str] = (
                "Deep match failed at path '{path}': {reason}"
            )
            ERR_PREDICATE_FAILED: Final[str] = (
                "Custom predicate failed for value: {value!r}"
            )
            ERR_ALL_ITEMS_FAILED: Final[str] = (
                "Not all items match: failed at index {index}"
            )
            ERR_ANY_ITEMS_FAILED: Final[str] = "No items match the predicate"
            ERR_KEYS_MISSING: Final[str] = "Missing required keys: {keys}"
            ERR_KEYS_EXTRA: Final[str] = "Unexpected keys present: {keys}"
            ERR_SCOPE_PATH_NOT_FOUND: Final[str] = (
                "Path '{path}' not found in value: {error}"
            )

            # Error code/data validation messages
            ERR_ERROR_CODE_MISMATCH: Final[str] = (
                "Expected error code {expected!r} but got {actual!r}"
            )
            ERR_ERROR_CODE_NOT_CONTAINS: Final[str] = (
                "Expected error code to contain {expected!r} but got {actual!r}"
            )
            ERR_ERROR_DATA_KEY_MISSING: Final[str] = (
                "Expected error data key {key!r} not found"
            )
            ERR_ERROR_DATA_VALUE_MISMATCH: Final[str] = (
                "Error data key {key!r}: expected {expected!r}, got {actual!r}"
            )

            # Scope and cleanup messages
            ERR_SCOPE_CLEANUP_FAILED: Final[str] = (
                "Cleanup function failed in scope: {error}"
            )

            # Validation messages
            ERR_EMAIL_NOT_STRING: Final[str] = (
                "{field} must be a string for email validation"
            )
            ERR_INVALID_EMAIL: Final[str] = "Invalid email format: '{value}'"
            ERR_CONFIG_NOT_DICT: Final[str] = (
                "{field} must be a dict for config validation"
            )
            ERR_CONFIG_MISSING_KEY: Final[str] = "Required config key '{key}' missing"
            ERR_EMPTY_VALUE: Final[str] = "{field} cannot be empty"

            # Config validation defaults
            CONFIG_REQUIRED_KEYS: Final[tuple[str, ...]] = (
                "service_type",
                "environment",
            )

            # Email validation pattern
            EMAIL_PATTERN: Final[str] = (
                r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            )

        class Factory:
            """Factory constants for test data generation (tt.* methods).

            Provides default values, error messages, and configuration templates
            for FlextTestsFactories. Use c.Tests.Factory.* for access.
            """

            # User defaults
            DEFAULT_USER_NAME: Final[str] = "Test User"
            DEFAULT_USER_EMAIL_TEMPLATE: Final[str] = "user_{id}@example.com"
            DEFAULT_USER_ACTIVE: Final[bool] = True

            # Config defaults
            DEFAULT_SERVICE_TYPE: Final[str] = "api"
            DEFAULT_ENVIRONMENT: Final[str] = "test"
            DEFAULT_DEBUG: Final[bool] = True
            DEFAULT_LOG_LEVEL: Final[str] = "DEBUG"
            DEFAULT_TIMEOUT: Final[int] = 30
            DEFAULT_MAX_RETRIES: Final[int] = 3

            # Service defaults
            DEFAULT_SERVICE_STATUS: Final[str] = "active"
            DEFAULT_SERVICE_NAME_TEMPLATE: Final[str] = "Test {type} Service"

            # Entity/Value defaults
            DEFAULT_ENTITY_NAME: Final[str] = "test_entity"
            DEFAULT_VALUE_DATA: Final[str] = "test_data"
            DEFAULT_VALUE_COUNT: Final[int] = 0

            # Batch defaults
            DEFAULT_BATCH_COUNT: Final[int] = 5
            DEFAULT_BATCH_ENVIRONMENTS: Final[tuple[str, ...]] = (
                "test",
                "staging",
                "production",
            )
            DEFAULT_BATCH_SERVICE_TYPES: Final[tuple[str, ...]] = (
                "api",
                "database",
                "cache",
            )

            # Result error messages (support .format() for customization)
            ERROR_VALUE_NONE: Final[str] = "Value cannot be None"
            ERROR_DEFAULT: Final[str] = "Operation failed"
            ERROR_VALIDATION: Final[str] = "Validation failed"
            ERROR_NOT_FOUND: Final[str] = "Not found"
            ERROR_VALIDATION_FAILED: Final[str] = "Validation failed"
            ERROR_OPERATION_FAILED: Final[str] = "Operation failed"

            # Operation messages
            SUCCESS_MESSAGE: Final[str] = "success"

            # Deprecation message templates - use .format() for method names
            DEPRECATION_RESULT_OK: Final[str] = (
                'Result.ok() is deprecated. Use tt.res("ok", value=value) instead.'
            )
            DEPRECATION_RESULT_FAIL: Final[str] = (
                'Result.fail() is deprecated. Use tt.res("fail", error=error) instead.'
            )
            DEPRECATION_RESULT_FROM_VALUE: Final[str] = (
                "Result.from_value() deprecated. Use tt.res('from_value')."
            )
            DEPRECATION_MODELS_USER: Final[str] = (
                "Models.user() deprecated. Use tt.model('user', ...)."
            )
            DEPRECATION_MODELS_CONFIG: Final[str] = (
                "Models.config() deprecated. Use tt.model('config', ...)."
            )
            DEPRECATION_MODELS_SERVICE: Final[str] = (
                "Models.service() deprecated. Use tt.model('service', ...)."
            )
            DEPRECATION_MODELS_ENTITY: Final[str] = (
                "Models.entity() deprecated. Use tt.model('entity', ...)."
            )
            DEPRECATION_MODELS_VALUE_OBJECT: Final[str] = (
                "Models.value_object() deprecated. Use tt.model('value', ...)."
            )
            DEPRECATION_OPS_SIMPLE: Final[str] = (
                "Operations.simple() deprecated. Use tt.op('simple')."
            )
            DEPRECATION_OPS_ADD: Final[str] = (
                "Operations.add() deprecated. Use tt.op('add')."
            )
            DEPRECATION_OPS_FORMAT: Final[str] = (
                "Operations.format() deprecated. Use tt.op('format')."
            )
            DEPRECATION_OPS_ERROR: Final[str] = (
                "Operations.error() deprecated. Use tt.op('error', ...)."
            )
            DEPRECATION_OPS_TYPE_ERROR: Final[str] = (
                "Operations.type_error() deprecated. Use tt.op('type_error')."
            )
            DEPRECATION_OPS_RESULT_OK: Final[str] = (
                "Operations.result_ok() deprecated. Use tt.op('result_ok')."
            )
            DEPRECATION_OPS_RESULT_FAIL: Final[str] = (
                "Operations.result_fail() deprecated. Use tt.op('result_fail')."
            )
            DEPRECATION_BATCH_USERS: Final[str] = (
                "Batch.users() deprecated. Use tt.batch('user', count=n)."
            )
            DEPRECATION_BATCH_CONFIGS: Final[str] = (
                "Batch.configs() deprecated. Use tt.batch('config', count=n)."
            )
            DEPRECATION_BATCH_SERVICES: Final[str] = (
                "Batch.services() deprecated. Use tt.batch('service', count=n)."
            )
            DEPRECATION_BATCH_RESULTS: Final[str] = (
                "Batch.results() deprecated. Use tt.results(values, ...)."
            )
            DEPRECATION_CREATE_USER: Final[str] = (
                "create_user() deprecated. Use tt.model('user', ...)."
            )
            DEPRECATION_CREATE_CONFIG: Final[str] = (
                "create_config() deprecated. Use tt.model('config', ...)."
            )
            DEPRECATION_CREATE_SERVICE: Final[str] = (
                "create_service() deprecated. Use tt.model('service', ...)."
            )
            DEPRECATION_BATCH_USERS_FUNC: Final[str] = (
                "batch_users() deprecated. Use tt.batch('user', count=n)."
            )
            DEPRECATION_CREATE_TEST_OPERATION: Final[str] = (
                "create_test_operation() deprecated. Use tt.op(kind, ...)."
            )
            DEPRECATION_CREATE_TEST_SERVICE: Final[str] = (
                "create_test_service() deprecated. Use tt.svc(...)."
            )

            @classmethod
            def user_email(cls, user_id: str) -> str:
                """Generate user email from template.

                Args:
                    user_id: User identifier for email generation.

                Returns:
                    Formatted email address.

                """
                return cls.DEFAULT_USER_EMAIL_TEMPLATE.format(id=user_id)

            @classmethod
            def service_name(cls, service_type: str) -> str:
                """Generate service name from template.

                Args:
                    service_type: Type of service.

                Returns:
                    Formatted service name.

                """
                return cls.DEFAULT_SERVICE_NAME_TEMPLATE.format(type=service_type)

        class Execution:
            """Test execution constants for test infrastructure."""

            # Test execution timeouts
            DEFAULT_TEST_TIMEOUT_SECONDS: Final[int] = 60
            MAX_TEST_TIMEOUT_SECONDS: Final[int] = 600

            # Test data generation
            DEFAULT_BATCH_SIZE: Final[int] = 10
            MAX_BATCH_SIZE: Final[int] = 1000

            # Test fixture constants
            DEFAULT_FIXTURE_COUNT: Final[int] = 5
            MAX_FIXTURE_COUNT: Final[int] = 100

        class Files:
            """File management constants for test infrastructure.

            Provides format mappings, default values, error messages, and deprecation
            message templates for FlextTestsFiles. Use c.Tests.Files.* for access.
            """

            # Format types
            class Format(StrEnum):
                """File format enumeration."""

                AUTO = "auto"
                TEXT = "text"
                BIN = "bin"
                JSON = "json"
                YAML = "yaml"
                CSV = "csv"
                UNKNOWN = "unknown"

            # Literal type for format
            type FormatLiteral = Literal[
                "auto",
                "text",
                "bin",
                "json",
                "yaml",
                "csv",
                "unknown",
            ]

            # Compare modes
            class CompareMode(StrEnum):
                """File comparison mode enumeration."""

                CONTENT = "content"
                SIZE = "size"
                HASH = "hash"
                LINES = "lines"

            type CompareModeLiteral = Literal["content", "size", "hash", "lines"]

            # Batch operation constants
            DEFAULT_BATCH_SIZE: Final[int] = 100
            BATCH_TIMEOUT_SECONDS: Final[int] = 30

            # Operation types
            class Operation(StrEnum):
                """File operation types for batch operations."""

                CREATE = "create"
                READ = "read"
                DELETE = "delete"
                COMPARE = "compare"
                INFO = "info"

            type OperationLiteral = Literal[
                "create",
                "read",
                "delete",
                "compare",
                "info",
            ]
            """Type-safe literal for file operations."""

            # Error handling modes
            class ErrorMode(StrEnum):
                """Error handling modes for batch operations."""

                STOP = "stop"
                SKIP = "skip"
                COLLECT = "collect"

            type ErrorModeLiteral = Literal["stop", "skip", "collect"]
            """Type-safe literal for error handling modes."""

            # Extension to format mapping
            EXT_TO_FMT: Final[Mapping[str, str]] = {
                ".txt": "text",
                ".log": "text",
                ".md": "text",
                ".rst": "text",
                ".bin": "bin",
                ".dat": "bin",
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".csv": "csv",
                ".tsv": "csv",
            }

            # Default values
            DEFAULT_FILENAME: Final[str] = "file"
            DEFAULT_TEXT_FILENAME: Final[str] = "test.txt"
            DEFAULT_BINARY_FILENAME: Final[str] = "binary_data.bin"
            DEFAULT_EMPTY_FILENAME: Final[str] = "empty.txt"
            DEFAULT_CONFIG_FILENAME: Final[str] = "config.json"
            DEFAULT_ENCODING: Final[str] = "utf-8"
            DEFAULT_BINARY_ENCODING: Final[str] = "binary"
            DEFAULT_JSON_INDENT: Final[int] = 2
            DEFAULT_CSV_DELIMITER: Final[str] = ","
            DEFAULT_EXTENSION: Final[str] = ".txt"

            # Directory defaults
            DEFAULT_READONLY_DIR_NAME: Final[str] = "readonly"

            # Permissions
            PERMISSION_READONLY_FILE: Final[int] = 0o444
            PERMISSION_WRITABLE_FILE: Final[int] = 0o644
            PERMISSION_READONLY_DIR: Final[int] = 0o555
            PERMISSION_WRITABLE_DIR: Final[int] = 0o755

            # Hash settings
            HASH_CHUNK_SIZE: Final[int] = 8192

            # Size units for human-readable format
            SIZE_UNITS: Final[tuple[str, ...]] = ("B", "KB", "MB", "GB", "TB", "PB")
            SIZE_THRESHOLD: Final[int] = 1024

            # Error messages (support .format() for customization)
            ERROR_FILE_NOT_FOUND: Final[str] = "File not found: {path}"
            ERROR_INVALID_JSON: Final[str] = "Invalid JSON: {error}"
            ERROR_INVALID_YAML: Final[str] = "Invalid YAML: {error}"
            ERROR_ENCODING: Final[str] = "Encoding error: {error}"
            ERROR_READ: Final[str] = "Read error: {error}"
            ERROR_COMPARE: Final[str] = "Compare error: {error}"
            ERROR_INFO: Final[str] = "Info error: {error}"

            # Deprecation message templates
            DEPRECATION_CREATE_TEXT: Final[str] = (
                "create_text_file() is deprecated. Use create(content, name) instead."
            )
            DEPRECATION_CREATE_BINARY: Final[str] = (
                "create_binary_file() is deprecated. "
                "Use create(content, name, fmt='bin') instead."
            )
            DEPRECATION_CREATE_EMPTY: Final[str] = (
                "create_empty_file() is deprecated. Use create('', name) instead."
            )
            DEPRECATION_CREATE_CONFIG: Final[str] = (
                "create_config_file() is deprecated. Use create(content, name) instead."
            )
            DEPRECATION_CREATE_FILE_SET: Final[str] = (
                "create_file_set() is deprecated. Use tf.files(content) instead."
            )
            DEPRECATION_GET_FILE_INFO: Final[str] = (
                "get_file_info() is deprecated. Use info(path) instead. "
                "Note: info() returns r[FileInfo]."
            )
            DEPRECATION_TEMPORARY_FILES: Final[str] = (
                "temporary_files() is deprecated. Use tf.files(content) instead."
            )

            @classmethod
            def format_size(cls, size: int) -> str:
                """Format size in human-readable format.

                Args:
                    size: Size in bytes.

                Returns:
                    Human-readable size string like "1.2 KB".

                """
                for unit in cls.SIZE_UNITS:
                    if size < cls.SIZE_THRESHOLD:
                        return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
                    size //= cls.SIZE_THRESHOLD
                return f"{size:.1f} PB"

            @classmethod
            def get_format(cls, extension: str) -> str:
                """Get format from file extension.

                Args:
                    extension: File extension (e.g., ".json").

                Returns:
                    Format string or "text" as default.

                """
                return cls.EXT_TO_FMT.get(extension.lower(), "text")

        # Network constants are available via FlextConstants.Network
        # Access via: FlextConstants.Network.MIN_PORT, FlextConstants.Network.MAX_PORT

        class Builders:
            """Builder constants for test data construction.

            Provides default values, error messages, and templates
            for FlextTestsBuilders. Use c.Tests.Builders.* for access.

            Usage:
                count = c.Tests.Builders.DEFAULT_USER_COUNT
                msg = c.Tests.Builders.ERROR_INVALID_COUNT.format(count=-1)
                email = c.Tests.Builders.validation_email(index=0)
            """

            # Default counts
            DEFAULT_USER_COUNT: Final[int] = 5
            DEFAULT_VALIDATION_COUNT: Final[int] = 5

            # Dict keys - dataset root level
            KEY_USERS: Final[str] = "users"
            KEY_CONFIGS: Final[str] = "configs"
            KEY_VALIDATION_FIELDS: Final[str] = "validation_fields"

            # Dict keys - validation fields
            KEY_VALID_EMAILS: Final[str] = "valid_emails"
            KEY_INVALID_EMAILS: Final[str] = "invalid_emails"
            KEY_VALID_HOSTNAMES: Final[str] = "valid_hostnames"
            KEY_INVALID_HOSTNAMES: Final[str] = "invalid_hostnames"

            # User dict keys
            KEY_ID: Final[str] = "id"
            KEY_NAME: Final[str] = "name"
            KEY_EMAIL: Final[str] = "email"
            KEY_ACTIVE: Final[str] = "active"

            # Config dict keys
            KEY_SERVICE_TYPE: Final[str] = "service_type"
            KEY_ENVIRONMENT: Final[str] = "environment"
            KEY_DEBUG: Final[str] = "debug"
            KEY_LOG_LEVEL: Final[str] = "log_level"
            KEY_TIMEOUT: Final[str] = "timeout"
            KEY_MAX_RETRIES: Final[str] = "max_retries"
            KEY_DATABASE_URL: Final[str] = "database_url"
            KEY_MAX_CONNECTIONS: Final[str] = "max_connections"

            # Default values
            DEFAULT_DATABASE_URL: Final[str] = "postgresql://localhost/testdb"
            DEFAULT_MAX_CONNECTIONS: Final[int] = 10
            DEFAULT_ENVIRONMENT_PRODUCTION: Final[str] = "production"
            DEFAULT_ENVIRONMENT_DEVELOPMENT: Final[str] = "development"

            # Validation test data
            INVALID_EMAIL_SAMPLES: Final[tuple[str, ...]] = (
                "invalid",
                "no-at-sign.com",
                "",
            )
            VALID_HOSTNAME_SAMPLES: Final[tuple[str, ...]] = (
                "example.com",
                "localhost",
            )
            INVALID_HOSTNAME_SAMPLES: Final[tuple[str, ...]] = ("invalid..hostname", "")

            # Email template for validation
            VALIDATION_EMAIL_TEMPLATE: Final[str] = "user{index}@example.com"

            # Error messages with .format() support
            ERROR_EMPTY_DATASET: Final[str] = "Cannot build empty dataset"
            ERROR_INVALID_COUNT: Final[str] = "Count must be positive: {count}"

            @classmethod
            def validation_email(cls, index: int) -> str:
                """Generate validation email from template.

                Args:
                    index: Index for email generation.

                Returns:
                    Formatted email address.

                """
                return cls.VALIDATION_EMAIL_TEMPLATE.format(index=index)

        class Validator:
            """Architecture validator constants.

            Provides rule definitions, severity levels, approved patterns.
            Use c.Tests.Validator.* for access.

            Usage:
                severity, desc = c.Tests.Validator.Rules.IMPORT_001
                c.Tests.Validator.Messages.VIOLATION.format(rule_id="X")
            """

            class Severity(StrEnum):
                """Violation severity levels."""

                CRITICAL = "CRITICAL"
                HIGH = "HIGH"
                MEDIUM = "MEDIUM"
                LOW = "LOW"

            # Literal type for severity
            type SeverityLiteral = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]

            class Rules:
                """Rule definitions with (severity, description) tuples.

                Access via c.Tests.Validator.Rules.IMPORT_001, etc.
                Each rule is a tuple of (severity: str, description: str).
                """

                # Import rules (IMPORT-001 to IMPORT-006)
                IMPORT_001: Final[tuple[str, str]] = (
                    "HIGH",
                    "Lazy import (not at module top)",
                )
                IMPORT_002: Final[tuple[str, str]] = (
                    "HIGH",
                    "TYPE_CHECKING block detected",
                )
                IMPORT_003: Final[tuple[str, str]] = (
                    "HIGH",
                    "try/except ImportError pattern",
                )
                IMPORT_004: Final[tuple[str, str]] = (
                    "CRITICAL",
                    "sys.path manipulation",
                )
                IMPORT_005: Final[tuple[str, str]] = (
                    "MEDIUM",
                    "Direct technology import (should use facade)",
                )
                IMPORT_006: Final[tuple[str, str]] = (
                    "HIGH",
                    "Non-root import from flext-* package",
                )

                # Type rules (TYPE-001 to TYPE-003)
                TYPE_001: Final[tuple[str, str]] = (
                    "CRITICAL",
                    "# type: ignore comment",
                )
                TYPE_002: Final[tuple[str, str]] = ("CRITICAL", "Any type annotation")
                TYPE_003: Final[tuple[str, str]] = ("MEDIUM", "Unapproved  usage")

                # Test rules (TEST-001 to TEST-003)
                TEST_001: Final[tuple[str, str]] = (
                    "HIGH",
                    "monkeypatch usage detected",
                )
                TEST_002: Final[tuple[str, str]] = (
                    "HIGH",
                    "Mock/MagicMock usage detected",
                )
                TEST_003: Final[tuple[str, str]] = (
                    "HIGH",
                    "@patch decorator usage detected",
                )

                # Config rules (CONFIG-001 to CONFIG-005)
                CONFIG_001: Final[tuple[str, str]] = (
                    "CRITICAL",
                    "mypy ignore_errors = true",
                )
                CONFIG_002: Final[tuple[str, str]] = (
                    "HIGH",
                    "Custom ruff ignore beyond approved list",
                )
                CONFIG_003: Final[tuple[str, str]] = (
                    "MEDIUM",
                    "disallow_incomplete_defs = false",
                )
                CONFIG_004: Final[tuple[str, str]] = (
                    "MEDIUM",
                    "warn_return_any = false",
                )
                CONFIG_005: Final[tuple[str, str]] = (
                    "LOW",
                    "reportPrivateUsage = false",
                )

                # Bypass rules (BYPASS-001 to BYPASS-003)
                BYPASS_001: Final[tuple[str, str]] = ("MEDIUM", "noqa comment detected")
                BYPASS_002: Final[tuple[str, str]] = (
                    "LOW",
                    "pragma: no cover (unapproved)",
                )
                BYPASS_003: Final[tuple[str, str]] = (
                    "HIGH",
                    "Exception swallowing (bare except or pass)",
                )

                # Layer rules (LAYER-001)
                LAYER_001: Final[tuple[str, str]] = (
                    "CRITICAL",
                    "Lower layer importing upper layer",
                )

                @classmethod
                def get(
                    cls,
                    rule_id: str,
                ) -> tuple[FlextTestsConstants.Tests.Validator.SeverityLiteral, str]:
                    """Get rule by ID string (e.g., 'IMPORT-001' -> IMPORT_001).

                    Args:
                        rule_id: Rule identifier like "IMPORT-001".

                    Returns:
                        Tuple of (severity, description).

                    Raises:
                        KeyError: If rule_id not found.

                    """
                    attr_name = rule_id.replace("-", "_")
                    rule: tuple[
                        FlextTestsConstants.Tests.Validator.SeverityLiteral,
                        str,
                    ] = getattr(cls, attr_name)
                    return rule

            class Messages:
                """Error message templates supporting .format().

                Usage:
                    c.Tests.Validator.Messages.VIOLATION.format(rule_id="X")
                """

                # Violation messages
                VIOLATION: Final[str] = "{rule_id} at {file}:{line}"
                VIOLATION_DETAIL: Final[str] = (
                    "{rule_id}: {description} at {file}:{line}"
                )
                VIOLATION_WITH_SNIPPET: Final[str] = (
                    "{rule_id}: {description}\n  → {snippet}"
                )

                # Scan messages
                SCAN_COMPLETE: Final[str] = (
                    "Scanned {count} files, found {violations} violations"
                )
                SCAN_PASSED: Final[str] = (
                    "Validation passed: {count} files, 0 violations"
                )
                SCAN_FAILED: Final[str] = (
                    "Validation failed: {violations} violations in {count} files"
                )

                # Layer messages
                LAYER_VIOLATION: Final[str] = (
                    "'{current}' L{current_level} -> '{imported}' L{imported_level}"
                )

                # Module-specific messages
                IMPORT_TECH: Final[str] = "Direct technology import: {module}"
                IMPORT_NON_ROOT: Final[str] = "Non-root import: from {module}"
                CONFIG_IGNORE: Final[str] = "ignore_errors = true for module '{module}'"
                CONFIG_RUFF: Final[str] = "Custom ruff ignore: {code}"
                TEST_MONKEYPATCH: Final[str] = "monkeypatch usage in function '{func}'"
                TYPE_ANY_ARG: Final[str] = "Any type in argument '{arg}'"
                TYPE_ANY_RETURN: Final[str] = "Any type in return type"
                BYPASS_EXCEPTION: Final[str] = "Exception swallowing: {pattern}"
                BYPASS_BARE_EXCEPT: Final[str] = "bare except"
                BYPASS_ONLY_PASS: Final[str] = "except with only pass"

            class Defaults:
                """Default values for validator configuration."""

                # File patterns to exclude from scanning
                EXCLUDE_PATTERNS: Final[tuple[str, ...]] = (
                    "**/.venv/**",
                    "**/venv/**",
                    "**/__pycache__/**",
                    "**/build/**",
                    "**/dist/**",
                    "**/.git/**",
                    "**/htmlcov/**",
                    "**/*.pyc",
                )

                # Include patterns for scanning
                INCLUDE_PATTERNS: Final[tuple[str, ...]] = ("**/*.py",)

                # Validator names
                VALIDATOR_IMPORTS: Final[str] = "imports"
                VALIDATOR_TYPES: Final[str] = "types"
                VALIDATOR_TESTS: Final[str] = "tests"
                VALIDATOR_CONFIG: Final[str] = "config"
                VALIDATOR_BYPASS: Final[str] = "bypass"
                VALIDATOR_LAYER: Final[str] = "layer"

            class Approved:
                """Approved patterns and exceptions for validators."""

                # Approved  file patterns (TYPE-003)
                CAST_PATTERNS: Final[tuple[str, ...]] = (
                    r"service\.py$",  # Protocol-to-concrete for nested classes
                    r"container\.py$",  # DI resolution casts
                )

                # Approved pragma: no cover file patterns (BYPASS-002)
                PRAGMA_PATTERNS: Final[tuple[str, ...]] = (
                    r"__init__\.py$",  # Init files may have conditional imports
                )

                # Approved ruff ignores (CONFIG-002) - from ruff-shared.toml
                RUFF_IGNORES: Final[frozenset[str]] = frozenset({
                    "BLE001",
                    "COM812",
                    "CPY001",
                    "D203",
                    "D213",
                    "D401",
                    "D417",
                    "DOC201",
                    "DOC202",
                    "DOC402",
                    "DOC501",
                    "DOC502",
                    "E501",
                    "ERA001",
                    "FBT003",
                    "G004",
                    "N813",
                    "N816",
                    "PLR0904",
                    "PLR0911",
                    "PLR0912",
                    "PLR0913",
                    "PLR0914",
                    "PLR0915",
                    "PLR0917",
                    "PLR6301",
                    "PYI042",
                    "Q000",
                    "RUF001",
                    "RUF002",
                    "RUF003",
                    "RUF005",
                    "S608",
                    "TC001",
                    "TC002",
                    "TC003",
                    "TRY003",
                    "TRY300",
                    "TRY301",
                    "UP007",
                    "UP040",
                    "W293",
                })

                # Direct technology imports that should use facades (IMPORT-005)
                TECH_IMPORTS: Final[frozenset[str]] = frozenset({
                    "ldap3",
                    "oracledb",
                    "cx_Oracle",
                    "click",
                    "rich",
                    "typer",
                })

                # Mock patterns to detect (TEST-002)
                MOCK_NAMES: Final[frozenset[str]] = frozenset({
                    "Mock",
                    "MagicMock",
                    "AsyncMock",
                    "PropertyMock",
                })

                # FLEXT packages for root import checks (IMPORT-006)
                FLEXT_PACKAGES: Final[frozenset[str]] = frozenset({
                    "flext_core",
                    "flext_cli",
                    "flext_ldap",
                    "flext_ldif",
                    "flext_tests",
                })

                # Approved internal import patterns (IMPORT-006)
                # __init__.py in internal packages can import siblings
                INTERNAL_INIT_PATTERNS: Final[tuple[str, ...]] = (
                    r"_[^/]+/__init__\.py$",  # _validator/__init__.py
                )

            class LayerHierarchy:
                """Layer hierarchy definitions for LAYER-001 validation.

                Lower number = lower layer (should NOT import higher).
                """

                # Tier 0 - Pure Foundation
                CONSTANTS: Final[int] = 0
                TYPINGS: Final[int] = 0
                PROTOCOLS: Final[int] = 0

                # Tier 0.1 - Configuration
                CONFIG: Final[int] = 1

                # Tier 0.5 - Runtime
                RUNTIME: Final[int] = 2

                # Tier 1 - Core Abstractions
                EXCEPTIONS: Final[int] = 3
                RESULT: Final[int] = 3

                # Tier 1.5 - Logging
                LOGGINGS: Final[int] = 4

                # Tier 2 - Domain Foundation
                MODELS: Final[int] = 5
                UTILITIES: Final[int] = 5
                MIXINS: Final[int] = 5

                # Tier 2.5 - Domain + DI
                CONTAINER: Final[int] = 6
                SERVICE: Final[int] = 6
                CONTEXT: Final[int] = 6

                # Tier 3 - Application
                HANDLERS: Final[int] = 7
                DISPATCHER: Final[int] = 8
                REGISTRY: Final[int] = 8
                DECORATORS: Final[int] = 9

                @classmethod
                def as_dict(cls) -> dict[str, int]:
                    """Get layer hierarchy as dictionary.

                    Returns:
                        Mapping of module name to layer number.

                    """
                    return {
                        "constants": cls.CONSTANTS,
                        "typings": cls.TYPINGS,
                        "protocols": cls.PROTOCOLS,
                        "config": cls.CONFIG,
                        "runtime": cls.RUNTIME,
                        "exceptions": cls.EXCEPTIONS,
                        "result": cls.RESULT,
                        "loggings": cls.LOGGINGS,
                        "models": cls.MODELS,
                        "utilities": cls.UTILITIES,
                        "mixins": cls.MIXINS,
                        "container": cls.CONTAINER,
                        "service": cls.SERVICE,
                        "context": cls.CONTEXT,
                        "handlers": cls.HANDLERS,
                        "dispatcher": cls.DISPATCHER,
                        "registry": cls.REGISTRY,
                        "decorators": cls.DECORATORS,
                    }


c = FlextTestsConstants

# Type aliases for mypy resolution of deeply nested classes
# These help mypy resolve nested class types correctly
ContainerStatus = FlextTestsConstants.Tests.Docker.ContainerStatus
"""Type alias for ContainerStatus enum to help mypy resolution."""

__all__ = ["ContainerStatus", "FlextTestsConstants", "c"]
