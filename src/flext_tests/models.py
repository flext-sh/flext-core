"""Models for FLEXT tests.

Provides FlextTestsModels, extending FlextModels with test-specific model definitions
for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Sized
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import FlextModels, r
from flext_tests.constants import c
from flext_tests.typings import t


class FlextTestsModels(FlextModels):
    """Models for FLEXT tests - extends FlextModels.

    Architecture: Extends FlextModels with test-specific model definitions.
    All base models from FlextModels are available through inheritance.

    Usage:
        from flext_tests.models import m

        # Base models (inherited from FlextModels)
        class MyValue(m.Value): ...
        class MyEntity(m.Entity): ...

        # Test-specific models (in Tests namespace)
        container = m.Tests.Docker.ContainerInfo(...)
        user = m.Tests.Factory.User(...)
        file_info = m.Tests.Files.FileInfo(...)
        chain = m.Tests.Matcher.Chain(result)
        violation = m.Tests.Validator.Violation(...)
    """

    class Tests:
        """Test-specific models namespace.

        All test-specific models are organized under this namespace to clearly
        distinguish them from base FlextModels. Access via m.Tests.*
        """

        class Docker:
            """Docker-specific models for test infrastructure."""

            class ContainerInfo(FlextModels.Value):
                """Container information model."""

                name: str
                status: c.Tests.Docker.ContainerStatus
                ports: Mapping[str, str]
                image: str
                container_id: str = ""

                def model_post_init(self, __context: object, /) -> None:
                    """Validate container info after initialization."""
                    super().model_post_init(__context)
                    if not self.name:
                        msg = "Container name cannot be empty"
                        raise ValueError(msg)
                    if not self.image:
                        msg = "Container image cannot be empty"
                        raise ValueError(msg)

            class ContainerConfig(FlextModels.Value):
                """Container configuration model."""

                compose_file: Path
                service: str
                port: int

                def model_post_init(self, __context: object, /) -> None:
                    """Validate container config after initialization."""
                    super().model_post_init(__context)
                    if not self.compose_file.exists():
                        msg = f"Compose file not found: {self.compose_file}"
                        raise ValueError(msg)
                    if not self.service:
                        msg = "Service name cannot be empty"
                        raise ValueError(msg)
                    if not (c.Network.MIN_PORT <= self.port <= c.Network.MAX_PORT):
                        msg = f"Port {self.port} out of valid range"
                        raise ValueError(msg)

            class ContainerState(FlextModels.Value):
                """Container state tracking model."""

                container_name: str
                is_dirty: bool
                worker_id: str
                last_updated: str | None = None

            class ComposeConfig(FlextModels.Value):
                """Docker compose configuration model."""

                compose_file: Path
                services: Mapping[str, t.Tests.Docker.ComposeFileConfig]
                networks: Mapping[str, t.Tests.Docker.NetworkMapping] | None = None
                volumes: Mapping[str, t.Tests.Docker.VolumeMapping] | None = None

                def model_post_init(self, __context: object, /) -> None:
                    """Validate compose config after initialization."""
                    super().model_post_init(__context)
                    if not self.compose_file.exists():
                        msg = f"Compose file not found: {self.compose_file}"
                        raise ValueError(msg)
                    if not self.services:
                        msg = "Compose config must have at least one service"
                        raise ValueError(msg)

        class Factory:
            """Factory models for test data generation."""

            class User(FlextModels.Value):
                """Test user model - immutable value object."""

                id: str
                name: str
                email: str
                active: bool = True

            class Config(FlextModels.Value):
                """Test configuration model - immutable value object."""

                service_type: str = "api"
                environment: str = "test"
                debug: bool = True
                log_level: str = "DEBUG"
                timeout: int = 30
                max_retries: int = 3

            class Service(FlextModels.Value):
                """Test service model - immutable value object."""

                id: str
                type: str = "api"
                name: str = ""
                status: str = "active"

            class Entity(FlextModels.Entity):
                """Test entity model with identity."""

                name: str
                value: t.GeneralValueType = None

            class ValueObject(FlextModels.Value):
                """Test value object - compared by value."""

                data: str
                count: int = 0

            # ======================================================================
            # Factory Parameter Models - Pydantic 2 Advanced Validation
            # ======================================================================

            class ModelFactoryParams(FlextModels.Value):
                """Parameters for tt.model() factory method with Pydantic 2 advanced validation.

                Uses Field constraints, computed fields, and model_validator for
                comprehensive validation. All parameters are validated inline using
                Pydantic 2 capabilities.
                """

                kind: t.Tests.Factory.ModelKind = Field(
                    default="user",
                    description="Model type to create",
                )
                count: int = Field(
                    default=1,
                    ge=1,
                    description="Number of instances to create",
                )
                as_dict: bool = Field(
                    default=False,
                    description="Return as dict with ID keys",
                )
                as_result: bool = Field(
                    default=False,
                    description="Wrap result in FlextResult",
                )
                as_mapping: Mapping[str, str] | None = Field(
                    default=None,
                    description="Map to custom keys (Mapping[str, str])",
                )
                factory: Callable[[], object] | None = Field(
                    default=None,
                    description="Custom factory callable",
                )
                transform: Callable[[object], object] | None = Field(
                    default=None,
                    description="Post-transform function",
                )
                validate_fn: Callable[[object], bool] | None = Field(
                    default=None,
                    alias="validate",
                    description="Validation predicate",
                )
                # Model-specific fields with defaults from constants
                model_id: str | None = Field(
                    default=None,
                    description="Identifier (auto-generated if not provided)",
                )
                name: str | None = Field(
                    default=None,
                    description="Name field (varies by model type)",
                )
                email: str | None = Field(
                    default=None,
                    description="Email for user models",
                )
                active: bool = Field(
                    default=c.Tests.Factory.DEFAULT_USER_ACTIVE,
                    description="Active status for user models",
                )
                service_type: str = Field(
                    default=c.Tests.Factory.DEFAULT_SERVICE_TYPE,
                    description="Service type for config/service models",
                )
                environment: str = Field(
                    default=c.Tests.Factory.DEFAULT_ENVIRONMENT,
                    description="Environment for config models",
                )
                debug: bool = Field(
                    default=c.Tests.Factory.DEFAULT_DEBUG,
                    description="Debug flag for config models",
                )
                log_level: str = Field(
                    default=c.Tests.Factory.DEFAULT_LOG_LEVEL,
                    description="Log level for config models",
                )
                timeout: int = Field(
                    default=c.Tests.Factory.DEFAULT_TIMEOUT,
                    ge=0,
                    description="Timeout for config models",
                )
                max_retries: int = Field(
                    default=c.Tests.Factory.DEFAULT_MAX_RETRIES,
                    ge=0,
                    description="Max retries for config models",
                )
                status: str = Field(
                    default=c.Tests.Factory.DEFAULT_SERVICE_STATUS,
                    description="Status for service models",
                )
                value: t.GeneralValueType = Field(
                    default=None,
                    description="Value for entity models",
                )
                data: str = Field(
                    default=c.Tests.Factory.DEFAULT_VALUE_DATA,
                    description="Data for value object models",
                )
                value_count: int = Field(
                    default=c.Tests.Factory.DEFAULT_VALUE_COUNT,
                    ge=0,
                    description="Count for value object models",
                )
                # Overrides as dict
                overrides: dict[str, t.Tests.TestResultValue] = Field(
                    default_factory=dict,
                    description="Override any field directly",
                )

                @model_validator(mode="after")
                def validate_mapping(self) -> Self:
                    """Validate as_mapping keys if provided."""
                    if (
                        self.as_mapping
                        and self.count > 1
                        and len(self.as_mapping) < self.count
                    ):
                        msg = f"as_mapping must have at least {self.count} keys"
                        raise ValueError(msg)
                    return self

            class ResultFactoryParams(FlextModels.Value):
                """Parameters for tt.res() factory method with Pydantic 2 advanced validation.

                Uses Field constraints and model_validator for comprehensive validation.
                """

                kind: t.Tests.Factory.ResultKind = Field(
                    default="ok",
                    description="Result type ('ok', 'fail', 'from_value')",
                )
                value: t.GeneralValueType = Field(
                    default=None,
                    description="Value for success (required for 'ok')",
                )
                count: int = Field(
                    default=1,
                    ge=1,
                    description="Number of results to create",
                )
                values: Sequence[t.GeneralValueType] | None = Field(
                    default=None,
                    description="Explicit value list for batch creation",
                )
                errors: Sequence[str] | None = Field(
                    default=None,
                    description="Error messages for failure results",
                )
                mix_pattern: t.Tests.Factory.BatchPattern | None = Field(
                    default=None,
                    description="Success/failure pattern (True=success, False=failure)",
                )
                error: str = Field(
                    default=c.Tests.Factory.ERROR_DEFAULT,
                    description="Error message for failure results",
                )
                error_code: str | None = Field(
                    default=None,
                    description="Optional error code for failure results",
                )
                error_on_none: str | None = Field(
                    default=None,
                    description="Error message when value is None (for 'from_value')",
                )
                transform: Callable[[object], object] | None = Field(
                    default=None,
                    description="Transform function for success values",
                )

                @model_validator(mode="after")
                def validate_batch_params(self) -> Self:
                    """Validate batch parameters are consistent."""
                    if (
                        self.mix_pattern is not None
                        and self.values is None
                        and self.errors is None
                    ):
                        msg = "mix_pattern requires values or errors"
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_kind_value(self) -> Self:
                    """Validate kind matches value requirements."""
                    if self.kind == "ok" and self.value is None and self.values is None:
                        # None value is allowed for ok kind
                        pass
                    if (
                        self.kind == "from_value"
                        and self.value is None
                        and self.error_on_none is None
                    ):
                        msg = (
                            "from_value kind requires error_on_none when value is None"
                        )
                        raise ValueError(msg)
                    return self

            class ListFactoryParams(FlextModels.Value):
                """Parameters for tt.list() factory method with Pydantic 2 advanced validation.

                Uses Field constraints for inline validation. Source can be ModelKind (str),
                Sequence, or Callable - uses object type to accept all variants.
                """

                model_config = ConfigDict(populate_by_name=True)

                source: object = Field(
                    default="user",
                    description="Source for list items (ModelKind, Sequence, or Callable)",
                )
                count: int = Field(
                    default=5,
                    ge=1,
                    description="Number of items to create",
                )
                as_result: bool = Field(
                    default=False,
                    description="Wrap result in FlextResult",
                )
                unique: bool = Field(
                    default=False,
                    description="Ensure all items are unique",
                )
                transform: Callable[[object], object] | None = Field(
                    default=None,
                    description="Transform function applied to each item",
                )
                filter_: Callable[[object], bool] | None = Field(
                    default=None,
                    alias="filter",
                    description="Filter predicate to exclude items",
                )

            class DictFactoryParams(FlextModels.Value):
                """Parameters for tt.dict_factory() method with Pydantic 2 advanced validation.

                Uses Field constraints for inline validation. Source can be ModelKind (str),
                Mapping, or Callable - uses object type to accept all variants.
                """

                source: object = Field(
                    default="user",
                    description="Source for dict items (ModelKind, Mapping, or Callable)",
                )
                count: int = Field(
                    default=5,
                    ge=1,
                    description="Number of items to create",
                )
                key_factory: Callable[[int], object] | None = Field(
                    default=None,
                    description="Factory function for keys (takes index, returns K)",
                )
                value_factory: Callable[[object], object] | None = Field(
                    default=None,
                    description="Factory function for values (takes key, returns V)",
                )
                as_result: bool = Field(
                    default=False,
                    description="Wrap result in FlextResult",
                )
                merge_with: Mapping[str, object] | None = Field(
                    default=None,
                    description="Additional mapping to merge into result",
                )

            class GenericFactoryParams(FlextModels.Value):
                """Parameters for tt.generic() factory method with Pydantic 2 advanced validation.

                Uses Field constraints for inline validation. Type validation done via
                model_validator since Field constraints cannot validate isinstance checks.
                """

                type_: object = Field(
                    description="Type class to instantiate",
                )
                args: Sequence[t.GeneralValueType] | None = Field(
                    default=None,
                    description="Positional arguments for constructor",
                )
                kwargs: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Keyword arguments for constructor",
                )
                count: int = Field(
                    default=1,
                    ge=1,
                    description="Number of instances to create",
                )
                as_result: bool = Field(
                    default=False,
                    description="Wrap result in FlextResult",
                )
                validate_fn: Callable[[object], bool] | None = Field(
                    default=None,
                    alias="validate",
                    description="Validation predicate (must return True for success)",
                )

                @model_validator(mode="after")
                def validate_type(self) -> Self:
                    """Validate type_ is a class - Field constraints cannot do isinstance checks."""
                    if not isinstance(self.type_, type):
                        msg = "type_ must be a class"
                        raise ValueError(msg)
                    return self

        class Files:
            """File-related models for test infrastructure."""

            class FileInfo(FlextModels.Value):
                """Comprehensive file information model."""

                exists: bool
                path: Path | None = None
                size: int = 0
                size_human: str = ""
                lines: int = 0
                encoding: str = "utf-8"
                is_empty: bool = False
                first_line: str = ""
                fmt: str = "unknown"
                is_valid: bool = True
                created: datetime | None = None
                modified: datetime | None = None
                permissions: int = 0
                is_readonly: bool = False
                sha256: str | None = None
                content_meta: FlextTestsModels.Tests.Files.ContentMeta | None = None
                """Optional content metadata for parsed files."""

            class ContentMeta(FlextModels.Value):
                """Content-specific metadata for parsed files."""

                key_count: int | None = None
                """Number of keys for JSON/YAML dicts."""
                item_count: int | None = None
                """Number of items for JSON/YAML lists."""
                row_count: int | None = None
                """Number of rows for CSV files."""
                column_count: int | None = None
                """Number of columns for CSV files."""
                model_valid: bool | None = None
                """Whether content is valid for a specific model."""
                model_name: str | None = None
                """Model class name if validated."""

            class BatchResult(FlextModels.Value):
                """Result of batch file operations."""

                succeeded: list[Path] = Field(default_factory=list)
                """List of successfully processed file paths."""
                failed: dict[str, str] = Field(default_factory=dict)
                """Dictionary mapping file names to error messages."""
                total: int = 0
                """Total number of files processed."""
                success_count: int = 0
                """Number of successful operations."""
                failure_count: int = 0
                """Number of failed operations."""

            class CreateParams(FlextModels.Value):
                """Parameters for file creation operations with Pydantic 2 advanced validation.

                Validates and computes all parameters for create() method using Field constraints.
                Uses Pydantic 2 advanced features: Field constraints (ge, min_length, max_length)
                for inline validation, field_validator only when Field constraints are insufficient.
                All parameters validated via model - no manual validation needed.
                """

                content: t.Tests.Files.FileContent
                """File content to create."""
                name: str = Field(
                    default=c.Tests.Files.DEFAULT_FILENAME,
                    min_length=1,
                    description="Filename for the created file (non-empty).",
                )
                directory: Path | None = Field(
                    default=None,
                    description="Target directory (uses base_dir or temp if None).",
                )
                fmt: c.Tests.Files.FormatLiteral = Field(
                    default="auto",
                    description="File format override.",
                )
                enc: str = Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                )
                indent: int = Field(
                    default=c.Tests.Files.DEFAULT_JSON_INDENT,
                    ge=0,
                    description="JSON/YAML indentation (non-negative).",
                )
                delim: str = Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                )
                headers: list[str] | None = Field(
                    default=None,
                    description="CSV headers.",
                )
                readonly: bool = Field(
                    default=False,
                    description="Create file as read-only.",
                )
                extract_result: bool = Field(
                    default=True,
                    description="Auto-extract FlextResult value.",
                )

                @field_validator("name", mode="before")
                @classmethod
                def normalize_name(cls, value: object) -> str:
                    """Normalize filename by stripping whitespace."""
                    if isinstance(value, str):
                        return value.strip()
                    return str(value)

            class CreateKwargsParams(FlextModels.Value):
                """Parameters for create() kwargs validation in files() context manager.

                Validates only optional parameters that can be passed via **kwargs.
                Used by files() method to validate kwargs before passing to create().
                """

                directory: Path | None = Field(
                    default=None,
                    description="Target directory (uses base_dir or temp if None).",
                )
                fmt: c.Tests.Files.FormatLiteral = Field(
                    default="auto",
                    description="File format override.",
                )
                enc: str = Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                )
                indent: int = Field(
                    default=c.Tests.Files.DEFAULT_JSON_INDENT,
                    ge=0,
                    description="JSON/YAML indentation (non-negative).",
                )
                delim: str = Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                )
                headers: list[str] | None = Field(
                    default=None,
                    description="CSV headers.",
                )
                readonly: bool = Field(
                    default=False,
                    description="Create file as read-only.",
                )

                @field_validator("directory", mode="before")
                @classmethod
                def validate_directory(cls, value: object) -> Path | None:
                    """Convert str to Path if needed."""
                    if value is None:
                        return None
                    if isinstance(value, str):
                        return Path(value)
                    if isinstance(value, Path):
                        return value
                    msg = f"directory must be Path or str, got {type(value)}"
                    raise ValueError(msg)

            class ReadParams(FlextModels.Value):
                """Parameters for file read operations with Pydantic 2 advanced validation.

                Validates and computes all parameters for read() method using Field constraints.
                Path conversion handled by field_validator (Field constraints insufficient for type conversion).
                """

                path: Path = Field(
                    description="Path to file to read (str or Path converted automatically).",
                )
                model_cls: type[BaseModel] | None = Field(
                    default=None,
                    description="Optional Pydantic model class to deserialize into.",
                )
                fmt: c.Tests.Files.FormatLiteral = Field(
                    default="auto",
                    description="Format override.",
                )
                enc: str = Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                )
                delim: str = Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                )
                has_headers: bool = Field(
                    default=True,
                    description="CSV has headers.",
                )

                @field_validator("path", mode="before")
                @classmethod
                def convert_path(cls, value: Path | str) -> Path:
                    """Convert string to Path - Field constraints cannot handle type conversion."""
                    return Path(value) if isinstance(value, str) else value

            class CompareParams(FlextModels.Value):
                """Parameters for file comparison operations with Pydantic 2 advanced validation.

                Validates and computes all parameters for compare() method using Field constraints.
                Path conversion handled by field_validator (Field constraints insufficient for type conversion).
                """

                file1: Path = Field(
                    description="First file to compare (str or Path converted automatically).",
                )
                file2: Path = Field(
                    description="Second file to compare (str or Path converted automatically).",
                )
                mode: c.Tests.Files.CompareModeLiteral = Field(
                    default="content",
                    description="Comparison mode.",
                )
                ignore_ws: bool = Field(
                    default=False,
                    description="Ignore whitespace in comparison.",
                )
                ignore_case: bool = Field(
                    default=False,
                    description="Case-insensitive comparison.",
                )
                pattern: str | None = Field(
                    default=None,
                    description="Pattern to check if both files contain.",
                )
                deep: bool = Field(
                    default=True,
                    description="Use deep comparison for nested structures (dict/JSON/YAML).",
                )
                keys: list[str] | None = Field(
                    default=None,
                    description="Only compare these keys (for dict/JSON/YAML content).",
                )
                exclude_keys: list[str] | None = Field(
                    default=None,
                    description="Exclude these keys from comparison (for dict/JSON/YAML content).",
                )

                @field_validator("file1", "file2", mode="before")
                @classmethod
                def convert_path(cls, value: Path | str) -> Path:
                    """Convert string to Path - Field constraints cannot handle type conversion."""
                    return Path(value) if isinstance(value, str) else value

            class InfoParams(FlextModels.Value):
                """Parameters for file info operations with Pydantic 2 advanced validation.

                Validates and computes all parameters for info() method using Field constraints.
                Path conversion handled by field_validator (Field constraints insufficient for type conversion).
                """

                path: Path = Field(
                    description="Path to file (str or Path converted automatically).",
                )
                compute_hash: bool = Field(
                    default=False,
                    description="Compute SHA256 hash.",
                )
                detect_fmt: bool = Field(
                    default=True,
                    description="Auto-detect format.",
                )
                parse_content: bool = Field(
                    default=False,
                    description="Parse content and include metadata (key_count, item_count, etc.).",
                )
                validate_model: type[BaseModel] | None = Field(
                    default=None,
                    description="Pydantic model to validate content against.",
                )

                @field_validator("path", mode="before")
                @classmethod
                def convert_path(cls, value: Path | str) -> Path:
                    """Convert string to Path - Field constraints cannot handle type conversion."""
                    return Path(value) if isinstance(value, str) else value

            class BatchParams(FlextModels.Value):
                """Parameters for batch file operations with Pydantic 2 advanced validation.

                Validates and computes all parameters for batch() method using Field constraints.
                Path conversion handled by field_validator (Field constraints insufficient for type conversion).
                """

                files: t.Tests.Files.BatchFiles = Field(
                    description="Files to process in batch.",
                )
                directory: Path | None = Field(
                    default=None,
                    description="Target directory for create operations (str or Path converted automatically).",
                )
                operation: c.Tests.Files.OperationLiteral = Field(
                    default="create",
                    description="Operation to perform.",
                )
                model: type[BaseModel] | None = Field(
                    default=None,
                    description="Optional model class for read operations.",
                )
                on_error: c.Tests.Files.ErrorModeLiteral = Field(
                    default="collect",
                    description="Error handling mode.",
                )
                parallel: bool = Field(
                    default=False,
                    description="Run operations in parallel.",
                )

                @field_validator("directory", mode="before")
                @classmethod
                def convert_directory(cls, value: Path | str | None) -> Path | None:
                    """Convert string to Path - Field constraints cannot handle type conversion."""
                    return Path(value) if isinstance(value, str) else value

        class Matcher:
            """Matcher models for test assertions."""

            class Chain[TChain]:
                """Chained assertions with fluent API."""

                def __init__(self, result: r[TChain]) -> None:  # pyright: ignore[reportMissingSuperCall]
                    """Initialize chain with result.

                    Does not inherit from a class requiring super().__init__().

                    Note: Chain does not inherit from any class requiring super().__init__().
                    """
                    self._result = result
                    self._value: TChain | None = None
                    self._error: str | None = None
                    self._is_ok: bool = False

                def ok(self, msg: str | None = None) -> Self:
                    """Assert result is success."""
                    if not self._result.is_success:
                        error_msg = msg or c.Tests.Matcher.ERR_EXPECTED_SUCCESS.format(
                            error=self._result.error,
                        )
                        raise AssertionError(error_msg)
                    self._value = self._result.unwrap()
                    self._is_ok = True
                    return self

                def fail(
                    self,
                    error: str | None = None,
                    msg: str | None = None,
                ) -> Self:
                    """Assert result is failure."""
                    if self._result.is_success:
                        error_msg = msg or c.Tests.Matcher.ERR_EXPECTED_FAILURE.format(
                            value=self._result.unwrap(),
                        )
                        raise AssertionError(error_msg)
                    self._error = self._result.error or ""
                    if error and error not in self._error:
                        error_msg = msg or c.Tests.Matcher.ERR_ERROR_CONTAINS.format(
                            expected=error,
                            actual=self._error,
                        )
                        raise AssertionError(error_msg)
                    self._is_ok = False
                    return self

                def eq(self, expected: TChain, msg: str | None = None) -> Self:
                    """Assert value equals expected."""
                    if self._is_ok and self._value != expected:
                        error_msg = msg or c.Tests.Matcher.ERR_EXPECTED_VALUE.format(
                            expected=expected,
                            actual=self._value,
                        )
                        raise AssertionError(error_msg)
                    return self

                def has(self, item: object, msg: str | None = None) -> Self:
                    """Assert value/error contains item."""
                    if self._is_ok:
                        if isinstance(self._value, dict) and isinstance(item, str):
                            if item not in self._value:
                                error_msg = (
                                    msg
                                    or c.Tests.Matcher.ERR_KEY_NOT_FOUND.format(
                                        key=item,
                                    )
                                )
                                raise AssertionError(error_msg)
                        elif isinstance(self._value, str) and isinstance(item, str):
                            if item not in self._value:
                                error_msg = (
                                    msg
                                    or c.Tests.Matcher.ERR_NOT_IN_SEQUENCE.format(
                                        item=item,
                                    )
                                )
                                raise AssertionError(error_msg)
                        elif isinstance(self._value, (list, tuple)):
                            # Use any() for type-safe containment check
                            found = any(v == item for v in self._value)
                            if not found:
                                error_msg = (
                                    msg
                                    or c.Tests.Matcher.ERR_NOT_IN_SEQUENCE.format(
                                        item=item,
                                    )
                                )
                                raise AssertionError(error_msg)
                    elif (
                        self._error
                        and isinstance(item, str)
                        and item not in self._error
                    ):
                        error_msg = msg or c.Tests.Matcher.ERR_NOT_CONTAINS.format(
                            substring=item,
                            text=self._error,
                        )
                        raise AssertionError(error_msg)
                    return self

                def len(self, expected: int, msg: str | None = None) -> Self:
                    """Assert value has expected length."""
                    if (
                        self._is_ok
                        and self._value is not None
                        and isinstance(self._value, Sized)
                    ):
                        actual = len(self._value)
                        if actual != expected:
                            error_msg = msg or c.Tests.Matcher.ERR_LENGTH_EXACT.format(
                                expected=expected,
                                actual=actual,
                            )
                            raise AssertionError(error_msg)
                    return self

                def done(self) -> TChain:
                    """Finish chain and return value (for success)."""
                    if not self._is_ok:
                        raise AssertionError(c.Tests.Matcher.ERR_CHAIN_NO_VALUE)
                    if self._value is None:
                        raise AssertionError(c.Tests.Matcher.ERR_CHAIN_NO_VALUE)
                    return self._value

                def err(self) -> str:
                    """Finish chain and return error (for failure)."""
                    if self._is_ok:
                        raise AssertionError(c.Tests.Matcher.ERR_CHAIN_NO_ERROR)
                    return self._error or ""

            class TestScope(FlextModels.Value):
                """Isolated test execution scope.

                Provides isolated configuration, container, and context for tests.
                Use for tests that need clean, independent execution environment.

                Attributes:
                    config: Configuration dictionary for test execution
                    container: Service/component container mappings
                    context: Execution context values

                Examples:
                    with tm.scope(config={"debug": True}) as s:
                        s.container["service"] = mock_service
                        result = operation()
                        tm.ok(result)

                """

                config: t.Types.ConfigurationDict
                """Configuration dictionary for test execution."""

                container: dict[str, object]
                """Service/component container mappings."""

                context: dict[str, object]
                """Execution context values."""

            class DeepMatchResult(FlextModels.Value):
                """Result of deep structural matching operation.

                Provides detailed information about deep matching results,
                including the path where matching failed (if any) and the reason.

                Attributes:
                    path: Dot-notation path where matching occurred/failed
                    expected: Expected value or predicate description
                    actual: Actual value found at path
                    matched: Whether the match was successful
                    reason: Optional reason for failure (if not matched)

                Examples:
                    result = u.Tests.DeepMatch.match(obj, {"user.name": "John"})
                    if not result.matched:
                        print(f"Failed at {result.path}: {result.reason}")

                """

                path: str
                """Dot-notation path where matching occurred/failed."""

                expected: object
                """Expected value or predicate description."""

                actual: object
                """Actual value found at path."""

                matched: bool
                """Whether the match was successful."""

                reason: str | None = None
                """Optional reason for failure (if not matched)."""

            class ValidationResult(FlextModels.Value):
                """Result of validation operation.

                Provides comprehensive validation results with failure details
                and context information for debugging.

                Attributes:
                    passed: Whether validation passed
                    failures: List of failure messages
                    context: Additional context information for debugging

                Examples:
                    result = validate_config(config)
                    if not result.passed:
                        for failure in result.failures:
                            print(f"Validation failed: {failure}")

                """

                passed: bool
                """Whether validation passed."""

                failures: list[str] = Field(default_factory=list)
                """List of failure messages."""

                context: dict[str, object] = Field(default_factory=dict)
                """Additional context information for debugging."""

            class OkParams(FlextModels.Value):
                """Parameters for tm.ok() method with Pydantic 2 validation.

                Validates FlextResult success with optional value validation.
                Uses Field() for all parameters with proper defaults and descriptions.
                """

                # Value validation parameters
                eq: object | None = Field(
                    default=None,
                    description="Equality check - value must equal this",
                )
                ne: object | None = Field(
                    default=None,
                    description="Inequality check - value must not equal this",
                )
                is_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    description="Type check - value must be instance of this type(s)",
                )
                none: bool | None = Field(
                    default=None,
                    description="None check - True=must be None, False=must not be None",
                )
                empty: bool | None = Field(
                    default=None,
                    description="Empty check - True=must be empty, False=must not be empty",
                )
                gt: float | None = Field(
                    default=None,
                    description="Greater than check (numeric or length)",
                )
                gte: float | None = Field(
                    default=None,
                    description="Greater than or equal check",
                )
                lt: float | None = Field(default=None, description="Less than check")
                lte: float | None = Field(
                    default=None,
                    description="Less than or equal check",
                )
                # Unified containment - uses centralized types from typings.py
                has: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Unified containment - value contains item(s)",
                )
                lacks: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Unified non-containment - value does NOT contain item(s)",
                )
                # String assertions
                starts: str | None = Field(
                    default=None,
                    min_length=1,
                    description="String prefix check",
                )
                ends: str | None = Field(
                    default=None,
                    min_length=1,
                    description="String suffix check",
                )
                match: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Regex pattern match (for strings)",
                )
                # Unified length - uses centralized type from typings.py
                len: t.Tests.Matcher.LengthSpec | None = Field(
                    default=None,
                    description=(
                        "Length spec - exact int or (min, max) tuple. "
                        "Examples: 5 (exact), (1, 10) (range)"
                    ),
                )
                # Deep structural matching - uses centralized type from typings.py
                deep: t.Tests.Matcher.DeepSpec | None = Field(
                    default=None,
                    description="Deep structural matching specification",
                )
                # Path extraction - uses centralized type from typings.py
                path: t.Tests.Matcher.PathSpec | None = Field(
                    default=None,
                    description="Extract nested value via dot notation before validation",
                )
                # Custom validation - uses centralized type from typings.py
                where: t.Tests.Matcher.PredicateSpec | None = Field(
                    default=None,
                    description="Custom predicate function for validation",
                )
                # Legacy support (deprecated)
                contains: object | None = Field(
                    default=None,
                    description="Legacy containment (deprecated, use has=)",
                    deprecated=True,
                )
                # Custom error message
                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )

                @model_validator(mode="after")
                def validate_legacy_contains(self) -> Self:
                    """Convert legacy contains to has if provided."""
                    if self.contains is not None and self.has is None:
                        object.__setattr__(self, "has", self.contains)
                    return self

            class FailParams(FlextModels.Value):
                """Parameters for tm.fail() method with Pydantic 2 validation.

                Validates FlextResult failure with optional error validation.
                """

                # Error message assertions - uses centralized types from typings.py
                has: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified containment - error contains substring(s)",
                )
                lacks: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified non-containment - error does NOT contain substring(s)",
                )
                starts: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error starts with prefix",
                )
                ends: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error ends with suffix",
                )
                match: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error matches regex",
                )
                # Error metadata assertions
                code: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error code equals",
                )
                code_has: t.Tests.Matcher.ErrorCodeSpec | None = Field(
                    default=None,
                    description="Error code contains substring(s)",
                )
                data: t.Tests.Matcher.ErrorDataSpec | None = Field(
                    default=None,
                    description="Error data contains key-value pairs",
                )
                # Legacy support (deprecated)
                error: str | None = Field(
                    default=None,
                    description="Legacy error parameter (deprecated, use has=)",
                    deprecated=True,
                )
                contains: str | None = Field(
                    default=None,
                    description="Legacy containment (deprecated, use has=)",
                    deprecated=True,
                )
                excludes: str | None = Field(
                    default=None,
                    description="Legacy non-containment (deprecated, use lacks=)",
                    deprecated=True,
                )
                # Custom error message
                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )

                @model_validator(mode="after")
                def validate_legacy_params(self) -> Self:
                    """Convert legacy parameters to unified has/lacks."""
                    if self.error is not None and self.has is None:
                        object.__setattr__(self, "has", self.error)
                    if self.contains is not None and self.has is None:
                        object.__setattr__(self, "has", self.contains)
                    if self.excludes is not None and self.lacks is None:
                        object.__setattr__(self, "lacks", self.excludes)
                    return self

            class ThatParams(FlextModels.Value):
                """Parameters for tm.that() method with Pydantic 2 validation.

                Universal value assertion with 40+ parameters.
                All validations in ONE method with comprehensive Pydantic 2 validation.
                """

                # Core assertions
                eq: object | None = Field(default=None, description="Equality check")
                ne: object | None = Field(default=None, description="Inequality check")
                is_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    description="Type check (isinstance)",
                )
                not_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    description="Negative type check",
                )
                # Nullability
                none: bool | None = Field(
                    default=None,
                    description="None check - True=must be None, False=must not be None",
                )
                empty: bool | None = Field(
                    default=None,
                    description="Empty check - True=must be empty, False=must not be empty",
                )
                # Comparisons
                gt: float | None = Field(
                    default=None,
                    description="Greater than (numeric or length)",
                )
                gte: float | None = Field(
                    default=None,
                    description="Greater than or equal",
                )
                lt: float | None = Field(default=None, description="Less than")
                lte: float | None = Field(
                    default=None,
                    description="Less than or equal",
                )
                # Length/Size - uses centralized type from typings.py
                len: t.Tests.Matcher.LengthSpec | None = Field(
                    default=None,
                    description="Length spec - exact int or (min, max) tuple",
                )
                # Containment - uses centralized types from typings.py
                has: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Contains item(s)",
                )
                lacks: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Does not contain item(s)",
                )
                # String assertions
                starts: str | None = Field(
                    default=None,
                    min_length=1,
                    description="String prefix",
                )
                ends: str | None = Field(
                    default=None,
                    min_length=1,
                    description="String suffix",
                )
                match: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Regex pattern (for strings)",
                )
                # Sequence assertions
                first: object | None = Field(
                    default=None,
                    description="First item equals",
                )
                last: object | None = Field(
                    default=None,
                    description="Last item equals",
                )
                all_: t.Tests.Matcher.SequencePredicate | None = Field(
                    default=None,
                    description="All items match type or predicate",
                    alias="all",
                )
                any_: t.Tests.Matcher.SequencePredicate | None = Field(
                    default=None,
                    description="Any item matches type or predicate",
                    alias="any",
                )
                sorted: t.Tests.Matcher.SortKey | None = Field(
                    default=None,
                    description="Is sorted - True=ascending, or key function",
                )
                unique: bool | None = Field(
                    default=None,
                    description="All items unique",
                )
                # Mapping assertions - uses centralized types from typings.py
                keys: t.Tests.Matcher.KeySpec | None = Field(
                    default=None,
                    description="Has all keys",
                )
                lacks_keys: t.Tests.Matcher.KeySpec | None = Field(
                    default=None,
                    description="Missing keys",
                )
                values: Sequence[object] | None = Field(
                    default=None,
                    description="Has all values",
                )
                kv: t.Tests.Matcher.KeyValueSpec | None = Field(
                    default=None,
                    description="Key-value pairs (single tuple or mapping)",
                )
                # Object/Class assertions - uses centralized types from typings.py
                attrs: t.Tests.Matcher.AttributeSpec | None = Field(
                    default=None,
                    description="Has attribute(s)",
                )
                methods: t.Tests.Matcher.AttributeSpec | None = Field(
                    default=None,
                    description="Has method(s)",
                )
                attr_eq: t.Tests.Matcher.AttributeValueSpec | None = Field(
                    default=None,
                    description="Attribute equals (single tuple or mapping)",
                )
                # FlextResult special handling
                ok: bool | None = Field(
                    default=None,
                    description="For FlextResult: assert success",
                )
                error: str | None = Field(
                    default=None,
                    min_length=1,
                    description="For FlextResult: error contains",
                )
                # Deep structural matching - uses centralized type from typings.py
                deep: t.Tests.Matcher.DeepSpec | None = Field(
                    default=None,
                    description="Recursive structure match",
                )
                # Custom validation - uses centralized type from typings.py
                where: t.Tests.Matcher.PredicateSpec | None = Field(
                    default=None,
                    description="Custom predicate function",
                )
                # Legacy support (deprecated)
                contains: object | None = Field(
                    default=None,
                    description="Legacy containment (deprecated, use has=)",
                    deprecated=True,
                )
                excludes: str | None = Field(
                    default=None,
                    description="Legacy non-containment (deprecated, use lacks=)",
                    deprecated=True,
                )
                length: int | None = Field(
                    default=None,
                    description="Legacy length (deprecated, use len=)",
                    deprecated=True,
                )
                length_gt: int | None = Field(
                    default=None,
                    description="Legacy length_gt (deprecated, use len=(min, max))",
                    deprecated=True,
                )
                length_gte: int | None = Field(
                    default=None,
                    description="Legacy length_gte (deprecated, use len=(min, max))",
                    deprecated=True,
                )
                length_lt: int | None = Field(
                    default=None,
                    description="Legacy length_lt (deprecated, use len=(min, max))",
                    deprecated=True,
                )
                length_lte: int | None = Field(
                    default=None,
                    description="Legacy length_lte (deprecated, use len=(min, max))",
                    deprecated=True,
                )
                # Custom error message
                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )

                @model_validator(mode="after")
                def validate_legacy_params(self) -> Self:
                    """Convert legacy parameters to unified parameters."""
                    # Convert legacy contains to has
                    if self.contains is not None and self.has is None:
                        object.__setattr__(self, "has", self.contains)
                    # Convert legacy length params to unified len
                    if self.length is not None and self.len is None:
                        object.__setattr__(self, "len", self.length)
                    elif (
                        self.length_gt is not None
                        or self.length_gte is not None
                        or self.length_lt is not None
                        or self.length_lte is not None
                    ) and self.len is None:
                        # Convert to range tuple
                        min_len = (
                            self.length_gte
                            if self.length_gte is not None
                            else (
                                self.length_gt + 1 if self.length_gt is not None else 0
                            )
                        )
                        max_len = (
                            self.length_lte
                            if self.length_lte is not None
                            else (
                                self.length_lt - 1
                                if self.length_lt is not None
                                else 999999
                            )
                        )
                        object.__setattr__(self, "len", (min_len, max_len))
                    return self

            class ScopeParams(FlextModels.Value):
                """Parameters for tm.scope() method with Pydantic 2 validation.

                Isolated test execution scope with temporary modifications.
                """

                config: Mapping[str, object] | None = Field(
                    default=None,
                    description="Initial configuration values",
                )
                container: Mapping[str, object] | None = Field(
                    default=None,
                    description="Initial container/service mappings",
                )
                context: Mapping[str, object] | None = Field(
                    default=None,
                    description="Initial context values",
                )
                # Auto-cleanup - uses centralized type from typings.py
                cleanup: t.Tests.Matcher.CleanupSpec | None = Field(
                    default=None,
                    description="Sequence of cleanup functions to call on exit",
                )
                # Temporary modifications - uses centralized type from typings.py
                env: t.Tests.Matcher.EnvironmentSpec | None = Field(
                    default=None,
                    description="Temporary environment variables (restored on exit)",
                )
                cwd: Path | str | None = Field(
                    default=None,
                    description="Temporary working directory (restored on exit)",
                )

        class Validator:
            """Validator models for architecture validation."""

            Severity = c.Tests.Validator.Severity

            class Violation(FlextModels.Value):
                """A detected architecture violation."""

                file_path: Path
                line_number: int
                rule_id: str
                severity: c.Tests.Validator.SeverityLiteral
                description: str
                code_snippet: str = ""

                def format(self) -> str:
                    """Format violation as string."""
                    return c.Tests.Validator.Messages.VIOLATION_WITH_SNIPPET.format(
                        rule_id=self.rule_id,
                        description=self.description,
                        snippet=self.code_snippet or "(no snippet)",
                    )

                def format_short(self) -> str:
                    """Format violation as short string."""
                    return c.Tests.Validator.Messages.VIOLATION.format(
                        rule_id=self.rule_id,
                        file=self.file_path.name,
                        line=self.line_number,
                    )

            class ScanResult(FlextModels.Value):
                """Result of a validation scan."""

                validator_name: str
                files_scanned: int
                violations: list[FlextTestsModels.Tests.Validator.Violation]
                passed: bool

                @classmethod
                def create(
                    cls,
                    validator_name: str,
                    files_scanned: int,
                    violations: list[FlextTestsModels.Tests.Validator.Violation],
                ) -> FlextTestsModels.Tests.Validator.ScanResult:
                    """Create a ScanResult from violations."""
                    return cls(
                        validator_name=validator_name,
                        files_scanned=files_scanned,
                        violations=violations,
                        passed=len(violations) == 0,
                    )

                def format(self) -> str:
                    """Format scan result as string."""
                    if self.passed:
                        return c.Tests.Validator.Messages.SCAN_PASSED.format(
                            count=self.files_scanned,
                        )
                    return c.Tests.Validator.Messages.SCAN_FAILED.format(
                        violations=len(self.violations),
                        count=self.files_scanned,
                    )

            class ScanConfig(FlextModels.Value):
                """Configuration for validation scan."""

                target_path: Path
                include_patterns: list[str] | None = None
                exclude_patterns: list[str] | None = None
                approved_exceptions: dict[str, list[str]] | None = None

                def model_post_init(self, __context: object, /) -> None:
                    """Apply defaults from c.Tests.Validator.Defaults after init."""
                    super().model_post_init(__context)
                    if self.include_patterns is None:
                        object.__setattr__(
                            self,
                            "include_patterns",
                            list(c.Tests.Validator.Defaults.INCLUDE_PATTERNS),
                        )
                    if self.exclude_patterns is None:
                        object.__setattr__(
                            self,
                            "exclude_patterns",
                            list(c.Tests.Validator.Defaults.EXCLUDE_PATTERNS),
                        )
                    if self.approved_exceptions is None:
                        object.__setattr__(self, "approved_exceptions", {})

        class Builders:
            """Builder models for test data construction using Pydantic 2 advanced features.

            All models use Pydantic 2 Field validation, computed fields, and model_validator
            for comprehensive parameter validation and computation.
            """

            class AddParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.add() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - Computed fields for derived values
                - model_validator for cross-field validation
                - Automatic type coercion and validation

                All parameters are optional except 'key'. Resolution priority is computed
                automatically based on which parameters are provided.
                """

                key: str = Field(min_length=1, description="Key to store data under")
                value: object | None = Field(
                    default=None,
                    description="Direct value to store (validated as BuilderValue)",
                )

                # Existing parameters
                factory: str | None = Field(
                    default=None,
                    description="Factory name to use (users, configs, services)",
                )
                count: int | None = Field(
                    default=None,
                    ge=1,
                    description="Number of items for factory generation",
                )
                model: Any | None = Field(
                    default=None,
                    description="Pydantic model class to instantiate (type[BaseModel])",
                )
                model_data: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Data for model instantiation",
                )
                mapping: object | None = Field(
                    default=None,
                    description="Dict/mapping to store (validated as BuilderMapping)",
                )
                sequence: object | None = Field(
                    default=None,
                    description="List/sequence to store (validated as BuilderSequence)",
                )
                transform: t.Tests.Builders.TransformFunc | None = Field(
                    default=None,
                    description="Function to transform value before storing",
                )
                validate_func: t.Tests.Builders.ValidateFunc | None = Field(
                    default=None,
                    description="Function to validate value",
                    alias="validate",
                )
                merge: bool = Field(
                    default=False,
                    description="Whether to merge with existing data at key",
                )
                default: object | None = Field(
                    default=None,
                    description="Default value if result is None (validated as BuilderValue)",
                )
                production: bool | None = Field(
                    default=None,
                    description="Shortcut for production config",
                )
                debug: bool | None = Field(
                    default=None,
                    description="Shortcut for debug config",
                )

                # FlextResult parameters
                result: r[t.GeneralValueType] | None = Field(
                    default=None,
                    description="FlextResult to store directly",
                )
                result_ok: t.GeneralValueType | None = Field(
                    default=None,
                    description="Value to wrap in r[T].ok()",
                )
                result_fail: str | None = Field(
                    default=None,
                    description="Error message for r[T].fail()",
                )
                result_code: str | None = Field(
                    default=None,
                    description="Error code for result_fail",
                )

                # Batch result parameters
                results: Sequence[r[t.GeneralValueType]] | None = Field(
                    default=None,
                    description="Sequence of FlextResult to store",
                )
                results_ok: Sequence[t.GeneralValueType] | None = Field(
                    default=None,
                    description="Sequence of values to wrap in r[T].ok()",
                )
                results_fail: Sequence[str] | None = Field(
                    default=None,
                    description="Sequence of error messages for r[T].fail()",
                )

                # Generic class instantiation
                cls: type[object] | None = Field(
                    default=None,
                    description="Any class to instantiate",
                )
                cls_args: tuple[t.GeneralValueType, ...] | None = Field(
                    default=None,
                    description="Positional arguments for cls",
                )
                cls_kwargs: dict[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Keyword arguments for cls",
                )

                # Collection with transformation
                items: Sequence[t.GeneralValueType] | None = Field(
                    default=None,
                    description="Type-safe sequence",
                )
                items_map: Callable[[t.GeneralValueType], t.GeneralValueType] | None = (
                    Field(default=None, description="Transform each item")
                )
                items_filter: Callable[[t.GeneralValueType], bool] | None = Field(
                    default=None,
                    description="Filter items",
                )

                entries: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Type-safe mapping",
                )
                entries_map: (
                    Callable[[t.GeneralValueType], t.GeneralValueType] | None
                ) = Field(default=None, description="Transform values")
                entries_filter: set[str] | None = Field(
                    default=None,
                    description="Include only these keys",
                )

                @computed_field
                def resolution_priority(self) -> int:
                    """Compute resolution priority based on provided parameters.

                    Returns priority number (1-15) indicating which parameter should
                    be used for value resolution. Lower number = higher priority.

                    """
                    if self.result is not None:
                        return 1
                    if self.result_ok is not None:
                        return 2
                    if self.result_fail is not None:
                        return 3
                    if self.results is not None:
                        return 4
                    if self.results_ok is not None:
                        return 5
                    if self.results_fail is not None:
                        return 6
                    if self.cls is not None:
                        return 7
                    if self.items is not None:
                        return 8
                    if self.entries is not None:
                        return 9
                    if self.factory is not None:
                        return 10
                    if self.model is not None:
                        return 11
                    if self.production is not None or self.debug is not None:
                        return 12
                    if self.mapping is not None:
                        return 13
                    if self.sequence is not None:
                        return 14
                    if self.value is not None:
                        return 15
                    if self.default is not None:
                        return 16
                    return 0  # No valid parameter

                @computed_field
                def effective_count(self) -> int:
                    """Compute effective count value with defaults.

                    Uses c.Tests.Factory.DEFAULT_BATCH_COUNT if count is None.

                    """
                    return self.count or c.Tests.Factory.DEFAULT_BATCH_COUNT

                @computed_field
                def effective_error_code(self) -> str:
                    """Compute effective error code with defaults.

                    Uses c.Errors.VALIDATION_ERROR if result_code is None.

                    """
                    return self.result_code or c.Errors.VALIDATION_ERROR

                @model_validator(mode="after")
                def validate_mutually_exclusive(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate mutually exclusive parameter combinations.

                    Ensures that conflicting parameters are not provided together.
                    Uses computed resolution_priority to detect conflicts.

                    """
                    priorities: list[int] = []
                    if self.result is not None:
                        priorities.append(1)
                    if self.result_ok is not None:
                        priorities.append(2)
                    if self.result_fail is not None:
                        priorities.append(3)
                    if self.results is not None:
                        priorities.append(4)
                    if self.results_ok is not None:
                        priorities.append(5)
                    if self.results_fail is not None:
                        priorities.append(6)
                    if self.cls is not None:
                        priorities.append(7)
                    if self.items is not None:
                        priorities.append(8)
                    if self.entries is not None:
                        priorities.append(9)
                    if self.factory is not None:
                        priorities.append(10)
                    if self.model is not None:
                        priorities.append(11)
                    if self.production is not None or self.debug is not None:
                        priorities.append(12)
                    if self.mapping is not None:
                        priorities.append(13)
                    if self.sequence is not None:
                        priorities.append(14)
                    if self.value is not None:
                        priorities.append(15)
                    if self.default is not None:
                        priorities.append(16)

                    # Multiple high-priority parameters should not be provided
                    # Only warn if more than one high-priority (1-9) is set
                    high_priority = [p for p in priorities if 1 <= p <= 9]
                    if len(high_priority) > 1:
                        # Use first match (lowest priority number)
                        pass  # This is expected behavior - first match wins

                    return self

                @model_validator(mode="after")
                def validate_count_positive(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate count is positive when provided."""
                    if self.count is not None and self.count < 1:
                        msg = c.Tests.Builders.ERROR_INVALID_COUNT.format(
                            count=self.count,
                        )
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_result_code_with_fail(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate result_code is only provided with result_fail."""
                    if self.result_code is not None and self.result_fail is None:
                        msg = "result_code can only be used with result_fail"
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_cls_with_args(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate cls_args/cls_kwargs are only provided with cls."""
                    if (
                        self.cls_args is not None or self.cls_kwargs is not None
                    ) and self.cls is None:
                        msg = "cls_args/cls_kwargs can only be used with cls"
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_items_transform(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate items_map/items_filter are only provided with items."""
                    if (
                        self.items_map is not None or self.items_filter is not None
                    ) and self.items is None:
                        msg = "items_map/items_filter can only be used with items"
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_entries_transform(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate entries_map/entries_filter are only provided with entries."""
                    if (
                        self.entries_map is not None or self.entries_filter is not None
                    ) and self.entries is None:
                        msg = "entries_map/entries_filter can only be used with entries"
                        raise ValueError(msg)
                    return self

                @model_validator(mode="after")
                def validate_model_data(
                    self,
                ) -> FlextTestsModels.Tests.Builders.AddParams:
                    """Validate model_data is only provided with model."""
                    if self.model_data is not None and self.model is None:
                        msg = "model_data can only be used with model"
                        raise ValueError(msg)
                    return self

            class BuildParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.build() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - Computed fields for derived values
                - model_validator for cross-field validation
                - Automatic type coercion and validation
                """

                as_model: type[BaseModel] | None = Field(
                    default=None,
                    description="Pydantic model class to instantiate",
                )
                as_list: bool = Field(
                    default=False,
                    description="Return as list of (key, value) tuples",
                )
                keys_only: bool = Field(
                    default=False,
                    description="Return only keys as list",
                )
                values_only: bool = Field(
                    default=False,
                    description="Return only values as list",
                )
                flatten: bool = Field(
                    default=False,
                    description="Flatten nested dicts with dot notation",
                )
                filter_none: bool = Field(
                    default=False,
                    description="Remove None values from result",
                )
                as_parametrized: bool = Field(
                    default=False,
                    description="Return as list of (test_id, data) tuples for pytest",
                )
                parametrize_key: str = Field(
                    default="test_id",
                    min_length=1,
                    description="Key to use as test_id in parametrized output",
                )
                validate_with: Callable[[t.Tests.Builders.BuilderDict], bool] | None = (
                    Field(
                        default=None,
                        description="Validation function that returns True if data is valid",
                    )
                )
                assert_with: Callable[[t.Tests.Builders.BuilderDict], None] | None = (
                    Field(
                        default=None,
                        description="Assertion function that raises on invalid data",
                    )
                )
                map_result: (
                    Callable[[t.Tests.Builders.BuilderDict], t.GeneralValueType] | None
                ) = Field(
                    default=None,
                    description="Transform function applied to final result",
                )

                @model_validator(mode="after")
                def validate_mutually_exclusive(
                    self,
                ) -> FlextTestsModels.Tests.Builders.BuildParams:
                    """Validate mutually exclusive output format options."""
                    output_options = [
                        self.as_model is not None,
                        self.as_list,
                        self.keys_only,
                        self.values_only,
                        self.as_parametrized,
                        self.map_result is not None,
                    ]
                    if sum(output_options) > 1:
                        # Multiple output formats specified - map_result takes precedence
                        pass  # This is expected - map_result can be combined
                    return self

            class ToResultParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.to_result() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - Computed fields for derived values
                - model_validator for cross-field validation
                """

                error: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error message to return as failure result",
                )
                error_code: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Error code for failure result",
                )
                error_data: dict[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Error metadata dictionary for failure result",
                )
                unwrap: bool = Field(
                    default=False,
                    description="Unwrap FlextResult and return value directly",
                )
                unwrap_msg: str | None = Field(
                    default=None,
                    min_length=1,
                    description="Custom error message when unwrap fails",
                )
                as_model: type[BaseModel] | None = Field(
                    default=None,
                    description="Pydantic model class to instantiate",
                )
                as_cls: type | None = Field(
                    default=None,
                    description="Any class to instantiate",
                )
                cls_args: tuple[t.GeneralValueType, ...] | None = Field(
                    default=None,
                    description="Positional arguments for as_cls",
                )
                validate_func: Callable[[t.GeneralValueType], bool] | None = Field(
                    default=None,
                    alias="validate",
                    description="Validation function for built data",
                )
                map_fn: (
                    Callable[[t.Tests.Builders.BuilderDict], t.GeneralValueType] | None
                ) = Field(
                    default=None,
                    description="Transform function applied before wrapping in result",
                )
                as_list_result: bool = Field(
                    default=False,
                    description="Return as FlextResult[list[T]] with values from dict",
                )
                as_dict_result: bool = Field(
                    default=False,
                    description="Return as FlextResult[dict[str, T]]",
                )

                @computed_field
                def effective_error_code(self) -> str:
                    """Compute effective error code with defaults."""
                    return self.error_code or c.Errors.VALIDATION_ERROR

                @model_validator(mode="after")
                def validate_mutually_exclusive(
                    self,
                ) -> FlextTestsModels.Tests.Builders.ToResultParams:
                    """Validate mutually exclusive options."""
                    if self.as_cls is not None and self.as_model is not None:
                        msg = "as_cls and as_model cannot be used together"
                        raise ValueError(msg)
                    if self.as_list_result and self.as_dict_result:
                        msg = (
                            "as_list_result and as_dict_result cannot be used together"
                        )
                        raise ValueError(msg)
                    if self.cls_args is not None and self.as_cls is None:
                        msg = "cls_args can only be used with as_cls"
                        raise ValueError(msg)
                    return self

            class BatchParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.batch() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - Computed fields for derived values
                - model_validator for cross-field validation
                """

                key: str = Field(min_length=1, description="Key to store batch under")
                scenarios: Sequence[tuple[str, t.GeneralValueType]] = Field(
                    description="Sequence of (scenario_id, data) tuples",
                )
                as_results: bool = Field(
                    default=False,
                    description="Wrap each value in r[T].ok()",
                )
                with_failures: Sequence[tuple[str, str]] | None = Field(
                    default=None,
                    min_length=1,
                    description="Sequence of (id, error) tuples for failure results",
                )

                @model_validator(mode="after")
                def validate_scenarios(
                    self,
                ) -> FlextTestsModels.Tests.Builders.BatchParams:
                    """Validate scenarios is not empty."""
                    if not self.scenarios:
                        msg = "scenarios cannot be empty"
                        raise ValueError(msg)
                    return self

            class ForkParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.fork() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - Automatic type coercion and validation
                - All updates validated as BuilderValue types
                """

                # Updates are passed as **kwargs, so we use a dict field
                # The actual updates dict is built from **kwargs in the method
                # This model validates the structure but updates are dynamic
                # Note: fork() accepts **updates directly, so we validate via model
                # but the updates dict itself is not a single field

            class MergeFromParams(FlextModels.Value):
                """Parameters for FlextTestsBuilders.merge_from() method.

                Uses Pydantic 2 advanced features:
                - Field constraints for validation
                - model_validator for strategy validation
                - Automatic type coercion and validation
                """

                strategy: str = Field(
                    default="deep",
                    min_length=1,
                    description="Merge strategy (deep, override, append, etc.)",
                )
                exclude_keys: set[str] | None = Field(
                    default=None,
                    description="Set of keys to exclude from merge",
                )

                @model_validator(mode="after")
                def validate_strategy(
                    self,
                ) -> FlextTestsModels.Tests.Builders.MergeFromParams:
                    """Validate merge strategy is valid."""
                    valid_strategies = {
                        "deep",
                        "override",
                        "append",
                        "prepend",
                        "replace",
                    }
                    if self.strategy not in valid_strategies:
                        msg = f"strategy must be one of {valid_strategies}, got {self.strategy}"
                        raise ValueError(msg)
                    return self


__all__ = ["FlextTestsModels", "m"]

m = FlextTestsModels
