"""Models for FLEXT tests.

Provides FlextTestsModels, extending FlextModels with test-specific model definitions
for factories, test data, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import r
from flext_core.models import FlextModels as FlextModelsBase
from flext_tests.constants import ContainerStatus, c
from flext_tests.typings import t


# Create FlextTestsModels that extends FlextModels with Tests.Factory namespace
class FlextTestsModels(FlextModelsBase):
    """Test models extending FlextModels with test-specific factory models."""

    class Tests:
        """Test-specific models namespace."""

        class Docker:
            """Docker-specific models for test infrastructure."""

            class ContainerInfo(FlextModelsBase.Value):
                """Container information model."""

                name: str
                status: ContainerStatus
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

            class ContainerConfig(FlextModelsBase.Value):
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

            class ContainerState(FlextModelsBase.Value):
                """Container state tracking model."""

                container_name: str
                is_dirty: bool
                worker_id: str
                last_updated: str | None = None

        class Factory:
            """Factory models for test data generation."""

            class ModelFactoryParams(FlextModelsBase.Value):
                """Parameters for factory model() method with Pydantic 2 validation."""

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
                    description="Return as dict keyed by ID",
                )
                as_mapping: Mapping[str, str] | None = Field(
                    default=None,
                    description="Custom key mapping for dict output",
                )
                as_result: bool = Field(
                    default=False,
                    description="Wrap result in FlextResult",
                )
                # User-specific
                model_id: str | None = Field(
                    default=None,
                    description="Model ID override",
                )
                name: str | None = Field(default=None, description="Name override")
                email: str | None = Field(default=None, description="Email override")
                active: bool | None = Field(
                    default=None,
                    description="Active status override",
                )
                # Config-specific
                service_type: str | None = Field(
                    default=None,
                    description="Service type override",
                )
                environment: str | None = Field(
                    default=None,
                    description="Environment override",
                )
                debug: bool | None = Field(default=None, description="Debug override")
                log_level: str | None = Field(
                    default=None,
                    description="Log level override",
                )
                timeout: int | None = Field(
                    default=None,
                    description="Timeout override",
                )
                max_retries: int | None = Field(
                    default=None,
                    description="Max retries override",
                )
                # Service-specific
                status: str | None = Field(default=None, description="Status override")
                # Entity-specific
                value: t.GeneralValueType | None = Field(
                    default=None,
                    description="Value override",
                )
                # ValueObject-specific
                data: str | None = Field(default=None, description="Data override")
                value_count: int | None = Field(
                    default=None,
                    description="Value count override",
                )
                # Generic overrides
                overrides: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Generic field overrides",
                )
                # Factory/transform/validation
                factory: Callable[[], object] | None = Field(
                    default=None,
                    description="Custom factory function",
                )
                transform: Callable[[object], object] | None = Field(
                    default=None,
                    description="Transform function",
                )
                validate_fn: Callable[[object], bool] | None = Field(
                    default=None,
                    alias="validate",
                    description="Validation function",
                )

                @model_validator(mode="after")
                def validate_mapping(
                    self,
                ) -> FlextTestsModels.Tests.Factory.ModelFactoryParams:
                    """Validate as_mapping keys if provided."""
                    if (
                        self.as_mapping
                        and self.count > 1
                        and len(self.as_mapping) < self.count
                    ):
                        msg = f"as_mapping must have at least {self.count} keys"
                        raise ValueError(msg)
                    return self

            class ResultFactoryParams(FlextModelsBase.Value):
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
                def validate_batch_params(
                    self,
                ) -> FlextTestsModels.Tests.Factory.ResultFactoryParams:
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
                def validate_kind_value(
                    self,
                ) -> FlextTestsModels.Tests.Factory.ResultFactoryParams:
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

            class ListFactoryParams(FlextModelsBase.Value):
                """Parameters for tt.list() factory method with Pydantic 2 advanced validation.

                Uses Field constraints for inline validation. Source can be ModelKind (str),
                Sequence, or Callable - uses object type to accept all variants.
                """

                model_config = BaseModel.model_config.copy()
                model_config["populate_by_name"] = True

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

            class DictFactoryParams(FlextModelsBase.Value):
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

            class GenericFactoryParams(FlextModelsBase.Value):
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
                def validate_type(
                    self,
                ) -> FlextTestsModels.Tests.Factory.GenericFactoryParams:
                    """Validate type_ is a class - Field constraints cannot do isinstance checks."""
                    if not isinstance(self.type_, type):
                        msg = "type_ must be a class"
                        raise ValueError(msg)
                    return self

            class User(FlextModelsBase.Value):
                """Test user model - immutable value object."""

                id: str
                name: str
                email: str
                active: bool = True

            class Config(FlextModelsBase.Value):
                """Test configuration model - immutable value object."""

                service_type: str = "api"
                environment: str = "test"
                debug: bool = True
                log_level: str = "DEBUG"
                timeout: int = 30
                max_retries: int = 3

            class Service(FlextModelsBase.Value):
                """Test service model - immutable value object."""

                id: str
                type: str = "api"
                name: str = ""
                status: str = "active"

            class Entity(FlextModelsBase.Entity):
                """Test entity model with identity."""

                name: str
                value: t.GeneralValueType = ""

            class ValueObject(FlextModelsBase.Value):
                """Test value object model."""

                data: str = ""
                count: int = 1

        class Files:
            """File-related models for test infrastructure."""

            class FileInfo(FlextModelsBase.Value):
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

            class ContentMeta(FlextModelsBase.Value):
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

            class CreateParams(FlextModelsBase.Value):
                """Parameters for file creation operations with Pydantic 2 advanced validation."""

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

            class ReadParams(FlextModelsBase.Value):
                """Parameters for file read operations with Pydantic 2 advanced validation."""

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

            class CompareParams(FlextModelsBase.Value):
                """Parameters for file comparison operations with Pydantic 2 advanced validation."""

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

            class InfoParams(FlextModelsBase.Value):
                """Parameters for file info() operations with Pydantic 2 validation."""

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
                    description="Parse content and include metadata.",
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

            class CreateKwargsParams(FlextModelsBase.Value):
                """Parameters for file create() kwargs with Pydantic 2 validation.

                Fields match FlextTestsFileManager.create() method signature exactly.
                """

                directory: Path | None = Field(
                    default=None,
                    description="Directory to create file in.",
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
                    description="JSON indentation level.",
                )
                delim: str = Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                )
                headers: list[str] | None = Field(
                    default=None,
                    description="CSV column headers.",
                )
                readonly: bool = Field(
                    default=False,
                    description="Create file as read-only.",
                )

            class BatchParams(FlextModelsBase.Value):
                """Parameters for FlextTestsFiles.batch() method."""

                files: t.Tests.Files.BatchFiles = Field(
                    description="Mapping or Sequence of files to process",
                )
                directory: Path | None = Field(
                    default=None,
                    description="Target directory for create operations",
                )
                operation: c.Tests.Files.OperationLiteral = Field(
                    default="create",
                    description="Operation type: create, read, or delete",
                )
                model: type[BaseModel] | None = Field(
                    default=None,
                    description="Optional model class for read operations",
                )
                on_error: c.Tests.Files.ErrorModeLiteral = Field(
                    default="collect",
                    description="Error handling mode: stop, skip, or collect",
                )
                parallel: bool = Field(
                    default=False,
                    description="Run operations in parallel",
                )

            class BatchResult(FlextModelsBase.Value):
                """Result of batch file operations."""

                succeeded: int = Field(
                    ge=0,
                    description="Number of successful operations",
                )
                failed: int = Field(ge=0, description="Number of failed operations")
                total: int = Field(ge=0, description="Total number of operations")
                results: Mapping[str, r[object]] = Field(
                    default_factory=dict,
                    description="Mapping of file names to operation results",
                )
                errors: Mapping[str, str] = Field(
                    default_factory=dict,
                    description="Mapping of file names to error messages",
                )

                @computed_field
                def success_count(self) -> int:
                    """Alias for succeeded count."""
                    return self.succeeded

                @computed_field
                def failure_count(self) -> int:
                    """Alias for failed count."""
                    return self.failed

                @computed_field
                def success_rate(self) -> float:
                    """Compute success rate as percentage."""
                    if self.total == 0:
                        return 0.0
                    return (self.succeeded / self.total) * 100.0

        class Validator:
            """Validator models for architecture validation."""

            Severity = c.Tests.Validator.Severity

            class Violation(FlextModelsBase.Value):
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

            class ScanResult(FlextModelsBase.Value):
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

        class Builders:
            """Builder models for test data construction using Pydantic 2 advanced features.

            All models use Pydantic 2 Field validation, computed fields, and model_validator
            for comprehensive parameter validation and computation.
            """

            class AddParams(FlextModelsBase.Value):
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
                model: type[BaseModel] | None = Field(
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

            class BuildParams(FlextModelsBase.Value):
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
                def validate_parametrize_key(
                    self,
                ) -> FlextTestsModels.Tests.Builders.BuildParams:
                    """Validate parametrize_key is not empty when as_parametrized is True."""
                    if self.as_parametrized and not self.parametrize_key:
                        msg = "parametrize_key cannot be empty when as_parametrized is True"
                        raise ValueError(msg)
                    return self

            class ToResultParams(FlextModelsBase.Value):
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

            class BatchParams(FlextModelsBase.Value):
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

            class MergeFromParams(FlextModelsBase.Value):
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
                exclude_keys: frozenset[str] | None = Field(
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

        class Matcher:
            """Matcher models for test assertions and matching operations using Pydantic 2 advanced features."""

            class OkParams(FlextModelsBase.Value):
                """Parameters for matcher ok() operations with Pydantic 2 validation."""

                model_config = ConfigDict(populate_by_name=True)

                eq: object | None = Field(
                    default=None,
                    description="Expected value (equality check)",
                )
                ne: object | None = Field(
                    default=None,
                    description="Value must not equal",
                )
                is_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    validation_alias=AliasChoices("is_", "is"),
                    description="Type check (isinstance) - single type or tuple",
                )
                none: bool | None = Field(
                    default=None,
                    description="None check (True=must be None, False=must not be None)",
                )
                empty: bool | None = Field(
                    default=None,
                    description="Empty check (True=must be empty, False=must not be empty)",
                )
                gt: float | int | None = Field(
                    default=None,
                    description="Greater than (numeric or length)",
                )
                gte: float | int | None = Field(
                    default=None,
                    description="Greater than or equal",
                )
                lt: float | int | None = Field(default=None, description="Less than")
                lte: float | int | None = Field(
                    default=None,
                    description="Less than or equal",
                )
                has: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Unified containment - value contains item(s) (replaces contains)",
                )
                lacks: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified non-containment - value does NOT contain item(s) (replaces excludes)",
                )
                starts: str | None = Field(
                    default=None,
                    description="String starts with prefix",
                )
                ends: str | None = Field(
                    default=None,
                    description="String ends with suffix",
                )
                match: str | None = Field(
                    default=None,
                    description="Regex pattern (for strings)",
                )
                len: t.Tests.Matcher.LengthSpec | None = Field(
                    default=None,
                    description="Length spec - exact int or (min, max) tuple",
                )
                deep: t.Tests.Matcher.DeepSpec | None = Field(
                    default=None,
                    description="Deep structural matching specification",
                )
                path: t.Tests.Matcher.PathSpec | None = Field(
                    default=None,
                    description="Extract nested value via dot notation before validation",
                )
                where: t.Tests.Matcher.PredicateSpec | None = Field(
                    default=None,
                    description="Custom predicate function for validation",
                )
                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )
                # Legacy parameters (deprecated)
                contains: object | None = Field(
                    default=None,
                    description="Legacy: use has=",
                )
                excludes: object | None = Field(
                    default=None,
                    description="Legacy: use lacks=",
                )

                @model_validator(mode="before")
                @classmethod
                def convert_legacy_params(
                    cls,
                    data: dict[str, object] | FlextTestsModels.Tests.Matcher.OkParams,
                ) -> dict[str, object]:
                    """Convert legacy contains/excludes to has/lacks."""
                    if isinstance(data, dict):
                        # Convert legacy parameters before model creation
                        if "contains" in data and "has" not in data:
                            data["has"] = data.pop("contains")
                        if "excludes" in data and "lacks" not in data:
                            data["lacks"] = data.pop("excludes")
                        return data
                    # If data is OkParams, convert to dict
                    if isinstance(data, FlextTestsModels.Tests.Matcher.OkParams):
                        return data.model_dump()
                    return cast("dict[str, object]", data)

            class FailParams(FlextModelsBase.Value):
                """Parameters for matcher fail() operations with Pydantic 2 validation."""

                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )
                has: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified containment - error contains substring(s) (replaces contains)",
                )
                lacks: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified non-containment - error does NOT contain substring(s) (replaces excludes)",
                )
                starts: str | None = Field(
                    default=None,
                    description="Error starts with prefix",
                )
                ends: str | None = Field(
                    default=None,
                    description="Error ends with suffix",
                )
                match: str | None = Field(
                    default=None,
                    description="Error matches regex",
                )
                code: str | None = Field(default=None, description="Error code equals")
                code_has: t.Tests.Matcher.ErrorCodeSpec | None = Field(
                    default=None,
                    description="Error code contains substring(s)",
                )
                data: t.Tests.Matcher.ErrorDataSpec | None = Field(
                    default=None,
                    description="Error data contains key-value pairs",
                )
                # Legacy parameters (deprecated)
                error: str | None = Field(default=None, description="Legacy: use has=")
                contains: str | Sequence[str] | None = Field(
                    default=None,
                    description="Legacy: use has=",
                )
                excludes: str | Sequence[str] | None = Field(
                    default=None,
                    description="Legacy: use lacks=",
                )

                @model_validator(mode="before")
                @classmethod
                def convert_legacy_params(
                    cls,
                    data: dict[str, object] | FlextTestsModels.Tests.Matcher.FailParams,
                ) -> dict[str, object]:
                    """Convert legacy error/contains/excludes to has/lacks."""
                    if isinstance(data, dict):
                        # Convert legacy parameters before model creation
                        if "error" in data and "has" not in data:
                            data["has"] = data.pop("error")
                        if "contains" in data and "has" not in data:
                            data["has"] = data.pop("contains")
                        if "excludes" in data and "lacks" not in data:
                            data["lacks"] = data.pop("excludes")
                        return data
                    # If data is FailParams, convert to dict
                    if isinstance(data, FlextTestsModels.Tests.Matcher.FailParams):
                        return data.model_dump()
                    return cast("dict[str, object]", data)

            class ThatParams(FlextModelsBase.Value):
                """Parameters for matcher that() operations with Pydantic 2 validation."""

                model_config = ConfigDict(populate_by_name=True)

                msg: str | None = Field(
                    default=None,
                    description="Custom error message",
                )
                eq: object | None = Field(
                    default=None,
                    description="Expected value (equality check)",
                )
                ne: object | None = Field(
                    default=None,
                    description="Value must not equal",
                )
                is_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    validation_alias=AliasChoices("is_", "is"),
                    description="Type check (isinstance) - single type or tuple",
                )
                not_: type[object] | tuple[type[object], ...] | None = Field(
                    default=None,
                    validation_alias=AliasChoices("not_", "not"),
                    description="Type check - value is NOT instance of type(s)",
                )
                none: bool | None = Field(
                    default=None,
                    description="None check (True=must be None, False=must not be None)",
                )
                empty: bool | None = Field(
                    default=None,
                    description="Empty check (True=must be empty, False=must not be empty)",
                )
                gt: float | int | None = Field(
                    default=None,
                    description="Greater than (numeric or length)",
                )
                gte: float | int | None = Field(
                    default=None,
                    description="Greater than or equal",
                )
                lt: float | int | None = Field(default=None, description="Less than")
                lte: float | int | None = Field(
                    default=None,
                    description="Less than or equal",
                )
                len: t.Tests.Matcher.LengthSpec | None = Field(
                    default=None,
                    description="Length spec - exact int or (min, max) tuple",
                )
                has: t.Tests.Matcher.ContainmentSpec | None = Field(
                    default=None,
                    description="Unified containment - value contains item(s) (replaces contains)",
                )
                lacks: t.Tests.Matcher.ExclusionSpec | None = Field(
                    default=None,
                    description="Unified non-containment - value does NOT contain item(s) (replaces excludes)",
                )
                starts: str | None = Field(
                    default=None,
                    description="String starts with prefix",
                )
                ends: str | None = Field(
                    default=None,
                    description="String ends with suffix",
                )
                match: str | None = Field(
                    default=None,
                    description="Regex pattern (for strings)",
                )
                first: object | None = Field(
                    default=None,
                    description="Sequence first item equals",
                )
                last: object | None = Field(
                    default=None,
                    description="Sequence last item equals",
                )
                all_: t.Tests.Matcher.SequencePredicate | None = Field(
                    default=None,
                    validation_alias=AliasChoices("all_", "all"),
                    description="All items match type or predicate",
                )
                any_: t.Tests.Matcher.SequencePredicate | None = Field(
                    default=None,
                    validation_alias=AliasChoices("any_", "any"),
                    description="Any item matches type or predicate",
                )
                sorted: t.Tests.Matcher.SortKey | None = Field(
                    default=None,
                    description="Is sorted (True=ascending, or key function)",
                )
                unique: bool | None = Field(
                    default=None,
                    description="All items unique",
                )
                keys: t.Tests.Matcher.KeySpec | None = Field(
                    default=None,
                    description="Mapping has all keys",
                )
                lacks_keys: t.Tests.Matcher.KeySpec | None = Field(
                    default=None,
                    description="Mapping missing keys",
                )
                values: Sequence[object] | None = Field(
                    default=None,
                    description="Mapping has all values",
                )
                kv: t.Tests.Matcher.KeyValueSpec | None = Field(
                    default=None,
                    description="Key-value pairs (single tuple or mapping)",
                )
                attrs: t.Tests.Matcher.AttributeSpec | None = Field(
                    default=None,
                    description="Object has attribute(s)",
                )
                methods: t.Tests.Matcher.AttributeSpec | None = Field(
                    default=None,
                    description="Object has method(s)",
                )
                attr_eq: t.Tests.Matcher.AttributeValueSpec | None = Field(
                    default=None,
                    description="Attribute equals (single tuple or mapping)",
                )
                ok: bool | None = Field(
                    default=None,
                    description="For FlextResult: assert success",
                )
                error: str | Sequence[str] | None = Field(
                    default=None,
                    description="For FlextResult: error contains",
                )
                deep: t.Tests.Matcher.DeepSpec | None = Field(
                    default=None,
                    description="Deep structural matching specification",
                )
                where: t.Tests.Matcher.PredicateSpec | None = Field(
                    default=None,
                    description="Custom predicate function",
                )
                # Legacy parameters (deprecated)
                contains: object | None = Field(
                    default=None,
                    description="Legacy: use has=",
                )
                excludes: object | None = Field(
                    default=None,
                    description="Legacy: use lacks=",
                )
                length: int | None = Field(default=None, description="Legacy: use len=")
                length_gt: int | None = Field(
                    default=None,
                    description="Legacy: use len=(min, max)",
                )
                length_gte: int | None = Field(
                    default=None,
                    description="Legacy: use len=(min, max)",
                )
                length_lt: int | None = Field(
                    default=None,
                    description="Legacy: use len=(min, max)",
                )
                length_lte: int | None = Field(
                    default=None,
                    description="Legacy: use len=(min, max)",
                )

                @model_validator(mode="before")
                @classmethod
                def convert_legacy_params(
                    cls,
                    data: dict[str, object] | FlextTestsModels.Tests.Matcher.ThatParams,
                ) -> dict[str, object]:
                    """Convert legacy parameters to unified ones."""
                    if isinstance(data, dict):
                        # Convert legacy parameters before model creation
                        if "contains" in data and "has" not in data:
                            data["has"] = data.pop("contains")
                        if "excludes" in data and "lacks" not in data:
                            data["lacks"] = data.pop("excludes")
                        if "error" in data and "has" not in data:
                            data["has"] = data.pop("error")
                        if "length" in data and "len" not in data:
                            data["len"] = data.pop("length")
                        if (
                            "length_gt" in data
                            or "length_gte" in data
                            or "length_lt" in data
                            or "length_lte" in data
                        ):
                            min_len = data.pop(
                                "length_gte",
                                data.pop("length_gt", None),
                            )
                            max_len = data.pop(
                                "length_lte",
                                data.pop("length_lt", None),
                            )
                            if (
                                min_len is not None or max_len is not None
                            ) and "len" not in data:
                                # Ensure tuple[int, int] not tuple[int, float]
                                max_val_raw = (
                                    max_len if max_len is not None else 2147483647
                                )  # max int32
                                max_val: int = (
                                    int(max_val_raw)
                                    if isinstance(max_val_raw, (int, float))
                                    else 2147483647
                                )
                                min_val: int = (
                                    int(min_len)
                                    if min_len is not None
                                    and isinstance(min_len, (int, float))
                                    else 0
                                )
                                data["len"] = (min_val, max_val)
                        return data
                    # If data is ThatParams, convert to dict
                    if isinstance(data, FlextTestsModels.Tests.Matcher.ThatParams):
                        return data.model_dump()
                    return cast("dict[str, object]", data)

            class ScopeParams(FlextModelsBase.Value):
                """Parameters for matcher scope() operations with Pydantic 2 validation."""

                config: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Initial configuration values",
                )
                container: Mapping[str, object] | None = Field(
                    default=None,
                    description="Initial container/service mappings",
                )
                context: Mapping[str, t.GeneralValueType] | None = Field(
                    default=None,
                    description="Initial context values",
                )
                cleanup: t.Tests.Matcher.CleanupSpec | None = Field(
                    default=None,
                    description="Sequence of cleanup functions to call on exit",
                )
                env: t.Tests.Matcher.EnvironmentSpec | None = Field(
                    default=None,
                    description="Temporary environment variables (restored on exit)",
                )
                cwd: Path | str | None = Field(
                    default=None,
                    description="Temporary working directory (restored on exit)",
                )

                @field_validator("cwd", mode="before")
                @classmethod
                def convert_cwd(cls, value: Path | str | None) -> Path | str | None:
                    """Convert string to Path if needed."""
                    if isinstance(value, str):
                        return Path(value)
                    return value

            class DeepMatchResult(FlextModelsBase.Value):
                """Result of deep matching operations."""

                path: str = Field(description="Path where match occurred or failed")
                expected: object = Field(description="Expected value or predicate")
                actual: object | None = Field(
                    default=None,
                    description="Actual value found",
                )
                matched: bool = Field(description="Whether match succeeded")
                reason: str = Field(
                    default="",
                    description="Reason for match failure if matched=False",
                )

            class Chain(FlextModelsBase.Value):
                """Chain matcher configuration for railway pattern assertions."""

                result: r[Any] = Field(description="FlextResult being chained")

            class TestScope(FlextModelsBase.Value):
                """Test scope configuration for isolated test execution."""

                config: dict[str, t.GeneralValueType] = Field(
                    default_factory=dict,
                    description="Configuration dictionary",
                )
                container: dict[str, object] = Field(
                    default_factory=dict,
                    description="Container/service mappings",
                )
                context: dict[str, t.GeneralValueType] = Field(
                    default_factory=dict,
                    description="Context values",
                )


# Type alias for convenience
m = FlextTestsModels

__all__ = ["FlextTestsModels", "m"]
