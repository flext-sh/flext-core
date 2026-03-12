"""Models for FLEXT tests.

Provides FlextTestsModels, extending FlextModels with test-specific model definitions
for factories, test data, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import override

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import FlextModels, r, u
from flext_core._models.base import FlextModelFoundation
from flext_tests import c, t


# Create FlextTestsModels that extends FlextModels with Tests.Factory namespace
class FlextTestsModelsDocker(FlextModelFoundation):
    """Docker-specific models for test infrastructure."""

    class ContainerInfo(FlextModels.Value):
        """Container information model."""

        name: str
        status: c.Tests.Docker.ContainerStatus
        ports: Mapping[str, str]
        image: str
        container_id: str = ""

        @override
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

        @override
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


class FlextTestsModelsFactory(FlextModelFoundation):
    """Factory models for test data generation."""

    class ModelFactoryParams(FlextModels.Value):
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
            description="Wrap result in r",
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
        value: t.Tests.object | None = Field(
            default=None,
            description="Value override",
        )
        # Value-specific
        data: str | None = Field(default=None, description="Data override")
        value_count: int | None = Field(
            default=None,
            description="Value count override",
        )
        # Generic overrides
        overrides: Mapping[str, t.Tests.object] | None = Field(
            default=None,
            description="Generic field overrides",
        )
        # Factory/transform/validation
        factory: Callable[[], BaseModel] | None = Field(
            default=None,
            description="Custom factory function",
        )
        transform: Callable[[BaseModel], BaseModel] | None = Field(
            default=None,
            description="Transform function",
        )
        validate_fn: Callable[[BaseModel], bool] | None = Field(
            default=None,
            alias="validate",
            description="Validation function",
        )

        @model_validator(mode="after")
        def validate_mapping(
            self,
        ) -> FlextTestsModels.Tests.Factory.ModelFactoryParams:
            """Validate as_mapping keys if provided."""
            if self.as_mapping and self.count > 1 and len(self.as_mapping) < self.count:
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
        value: t.Tests.object = Field(
            default=None,
            description="Value for success (required for 'ok')",
        )
        count: int = Field(
            default=1,
            ge=1,
            description="Number of results to create",
        )
        values: Sequence[t.Tests.object] | None = Field(
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
        transform: Callable[[t.Tests.object], t.Tests.object] | None = Field(
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
                msg = "from_value kind requires error_on_none when value is None"
                raise ValueError(msg)
            return self

    class ListFactoryParams(FlextModels.Value):
        """Parameters for tt.list() factory method with Pydantic 2 advanced validation.

        Uses Field constraints for inline validation. Source can be ModelKind (str),
        Sequence, or Callable - uses object type to accept all variants.
        """

        model_config = BaseModel.model_config.copy()
        model_config["populate_by_name"] = True

        source: (
            t.Tests.Factory.ModelKind
            | Sequence[t.Tests.object]
            | Callable[[], t.Tests.object]
        ) = Field(
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
            description="Wrap result in r",
        )
        unique: bool = Field(
            default=False,
            description="Ensure all items are unique",
        )
        transform: Callable[[t.Tests.object], t.Tests.object] | None = Field(
            default=None,
            description="Transform function applied to each item",
        )
        filter_: Callable[[t.Tests.object], bool] | None = Field(
            default=None,
            alias="filter",
            description="Filter predicate to exclude items",
        )

    class DictFactoryParams(FlextModels.Value):
        """Parameters for tt.dict_factory() method with Pydantic 2 advanced validation.

        Uses Field constraints for inline validation. Source can be ModelKind (str),
        Mapping, or Callable - uses object type to accept all variants.
        """

        source: (
            t.Tests.Factory.ModelKind
            | Mapping[str, t.Tests.object]
            | Callable[[], tuple[str, t.Tests.object]]
        ) = Field(
            default="user",
            description="Source for dict items (ModelKind, Mapping, or Callable)",
        )
        count: int = Field(
            default=5,
            ge=1,
            description="Number of items to create",
        )
        key_factory: Callable[[int], str] | None = Field(
            default=None,
            description="Factory function for keys (takes index, returns str key)",
        )
        value_factory: Callable[[str], t.Tests.object] | None = Field(
            default=None,
            description="Factory function for values (takes key, returns value)",
        )
        as_result: bool = Field(
            default=False,
            description="Wrap result in r",
        )
        merge_with: Mapping[str, t.Tests.object] | None = Field(
            default=None,
            description="Additional mapping to merge into result",
        )

    class GenericFactoryParams(FlextModels.Value):
        """Parameters for tt.generic() factory method with Pydantic 2 advanced validation.

        Uses Field constraints for inline validation. Type validation done via
            model_validator since Field constraints cannot validate runtime type checks.
        """

        type_: type = Field(
            description="Type class to instantiate",
        )
        args: Sequence[t.Tests.object] | None = Field(
            default=None,
            description="Positional arguments for constructor",
        )
        call_kwargs: Mapping[str, t.Tests.object] | None = Field(
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
            description="Wrap result in r",
        )
        validate_fn: Callable[[object], bool] | None = Field(
            default=None,
            alias="validate",
            description="Validation predicate (must return True for success)",
        )

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

    # Use module-level Entity and Value to avoid Pydantic forward reference issues
    # Factory classes for test model creation
    class Entity(FlextModels.Entity):
        """Factory entity class for tests."""

        name: str = ""
        value: t.Tests.object = None

    class Value(FlextModels.Value):
        """Factory value object class for tests."""

        data: str = ""
        count: int = 0


class FlextTestsModelsFiles(FlextModelFoundation):
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
        created: datetime.datetime | None = None
        modified: datetime.datetime | None = None
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

    class CreateParams(FlextModels.Value):
        """Parameters for file creation operations with Pydantic 2 advanced validation."""

        content: t.Tests.object
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
            description="Auto-extract r value.",
        )

        @field_validator("name", mode="before")
        @classmethod
        def normalize_name(cls, value: t.Tests.object) -> str:
            """Normalize filename by stripping whitespace."""
            if isinstance(value, str):
                return value.strip()
            return str(value)

    class ReadParams(FlextModels.Value):
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

    class CompareParams(FlextModels.Value):
        """Parameters for file comparison operations with Pydantic 2 advanced validation."""

        file1: Path = Field(
            description="First file to compare (str or Path converted automatically).",
        )
        file2: Path = Field(
            description="Second file to compare (str or Path converted automatically).",
        )
        mode: str = Field(
            default=c.Tests.Files.CompareMode.CONTENT.value,
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
            return Path(value)

    class InfoParams(FlextModels.Value):
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
            return Path(value)

    class CreateKwargsParams(FlextModels.Value):
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

    class BatchParams(FlextModels.Value):
        """Parameters for FlextTestsFiles.batch() method."""

        files: t.Tests.Files.BatchFiles = Field(
            description="Mapping or Sequence of files to process",
        )
        directory: Path | None = Field(
            default=None,
            description="Target directory for create operations",
        )
        operation: t.Tests.Files.OperationLiteral = Field(
            default="create",
            description="Operation type: create, read, or delete",
        )
        model: type[BaseModel] | None = Field(
            default=None,
            description="Optional model class for read operations",
        )
        on_error: t.Tests.Files.ErrorModeLiteral = Field(
            default="collect",
            description="Error handling mode: stop, skip, or collect",
        )
        parallel: bool = Field(
            default=False,
            description="Run operations in parallel",
        )

    class BatchResult(FlextModels.Value):
        """Result of batch file operations."""

        succeeded: int = Field(
            ge=0,
            description="Number of successful operations",
        )
        failed: int = Field(ge=0, description="Number of failed operations")
        total: int = Field(ge=0, description="Total number of operations")
        results: Mapping[str, r[Path | t.Tests.object]] = Field(
            default_factory=dict,
            description="Mapping of file names to operation results",
        )
        errors: Mapping[str, str] = Field(
            default_factory=dict,
            description="Mapping of file names to error messages",
        )

        @computed_field
        def failure_count(self) -> int:
            """Alias for failed count."""
            return self.failed

        @computed_field
        def success_count(self) -> int:
            """Alias for succeeded count."""
            return self.succeeded

        @computed_field
        def success_rate(self) -> float:
            """Compute success rate as percentage."""
            if self.total == 0:
                return 0.0
            return (self.succeeded / self.total) * 100.0


class FlextTestsModelsValidator(FlextModelFoundation):
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


class FlextTestsModels(
    FlextTestsModelsDocker,
    FlextTestsModelsFactory,
    FlextTestsModelsFiles,
    FlextTestsModelsValidator,
    FlextModels,
):
    """Test models extending FlextModels with test-specific factory models."""

    def __init_subclass__(
        cls,
        **kwargs: Mapping[str, object],
    ) -> None:
        """Warn when FlextTestsModels is subclassed directly."""
        super().__init_subclass__(**kwargs)
        if cls.__module__.startswith("tests"):
            return
        u.warn_once($$$)

    class Tests:
        """Test-specific models namespace."""

        class Docker(FlextTestsModelsDocker):
            """Docker models - real inheritance."""

        class Factory(FlextTestsModelsFactory):
            """Factory models - real inheritance."""

        class Files(FlextTestsModelsFiles):
            """File models - real inheritance."""

        class Validator(FlextTestsModelsValidator):
            """Validator models - real inheritance."""


m = FlextTestsModels

__all__ = ["FlextTestsModels", "m"]
