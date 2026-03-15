"""Models for FLEXT tests.

Provides FlextTestsModels, extending FlextModels with test-specific model definitions
for factories, test data, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, TypeAliasType, override

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core import FlextModels, r, u
from flext_tests import c, t


class FlextTestsModels(
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
        u.warn_once(
            f"subclass_{cls.__name__}",
            f"Subclassing FlextTestsModels in {cls.__name__} is deprecated. Use composition/containers.",
        )

    class Tests:
        """Test-specific models namespace."""

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

        class ModelFactoryParams(FlextModels.Value):
            """Parameters for factory model() method with Pydantic 2 validation."""

            kind: Annotated[
                t.Tests.Factory.ModelKind,
                Field(
                    default="user",
                    description="Model type to create",
                ),
            ]
            count: Annotated[
                int,
                Field(
                    default=1,
                    ge=1,
                    description="Number of instances to create",
                ),
            ]
            as_dict: Annotated[
                bool,
                Field(
                    default=False,
                    description="Return as dict keyed by ID",
                ),
            ]
            as_mapping: Annotated[
                Mapping[str, str] | None,
                Field(
                    default=None,
                    description="Custom key mapping for dict output",
                ),
            ]
            as_result: Annotated[
                bool,
                Field(
                    default=False,
                    description="Wrap result in r",
                ),
            ]
            # User-specific
            model_id: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Model ID override",
                ),
            ]
            name: Annotated[
                str | None, Field(default=None, description="Name override")
            ]
            email: Annotated[
                str | None, Field(default=None, description="Email override")
            ]
            active: Annotated[
                bool | None,
                Field(
                    default=None,
                    description="Active status override",
                ),
            ]
            # Config-specific
            service_type: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Service type override",
                ),
            ]
            environment: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Environment override",
                ),
            ]
            debug: Annotated[
                bool | None, Field(default=None, description="Debug override")
            ]
            log_level: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Log level override",
                ),
            ]
            timeout: Annotated[
                int | None,
                Field(
                    default=None,
                    description="Timeout override",
                ),
            ]
            max_retries: Annotated[
                int | None,
                Field(
                    default=None,
                    description="Max retries override",
                ),
            ]
            # Service-specific
            status: Annotated[
                str | None, Field(default=None, description="Status override")
            ]
            # Entity-specific
            value: Annotated[
                t.Tests.object | None,
                Field(
                    default=None,
                    description="Value override",
                ),
            ]
            # Value-specific
            data: Annotated[
                str | None, Field(default=None, description="Data override")
            ]
            value_count: Annotated[
                int | None,
                Field(
                    default=None,
                    description="Value count override",
                ),
            ]
            # Generic overrides
            overrides: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(
                    default=None,
                    description="Generic field overrides",
                ),
            ]
            # Factory/transform/validation
            factory: Annotated[
                Callable[[], BaseModel] | None,
                Field(
                    default=None,
                    description="Custom factory function",
                ),
            ]
            transform: Annotated[
                Callable[[BaseModel], BaseModel] | None,
                Field(
                    default=None,
                    description="Transform function",
                ),
            ]
            validate_fn: Annotated[
                Callable[[BaseModel], bool] | None,
                Field(
                    default=None,
                    alias="validate",
                    description="Validation function",
                ),
            ]

            @model_validator(mode="after")
            def validate_mapping(
                self,
            ) -> FlextTestsModels.Tests.ModelFactoryParams:
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

            kind: Annotated[
                t.Tests.Factory.ResultKind,
                Field(
                    default="ok",
                    description="Result type ('ok', 'fail', 'from_value')",
                ),
            ]
            value: Annotated[
                t.Tests.object,
                Field(
                    default=None,
                    description="Value for success (required for 'ok')",
                ),
            ]
            count: Annotated[
                int,
                Field(
                    default=1,
                    ge=1,
                    description="Number of results to create",
                ),
            ]
            values: Annotated[
                Sequence[t.Tests.object] | None,
                Field(
                    default=None,
                    description="Explicit value list for batch creation",
                ),
            ]
            errors: Annotated[
                Sequence[str] | None,
                Field(
                    default=None,
                    description="Error messages for failure results",
                ),
            ]
            mix_pattern: Annotated[
                t.Tests.Factory.BatchPattern | None,
                Field(
                    default=None,
                    description="Success/failure pattern (True=success, False=failure)",
                ),
            ]
            error: Annotated[
                str,
                Field(
                    default=c.Tests.Factory.ERROR_DEFAULT,
                    description="Error message for failure results",
                ),
            ]
            error_code: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Optional error code for failure results",
                ),
            ]
            error_on_none: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Error message when value is None (for 'from_value')",
                ),
            ]
            transform: Annotated[
                Callable[[t.Tests.object], t.Tests.object] | None,
                Field(
                    default=None,
                    description="Transform function for success values",
                ),
            ]

            @model_validator(mode="after")
            def validate_batch_params(
                self,
            ) -> FlextTestsModels.Tests.ResultFactoryParams:
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
            ) -> FlextTestsModels.Tests.ResultFactoryParams:
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

            source: Annotated[
                (
                    t.Tests.Factory.ModelKind
                    | Sequence[t.Tests.object]
                    | Callable[[], t.Tests.object]
                ),
                Field(
                    default="user",
                    description="Source for list items (ModelKind, Sequence, or Callable)",
                ),
            ]
            count: Annotated[
                int,
                Field(
                    default=5,
                    ge=1,
                    description="Number of items to create",
                ),
            ]
            as_result: Annotated[
                bool,
                Field(
                    default=False,
                    description="Wrap result in r",
                ),
            ]
            unique: Annotated[
                bool,
                Field(
                    default=False,
                    description="Ensure all items are unique",
                ),
            ]
            transform: Annotated[
                Callable[[t.Tests.object], t.Tests.object] | None,
                Field(
                    default=None,
                    description="Transform function applied to each item",
                ),
            ]
            filter_: Annotated[
                Callable[[t.Tests.object], bool] | None,
                Field(
                    default=None,
                    alias="filter",
                    description="Filter predicate to exclude items",
                ),
            ]

        class DictFactoryParams(FlextModels.Value):
            """Parameters for tt.dict_factory() method with Pydantic 2 advanced validation.

            Uses Field constraints for inline validation. Source can be ModelKind (str),
            Mapping, or Callable - uses object type to accept all variants.
            """

            source: Annotated[
                (
                    t.Tests.Factory.ModelKind
                    | Mapping[str, t.Tests.object]
                    | Callable[[], tuple[str, t.Tests.object]]
                ),
                Field(
                    default="user",
                    description="Source for dict items (ModelKind, Mapping, or Callable)",
                ),
            ]
            count: Annotated[
                int,
                Field(
                    default=5,
                    ge=1,
                    description="Number of items to create",
                ),
            ]
            key_factory: Annotated[
                Callable[[int], str] | None,
                Field(
                    default=None,
                    description="Factory function for keys (takes index, returns str key)",
                ),
            ]
            value_factory: Annotated[
                Callable[[str], t.Tests.object] | None,
                Field(
                    default=None,
                    description="Factory function for values (takes key, returns value)",
                ),
            ]
            as_result: Annotated[
                bool,
                Field(
                    default=False,
                    description="Wrap result in r",
                ),
            ]
            merge_with: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(
                    default=None,
                    description="Additional mapping to merge into result",
                ),
            ]

        class GenericFactoryParams(FlextModels.Value):
            """Parameters for tt.generic() factory method with Pydantic 2 advanced validation.

            Uses Field constraints for inline validation. Type validation done via
                model_validator since Field constraints cannot validate runtime type checks.
            """

            type_: Annotated[
                type,
                Field(
                    description="Type class to instantiate",
                ),
            ]
            args: Annotated[
                Sequence[t.Tests.object] | None,
                Field(
                    default=None,
                    description="Positional arguments for constructor",
                ),
            ]
            call_kwargs: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(
                    default=None,
                    description="Keyword arguments for constructor",
                ),
            ]
            count: Annotated[
                int,
                Field(
                    default=1,
                    ge=1,
                    description="Number of instances to create",
                ),
            ]
            as_result: Annotated[
                bool,
                Field(
                    default=False,
                    description="Wrap result in r",
                ),
            ]
            validate_fn: Annotated[
                Callable[..., bool] | None,
                Field(
                    default=None,
                    alias="validate",
                    description="Validation predicate (must return True for success)",
                ),
            ]

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
            """Factory entity class for tests."""

            name: str = ""
            value: t.Tests.object = None

        class Value(FlextModels.Value):
            """Factory value object class for tests."""

            data: str = ""
            count: int = 0

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
            content_meta: FlextTestsModels.Tests.ContentMeta | None = None
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
            name: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_FILENAME,
                    min_length=1,
                    description="Filename for the created file (non-empty).",
                ),
            ]
            directory: Annotated[
                Path | None,
                Field(
                    default=None,
                    description="Target directory (uses base_dir or temp if None).",
                ),
            ]
            fmt: Annotated[
                c.Tests.Files.FormatLiteral,
                Field(
                    default="auto",
                    description="File format override.",
                ),
            ]
            enc: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                ),
            ]
            indent: Annotated[
                int,
                Field(
                    default=c.Tests.Files.DEFAULT_JSON_INDENT,
                    ge=0,
                    description="JSON/YAML indentation (non-negative).",
                ),
            ]
            delim: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                ),
            ]
            headers: Annotated[
                list[str] | None,
                Field(
                    default=None,
                    description="CSV headers.",
                ),
            ]
            readonly: Annotated[
                bool,
                Field(
                    default=False,
                    description="Create file as read-only.",
                ),
            ]
            extract_result: Annotated[
                bool,
                Field(
                    default=True,
                    description="Auto-extract r value.",
                ),
            ]

            @field_validator("name", mode="before")
            @classmethod
            def normalize_name(cls, value: t.Tests.object) -> str:
                """Normalize filename by stripping whitespace."""
                if isinstance(value, str):
                    return value.strip()
                return str(value)

        class ReadParams(FlextModels.Value):
            """Parameters for file read operations with Pydantic 2 advanced validation."""

            path: Annotated[
                Path,
                Field(
                    description="Path to file to read (str or Path converted automatically).",
                ),
            ]
            model_cls: Annotated[
                type[BaseModel] | None,
                Field(
                    default=None,
                    description="Optional Pydantic model class to deserialize into.",
                ),
            ]
            fmt: Annotated[
                c.Tests.Files.FormatLiteral,
                Field(
                    default="auto",
                    description="Format override.",
                ),
            ]
            enc: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                ),
            ]
            delim: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                ),
            ]
            has_headers: Annotated[
                bool,
                Field(
                    default=True,
                    description="CSV has headers.",
                ),
            ]

            @field_validator("path", mode="before")
            @classmethod
            def convert_path(cls, value: Path | str) -> Path:
                """Convert string to Path - Field constraints cannot handle type conversion."""
                return Path(value) if isinstance(value, str) else value

        class CompareParams(FlextModels.Value):
            """Parameters for file comparison operations with Pydantic 2 advanced validation."""

            file1: Annotated[
                Path,
                Field(
                    description="First file to compare (str or Path converted automatically).",
                ),
            ]
            file2: Annotated[
                Path,
                Field(
                    description="Second file to compare (str or Path converted automatically).",
                ),
            ]
            mode: Annotated[
                str,
                Field(
                    default=c.Tests.Files.CompareMode.CONTENT.value,
                    description="Comparison mode.",
                ),
            ]
            ignore_ws: Annotated[
                bool,
                Field(
                    default=False,
                    description="Ignore whitespace in comparison.",
                ),
            ]
            ignore_case: Annotated[
                bool,
                Field(
                    default=False,
                    description="Case-insensitive comparison.",
                ),
            ]
            pattern: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Pattern to check if both files contain.",
                ),
            ]
            deep: Annotated[
                bool,
                Field(
                    default=True,
                    description="Use deep comparison for nested structures (dict/JSON/YAML).",
                ),
            ]
            keys: Annotated[
                list[str] | None,
                Field(
                    default=None,
                    description="Only compare these keys (for dict/JSON/YAML content).",
                ),
            ]
            exclude_keys: Annotated[
                list[str] | None,
                Field(
                    default=None,
                    description="Exclude these keys from comparison (for dict/JSON/YAML content).",
                ),
            ]

            @field_validator("file1", "file2", mode="before")
            @classmethod
            def convert_path(cls, value: Path | str) -> Path:
                """Convert string to Path - Field constraints cannot handle type conversion."""
                return Path(value)

        class InfoParams(FlextModels.Value):
            """Parameters for file info() operations with Pydantic 2 validation."""

            path: Annotated[
                Path,
                Field(
                    description="Path to file (str or Path converted automatically).",
                ),
            ]
            compute_hash: Annotated[
                bool,
                Field(
                    default=False,
                    description="Compute SHA256 hash.",
                ),
            ]
            detect_fmt: Annotated[
                bool,
                Field(
                    default=True,
                    description="Auto-detect format.",
                ),
            ]
            parse_content: Annotated[
                bool,
                Field(
                    default=False,
                    description="Parse content and include metadata.",
                ),
            ]
            validate_model: Annotated[
                type[BaseModel] | None,
                Field(
                    default=None,
                    description="Pydantic model to validate content against.",
                ),
            ]

            @field_validator("path", mode="before")
            @classmethod
            def convert_path(cls, value: Path | str) -> Path:
                """Convert string to Path - Field constraints cannot handle type conversion."""
                return Path(value)

        class CreateKwargsParams(FlextModels.Value):
            """Parameters for file create() kwargs with Pydantic 2 validation.

            Fields match FlextTestsFileManager.create() method signature exactly.
            """

            directory: Annotated[
                Path | None,
                Field(
                    default=None,
                    description="Directory to create file in.",
                ),
            ]
            fmt: Annotated[
                c.Tests.Files.FormatLiteral,
                Field(
                    default="auto",
                    description="File format override.",
                ),
            ]
            enc: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_ENCODING,
                    min_length=1,
                    description="File encoding.",
                ),
            ]
            indent: Annotated[
                int,
                Field(
                    default=c.Tests.Files.DEFAULT_JSON_INDENT,
                    ge=0,
                    description="JSON indentation level.",
                ),
            ]
            delim: Annotated[
                str,
                Field(
                    default=c.Tests.Files.DEFAULT_CSV_DELIMITER,
                    min_length=1,
                    max_length=1,
                    description="CSV delimiter (single character).",
                ),
            ]
            headers: Annotated[
                list[str] | None,
                Field(
                    default=None,
                    description="CSV column headers.",
                ),
            ]
            readonly: Annotated[
                bool,
                Field(
                    default=False,
                    description="Create file as read-only.",
                ),
            ]

        class BatchParams(FlextModels.Value):
            """Parameters for FlextTestsFiles.batch() method."""

            files: Annotated[
                t.Tests.Files.BatchFiles,
                Field(
                    description="Mapping or Sequence of files to process",
                ),
            ]
            directory: Annotated[
                Path | None,
                Field(
                    default=None,
                    description="Target directory for create operations",
                ),
            ]
            operation: Annotated[
                t.Tests.Files.OperationLiteral,
                Field(
                    default="create",
                    description="Operation type: create, read, or delete",
                ),
            ]
            model: Annotated[
                type[BaseModel] | None,
                Field(
                    default=None,
                    description="Optional model class for read operations",
                ),
            ]
            on_error: Annotated[
                t.Tests.Files.ErrorModeLiteral,
                Field(
                    default="collect",
                    description="Error handling mode: stop, skip, or collect",
                ),
            ]
            parallel: Annotated[
                bool,
                Field(
                    default=False,
                    description="Run operations in parallel",
                ),
            ]

        class BatchResult(FlextModels.Value):
            """Result of batch file operations."""

            succeeded: Annotated[
                int,
                Field(
                    ge=0,
                    description="Number of successful operations",
                ),
            ]
            failed: Annotated[
                int, Field(ge=0, description="Number of failed operations")
            ]
            total: Annotated[int, Field(ge=0, description="Total number of operations")]
            results: Annotated[
                Mapping[str, r[Path | t.Tests.object]],
                Field(
                    default_factory=dict,
                    description="Mapping of file names to operation results",
                ),
            ]
            errors: Annotated[
                Mapping[str, str],
                Field(
                    default_factory=dict,
                    description="Mapping of file names to error messages",
                ),
            ]

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
            violations: list[FlextTestsModels.Tests.Violation]
            passed: bool

            @classmethod
            def create(
                cls,
                validator_name: str,
                files_scanned: int,
                violations: list[FlextTestsModels.Tests.Violation],
            ) -> FlextTestsModels.Tests.ScanResult:
                """Create a ScanResult from violations."""
                return cls(
                    validator_name=validator_name,
                    files_scanned=files_scanned,
                    violations=violations,
                    passed=len(violations) == 0,
                )

        class AddParams(FlextModels.Value):
            """Parameters for builder add operations."""

            key: Annotated[
                str, Field(min_length=1, description="Key to store data under")
            ]
            value: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Direct value to store"),
            ]
            factory: Annotated[
                str | None, Field(default=None, description="Factory name to use")
            ]
            count: Annotated[
                int | None,
                Field(
                    default=None,
                    ge=1,
                    description="Number of items for factory generation",
                ),
            ]
            model: Annotated[
                type[BaseModel] | None,
                Field(default=None, description="Pydantic model class to instantiate"),
            ]
            model_data: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Data for model instantiation"),
            ]
            mapping: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Dict/mapping to store"),
            ]
            sequence: Annotated[
                Sequence[t.Tests.object] | None,
                Field(default=None, description="List/sequence to store"),
            ]
            transform: Annotated[
                t.Tests.Builders.TransformFunc | None,
                Field(default=None, description="Transform function before storing"),
            ]
            validate_func: Annotated[
                t.Tests.Builders.ValidateFunc | None,
                Field(
                    default=None, alias="validate", description="Validation function"
                ),
            ]
            merge: Annotated[
                bool,
                Field(
                    default=False,
                    description="Whether to merge with existing data at key",
                ),
            ]
            default: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Default value if result is None"),
            ]
            production: Annotated[
                bool | None,
                Field(default=None, description="Shortcut for production config"),
            ]
            debug: Annotated[
                bool | None,
                Field(default=None, description="Shortcut for debug config"),
            ]
            result: Annotated[
                r[t.Tests.object] | None,
                Field(default=None, description="r to store directly"),
            ]
            result_ok: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Value to wrap in r[T].ok()"),
            ]
            result_fail: Annotated[
                str | None,
                Field(default=None, description="Error message for r[T].fail()"),
            ]
            result_code: Annotated[
                str | None,
                Field(default=None, description="Error code for result_fail"),
            ]
            results: Annotated[
                Sequence[r[t.Tests.object]] | None,
                Field(default=None, description="Sequence of r to store"),
            ]
            results_ok: Annotated[
                Sequence[t.Tests.object] | None,
                Field(
                    default=None, description="Sequence of values to wrap in r[T].ok()"
                ),
            ]
            results_fail: Annotated[
                Sequence[str] | None,
                Field(
                    default=None,
                    description="Sequence of error messages for r[T].fail()",
                ),
            ]
            cls: Annotated[
                type[BaseModel] | None,
                Field(default=None, description="Class type to instantiate"),
            ]
            cls_args: Annotated[
                tuple[t.Tests.object, ...] | None,
                Field(default=None, description="Positional arguments for cls"),
            ]
            cls_kwargs: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Keyword arguments for cls"),
            ]
            items: Annotated[
                Sequence[t.Tests.object] | None,
                Field(default=None, description="Type-safe sequence"),
            ]
            items_map: Annotated[
                Callable[[t.Tests.object], t.Tests.object] | None,
                Field(default=None, description="Transform each item"),
            ]
            items_filter: Annotated[
                Callable[[t.Tests.object], bool] | None,
                Field(default=None, description="Filter items"),
            ]
            entries: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Type-safe mapping"),
            ]
            entries_map: Annotated[
                Callable[[t.Tests.object], t.Tests.object] | None,
                Field(default=None, description="Transform values"),
            ]
            entries_filter: Annotated[
                set[str] | None,
                Field(default=None, description="Include only these keys"),
            ]

            @computed_field
            def effective_count(self) -> int:
                return self.count or c.Tests.Factory.DEFAULT_BATCH_COUNT

            @computed_field
            def effective_error_code(self) -> str:
                return self.result_code or c.Errors.VALIDATION_ERROR

            @computed_field
            def resolution_priority(self) -> int:
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
                return 0

            @model_validator(mode="after")
            def validate_cls_with_args(self) -> FlextTestsModels.Tests.AddParams:
                if (
                    self.cls_args is not None or self.cls_kwargs is not None
                ) and self.cls is None:
                    msg = "cls_args/cls_kwargs can only be used with cls"
                    raise ValueError(msg)
                return self

            @model_validator(mode="after")
            def validate_count_positive(self) -> FlextTestsModels.Tests.AddParams:
                if self.count is not None and self.count < 1:
                    msg = c.Tests.Builders.ERROR_INVALID_COUNT.format(count=self.count)
                    raise ValueError(msg)
                return self

            @model_validator(mode="after")
            def validate_entries_transform(self) -> FlextTestsModels.Tests.AddParams:
                if (
                    self.entries_map is not None or self.entries_filter is not None
                ) and self.entries is None:
                    msg = "entries_map/entries_filter can only be used with entries"
                    raise ValueError(msg)
                return self

            @model_validator(mode="after")
            def validate_items_transform(self) -> FlextTestsModels.Tests.AddParams:
                if (
                    self.items_map is not None or self.items_filter is not None
                ) and self.items is None:
                    msg = "items_map/items_filter can only be used with items"
                    raise ValueError(msg)
                return self

            @model_validator(mode="after")
            def validate_model_data(self) -> FlextTestsModels.Tests.AddParams:
                if self.model_data is not None and self.model is None:
                    msg = "model_data can only be used with model"
                    raise ValueError(msg)
                return self

            @model_validator(mode="after")
            def validate_result_code_with_fail(
                self,
            ) -> FlextTestsModels.Tests.AddParams:
                if self.result_code is not None and self.result_fail is None:
                    msg = "result_code can only be used with result_fail"
                    raise ValueError(msg)
                return self

        class BuildParams(FlextModels.Value):
            """Parameters controlling builder output shaping."""

            as_model: Annotated[
                type[BaseModel] | None,
                Field(default=None, description="Pydantic model class to instantiate"),
            ]
            as_list: Annotated[
                bool,
                Field(
                    default=False, description="Return as list of (key, value) tuples"
                ),
            ]
            keys_only: Annotated[
                bool, Field(default=False, description="Return only keys as list")
            ]
            values_only: Annotated[
                bool, Field(default=False, description="Return only values as list")
            ]
            flatten: Annotated[
                bool,
                Field(
                    default=False, description="Flatten nested dicts with dot notation"
                ),
            ]
            filter_none: Annotated[
                bool, Field(default=False, description="Remove None values from result")
            ]
            as_parametrized: Annotated[
                bool,
                Field(
                    default=False,
                    description="Return as list of (test_id, data) tuples for pytest",
                ),
            ]
            parametrize_key: Annotated[
                str,
                Field(
                    default="test_id",
                    min_length=1,
                    description="Key to use as test_id in parametrized output",
                ),
            ]
            validate_with: Annotated[
                (Callable[[t.Tests.Builders.BuilderOutputDict], bool] | None),
                Field(default=None, description="Validation function"),
            ]
            assert_with: Annotated[
                Callable[[t.Tests.Builders.BuilderOutputDict], None] | None,
                (Field(default=None, description="Assertion function")),
            ]
            map_result: Annotated[
                (Callable[[t.Tests.Builders.BuilderOutputDict], t.Tests.object] | None),
                Field(
                    default=None,
                    description="Transform function applied to final result",
                ),
            ]

            @model_validator(mode="after")
            def validate_parametrize_key(self) -> FlextTestsModels.Tests.BuildParams:
                if self.as_parametrized and not self.parametrize_key:
                    msg = "parametrize_key cannot be empty when as_parametrized is True"
                    raise ValueError(msg)
                return self

        class ToResultParams(FlextModels.Value):
            """Parameters for converting builder output into results."""

            error: Annotated[
                str | None,
                Field(
                    default=None,
                    min_length=1,
                    description="Error message to return as failure result",
                ),
            ]
            error_code: Annotated[
                str | None,
                Field(
                    default=None,
                    min_length=1,
                    description="Error code for failure result",
                ),
            ]
            error_data: Annotated[
                FlextModels.ConfigMap | None,
                Field(default=None, description="Error metadata dictionary"),
            ]
            unwrap: Annotated[
                bool,
                Field(default=False, description="Unwrap r and return value directly"),
            ]
            unwrap_msg: Annotated[
                str | None,
                Field(
                    default=None,
                    min_length=1,
                    description="Custom error message when unwrap fails",
                ),
            ]
            as_model: Annotated[
                type[BaseModel] | None,
                Field(default=None, description="Pydantic model class to instantiate"),
            ]
            as_cls: Annotated[
                type | None,
                Field(default=None, description="Class type to instantiate"),
            ]
            cls_args: Annotated[
                tuple[t.Tests.object, ...] | None,
                Field(default=None, description="Positional arguments for as_cls"),
            ]
            validate_func: Annotated[
                Callable[[t.Tests.Builders.BuilderDict], bool] | None,
                (
                    Field(
                        default=None,
                        alias="validate",
                        description="Validation function for built data",
                    )
                ),
            ]
            map_fn: Annotated[
                Callable[[t.Tests.Builders.BuilderDict], t.Tests.object] | None,
                (
                    Field(
                        default=None,
                        description="Transform function applied before wrapping in result",
                    )
                ),
            ]
            as_list_result: Annotated[
                bool, Field(default=False, description="Return as r[list[T]]")
            ]
            as_dict_result: Annotated[
                bool, Field(default=False, description="Return as r[Mapping[str, T]]")
            ]

            @computed_field
            def effective_error_code(self) -> str:
                return self.error_code or c.Errors.VALIDATION_ERROR

            @model_validator(mode="after")
            def validate_mutually_exclusive(
                self,
            ) -> FlextTestsModels.Tests.ToResultParams:
                if self.as_cls is not None and self.as_model is not None:
                    msg = "as_cls and as_model cannot be used together"
                    raise ValueError(msg)
                if self.as_list_result and self.as_dict_result:
                    msg = "as_list_result and as_dict_result cannot be used together"
                    raise ValueError(msg)
                if self.cls_args is not None and self.as_cls is None:
                    msg = "cls_args can only be used with as_cls"
                    raise ValueError(msg)
                return self

        class BuildersBatchParams(FlextModels.Value):
            """Parameters for batching builder scenarios."""

            key: Annotated[
                str, Field(min_length=1, description="Key to store batch under")
            ]
            scenarios: Annotated[
                Sequence[tuple[str, t.Tests.object]],
                Field(description="Sequence of (scenario_id, data) tuples"),
            ]
            as_results: Annotated[
                bool, Field(default=False, description="Wrap each value in r[T].ok()")
            ]
            with_failures: Annotated[
                Sequence[tuple[str, str]] | None,
                Field(
                    default=None,
                    min_length=1,
                    description="Sequence of (id, error) tuples",
                ),
            ]

            @model_validator(mode="after")
            def validate_scenarios(self) -> FlextTestsModels.Tests.BuildersBatchParams:
                if not self.scenarios:
                    msg = "scenarios cannot be empty"
                    raise ValueError(msg)
                return self

        class MergeFromParams(FlextModels.Value):
            """Parameters for merge-from behavior selection."""

            strategy: Annotated[
                str,
                Field(
                    default="deep",
                    min_length=1,
                    description="Merge strategy (deep, override, append, etc.)",
                ),
            ]
            exclude_keys: Annotated[
                frozenset[str] | None,
                Field(default=None, description="Set of keys to exclude from merge"),
            ]

            @model_validator(mode="after")
            def validate_strategy(self) -> FlextTestsModels.Tests.MergeFromParams:
                valid_strategies = {"deep", "override", "append", "prepend", "replace"}
                if self.strategy not in valid_strategies:
                    msg = f"strategy must be one of {valid_strategies}, got {self.strategy}"
                    raise ValueError(msg)
                return self

        class OkParams(FlextModels.Value):
            """Matcher parameters for successful result assertions."""

            model_config = ConfigDict(populate_by_name=True)

            eq: Annotated[
                (
                    Mapping[str, t.Tests.object]
                    | Sequence[t.Tests.object]
                    | bytes
                    | str
                    | int
                    | float
                    | bool
                    | TypeAliasType
                    | None
                ),
                Field(
                    default=None,
                    description="Expected value (equality check)",
                    union_mode="left_to_right",
                ),
            ]
            ne: Annotated[
                (
                    Mapping[str, t.Tests.object]
                    | Sequence[t.Tests.object]
                    | bytes
                    | str
                    | int
                    | float
                    | bool
                    | TypeAliasType
                    | None
                ),
                Field(
                    default=None,
                    description="Value must not equal",
                    union_mode="left_to_right",
                ),
            ]
            is_: Annotated[
                type | tuple[type, ...] | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("is_", "is"),
                    description="Runtime type check",
                ),
            ]
            none: Annotated[bool | None, Field(default=None, description="None check")]
            empty: Annotated[
                bool | None, Field(default=None, description="Empty check")
            ]
            gt: Annotated[
                float | int | None, Field(default=None, description="Greater than")
            ]
            gte: Annotated[
                float | int | None,
                Field(default=None, description="Greater than or equal"),
            ]
            lt: Annotated[
                float | int | None, Field(default=None, description="Less than")
            ]
            lte: Annotated[
                float | int | None,
                Field(default=None, description="Less than or equal"),
            ]
            has: Annotated[
                t.Tests.Matcher.ContainmentSpec | None,
                Field(default=None, description="Unified containment check"),
            ]
            lacks: Annotated[
                t.Tests.Matcher.ExclusionSpec | None,
                Field(default=None, description="Unified non-containment check"),
            ]
            starts: Annotated[
                str | None, Field(default=None, description="String starts with prefix")
            ]
            ends: Annotated[
                str | None, Field(default=None, description="String ends with suffix")
            ]
            match: Annotated[
                str | None, Field(default=None, description="Regex pattern")
            ]
            len: Annotated[
                t.Tests.Matcher.LengthSpec | None,
                Field(default=None, description="Length spec"),
            ]
            deep: Annotated[
                t.Tests.Matcher.DeepSpec | None,
                Field(default=None, description="Deep structural matching"),
            ]
            path: Annotated[
                t.Tests.Matcher.PathSpec | None,
                Field(
                    default=None, description="Extract nested value via dot notation"
                ),
            ]
            where: Annotated[
                t.Tests.Matcher.PredicateSpec | None,
                Field(default=None, description="Custom predicate function"),
            ]
            msg: Annotated[
                str | None, Field(default=None, description="Custom error message")
            ]

        class FailParams(FlextModels.Value):
            """Matcher parameters for failure result assertions."""

            model_config = ConfigDict(populate_by_name=True)

            msg: Annotated[
                str | None, Field(default=None, description="Custom error message")
            ]
            has: Annotated[
                t.Tests.Matcher.ExclusionSpec | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("has", "contains"),
                    description="Error contains substring(s)",
                ),
            ]
            lacks: Annotated[
                t.Tests.Matcher.ExclusionSpec | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("lacks", "excludes"),
                    description="Error does NOT contain substring(s)",
                ),
            ]
            starts: Annotated[
                str | None, Field(default=None, description="Error starts with prefix")
            ]
            ends: Annotated[
                str | None, Field(default=None, description="Error ends with suffix")
            ]
            match: Annotated[
                str | None, Field(default=None, description="Error matches regex")
            ]
            code: Annotated[
                str | None, Field(default=None, description="Error code equals")
            ]
            code_has: Annotated[
                t.Tests.Matcher.ErrorCodeSpec | None,
                Field(default=None, description="Error code contains substring(s)"),
            ]
            data: Annotated[
                t.Tests.Matcher.ErrorDataSpec | None,
                Field(default=None, description="Error data contains key-value pairs"),
            ]

        class ThatParams(FlextModels.Value):
            """Generic matcher parameters for value assertions."""

            model_config = ConfigDict(populate_by_name=True)

            msg: Annotated[
                str | None, Field(default=None, description="Custom error message")
            ]
            eq: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Expected value (equality check)"),
            ]
            ne: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Value must not equal"),
            ]
            is_: Annotated[
                type | tuple[type, ...] | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("is_", "is"),
                    description="Runtime type check",
                ),
            ]
            not_: Annotated[
                type | tuple[type, ...] | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("not_", "not"),
                    description="Type check — value is NOT instance of type(s)",
                ),
            ]
            none: Annotated[bool | None, Field(default=None, description="None check")]
            empty: Annotated[
                bool | None, Field(default=None, description="Empty check")
            ]
            gt: Annotated[
                float | int | None, Field(default=None, description="Greater than")
            ]
            gte: Annotated[
                float | int | None,
                Field(default=None, description="Greater than or equal"),
            ]
            lt: Annotated[
                float | int | None, Field(default=None, description="Less than")
            ]
            lte: Annotated[
                float | int | None,
                Field(default=None, description="Less than or equal"),
            ]
            len: Annotated[
                t.Tests.Matcher.LengthSpec | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("len", "length"),
                    description="Length spec",
                ),
            ]
            length_gt: Annotated[
                int | None, Field(default=None, description="Length greater than")
            ]
            length_gte: Annotated[
                int | None,
                Field(default=None, description="Length greater than or equal"),
            ]
            length_lt: Annotated[
                int | None, Field(default=None, description="Length less than")
            ]
            length_lte: Annotated[
                int | None, Field(default=None, description="Length less than or equal")
            ]
            has: Annotated[
                t.Tests.Matcher.ContainmentSpec | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("has", "contains"),
                    description="Unified containment check",
                ),
            ]
            lacks: Annotated[
                t.Tests.Matcher.ExclusionSpec | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("lacks", "excludes"),
                    description="Unified non-containment check",
                ),
            ]
            starts: Annotated[
                str | None, Field(default=None, description="String starts with prefix")
            ]
            ends: Annotated[
                str | None, Field(default=None, description="String ends with suffix")
            ]
            match: Annotated[
                str | None, Field(default=None, description="Regex pattern")
            ]
            first: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Sequence first item equals"),
            ]
            last: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Sequence last item equals"),
            ]
            all_: Annotated[
                t.Tests.Matcher.SequencePredicate | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("all_", "all"),
                    description="All items match type or predicate",
                ),
            ]
            any_: Annotated[
                t.Tests.Matcher.SequencePredicate | None,
                Field(
                    default=None,
                    validation_alias=AliasChoices("any_", "any"),
                    description="Any item matches type or predicate",
                ),
            ]
            sorted: Annotated[
                t.Tests.Matcher.SortKey | None,
                Field(default=None, description="Is sorted"),
            ]
            unique: Annotated[
                bool | None, Field(default=None, description="All items unique")
            ]
            keys: Annotated[
                t.Tests.Matcher.KeySpec | None,
                Field(default=None, description="Mapping has all keys"),
            ]
            lacks_keys: Annotated[
                t.Tests.Matcher.KeySpec | None,
                Field(default=None, description="Mapping missing keys"),
            ]
            values: Annotated[
                Sequence[t.Tests.object] | None,
                Field(default=None, description="Mapping has all values"),
            ]
            kv: Annotated[
                t.Tests.Matcher.KeyValueSpec | None,
                Field(default=None, description="Key-value pairs"),
            ]
            attrs: Annotated[
                t.Tests.Matcher.AttributeSpec | None,
                Field(default=None, description="Object has attribute(s)"),
            ]
            methods: Annotated[
                t.Tests.Matcher.AttributeSpec | None,
                Field(default=None, description="Object has method(s)"),
            ]
            attr_eq: Annotated[
                t.Tests.Matcher.AttributeValueSpec | None,
                Field(default=None, description="Attribute equals"),
            ]
            ok: Annotated[
                bool | None, Field(default=None, description="For r: assert success")
            ]
            error: Annotated[
                str | Sequence[str] | None,
                Field(default=None, description="For r: error contains"),
            ]
            deep: Annotated[
                t.Tests.Matcher.DeepSpec | None,
                Field(default=None, description="Deep structural matching"),
            ]
            where: Annotated[
                t.Tests.Matcher.PredicateSpec | None,
                Field(default=None, description="Custom predicate function"),
            ]

            @model_validator(mode="after")
            def normalize_legacy_parameters(self) -> FlextTestsModels.Tests.ThatParams:
                updates: MutableMapping[str, t.Tests.object] = {}
                if self.error is not None and self.has is None:
                    updates["has"] = self.error
                if self.len is None and any(
                    v is not None
                    for v in (
                        self.length_gt,
                        self.length_gte,
                        self.length_lt,
                        self.length_lte,
                    )
                ):
                    min_len = 0
                    max_len = sys.maxsize
                    if self.length_gt is not None:
                        min_len = self.length_gt + 1
                    if self.length_gte is not None:
                        min_len = max(min_len, self.length_gte)
                    if self.length_lt is not None:
                        max_len = self.length_lt - 1
                    if self.length_lte is not None:
                        max_len = min(max_len, self.length_lte)
                    updates["len"] = (min_len, max_len)
                if updates:
                    return self.model_copy(update=updates)
                return self

        class ScopeParams(FlextModels.Value):
            """Parameters for temporary test scope configuration."""

            model_config = ConfigDict(populate_by_name=True)

            config: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Initial configuration values"),
            ]
            container: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Initial container/service mappings"),
            ]
            context: Annotated[
                Mapping[str, t.Tests.object] | None,
                Field(default=None, description="Initial context values"),
            ]
            cleanup: Annotated[
                t.Tests.Matcher.CleanupSpec | None,
                Field(default=None, description="Cleanup functions"),
            ]
            env: Annotated[
                t.Tests.Matcher.EnvironmentSpec | None,
                Field(default=None, description="Temporary environment variables"),
            ]
            cwd: Annotated[
                Path | str | None,
                Field(default=None, description="Temporary working directory"),
            ]

            @field_validator("cwd", mode="before")
            @classmethod
            def convert_cwd(cls, value: Path | str | None) -> Path | str | None:
                if isinstance(value, str):
                    return Path(value)
                return value

        class DeepMatchResult(FlextModels.Value):
            """Structured output for deep-match comparisons."""

            path: Annotated[
                str, Field(description="Path where match occurred or failed")
            ]
            expected: Annotated[
                t.Tests.Matcher.ValueSpec,
                Field(description="Expected value or predicate"),
            ]
            actual: Annotated[
                t.Tests.object | None,
                Field(default=None, description="Actual value found"),
            ]
            matched: Annotated[bool, Field(description="Whether match succeeded")]
            reason: Annotated[
                str, Field(default="", description="Reason for match failure")
            ]

        class Validate:
            """Centralized TypeAdapters for test data validation.

            All TypeAdapters used across flext_tests modules are defined here.
            Access via m.Tests.Validate.* with flat aliases.
            """

            DICT_ADAPTER: ClassVar[TypeAdapter[dict[str, t.Tests.object]]] = (
                TypeAdapter(dict[str, t.Tests.object])
            )
            LIST_ADAPTER: ClassVar[TypeAdapter[list[t.Tests.object]]] = TypeAdapter(
                list[t.Tests.object]
            )

        class Chain[TResult](FlextModels.Value):
            """Container for chained result assertions."""

            result: Annotated[r[TResult], Field(description="r being chained")]

        class TestScope(FlextModels.ArbitraryTypesModel):
            """Scope container for test configuration and runtime state."""

            __test__ = False

            config: Annotated[
                Mapping[str, t.Tests.object],
                Field(default_factory=dict, description="Configuration dictionary"),
            ]
            container: Annotated[
                Mapping[str, t.Tests.object],
                Field(default_factory=dict, description="Container/service mappings"),
            ]
            context: Annotated[
                Mapping[str, t.Tests.object],
                Field(default_factory=dict, description="Context values"),
            ]


m = FlextTestsModels

__all__ = ["FlextTestsModels", "m"]
