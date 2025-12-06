"""File management utilities for FLEXT ecosystem tests.

Provides comprehensive file operations for testing across the FLEXT ecosystem
with a simplified API using generalist methods with powerful optional parameters.

Supports:
- FlextResult: Automatically extracts value before serialization
- Pydantic models: Serializes to JSON/YAML via model_dump()
- Lists, dicts, Mappings: Proper JSON/YAML serialization
- Generic type loading: Load files directly into Pydantic models

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import warnings
from collections.abc import (
    Callable,
    Generator,
    Mapping,
    Mapping as ABCMapping,
    Sequence,
    Sized,
)
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self, TypeVar, cast, overload

import yaml
from pydantic import BaseModel, PrivateAttr

from flext_core import r
from flext_tests.base import s
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u

# TypeVar for Pydantic model loading (after imports for proper BaseModel reference)
TModel = TypeVar("TModel", bound=BaseModel)


class FlextTestsFiles(s[t.Tests.TestResultValue]):
    """Manages test files for FLEXT ecosystem testing.

    Extends FlextTestsServiceBase for consistent service patterns.

    Provides generalist file operations with powerful optional parameters:
    - `create()`: Create any file type with auto-detection
    - `read()`: Read any file type with FlextResult
    - `compare()`: Compare files with multiple modes
    - `info()`: Get comprehensive file information

    Example:
        from flext_tests import tf

        with tf() as files:
            # Auto-detect format from content type
            path = files.create({"key": "value"}, "config.json")
            result = files.read(path)

        # Or use context manager for multiple files
        with tf.files({"a": "text", "b": {"k": 1}}) as paths:
            assert paths["a"].exists()

    """

    # Re-export FileInfo from models for backward compatibility
    FileInfo: ClassVar[type[m.Tests.Files.FileInfo]] = m.Tests.Files.FileInfo

    # Mutable state using PrivateAttr (Pydantic pattern for frozen models)
    _base_dir: Path | None = PrivateAttr(default=None)
    _created_files: list[Path] = PrivateAttr(default_factory=list)
    _created_dirs: list[Path] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        base_dir: Path | None = None,
        **data: t.GeneralValueType,
    ) -> None:
        """Initialize file manager with optional base directory.

        Args:
            base_dir: Optional base directory for file operations.
                     If not provided, temporary directories are used.
            **data: Additional data passed to parent service.

        """
        super().__init__(**data)
        object.__setattr__(self, "_base_dir", base_dir)
        object.__setattr__(self, "_created_files", [])
        object.__setattr__(self, "_created_dirs", [])

    @property
    def base_dir(self) -> Path | None:
        """Get base directory."""
        return self._base_dir

    @property
    def created_files(self) -> list[Path]:
        """Get list of created files."""
        return self._created_files

    @property
    def created_dirs(self) -> list[Path]:
        """Get list of created directories."""
        return self._created_dirs

    def execute(self) -> r[t.Tests.TestResultValue]:
        """Execute service - returns success for file manager.

        FlextTestsFiles is a utility service that doesn't have a specific
        execution result. Returns success by default.
        """
        return r[t.Tests.TestResultValue].ok(None)

    # =========================================================================
    # STATIC CONVENIENCE METHODS - Direct usage without instantiation
    # =========================================================================

    @staticmethod
    def create_in(
        content: (
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel
            | r[str]
            | r[bytes]
            | r[t.Types.ConfigurationMapping]
            | r[Sequence[Sequence[str]]]
            | r[BaseModel]
        ),
        name: str,
        directory: Path,
        *,
        fmt: c.Tests.Files.FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        indent: int = c.Tests.Files.DEFAULT_JSON_INDENT,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        headers: list[str] | None = None,
        readonly: bool = False,
        extract_result: bool = True,
    ) -> Path:
        """Create file directly in directory - static convenience method.

        Supports FlextResult, Pydantic models, lists, dicts, and raw content.

        Args:
            content: File content (str, bytes, dict, list, BaseModel, or r[T])
            name: Filename
            directory: Target directory
            fmt: Format override ("auto", "text", "bin", "json", "yaml", "csv")
            enc: Encoding (default: utf-8)
            indent: JSON/YAML indent (default: 2)
            delim: CSV delimiter (default: ",")
            headers: CSV headers (default: None)
            readonly: Create as read-only (default: False)
            extract_result: Auto-extract FlextResult value (default: True)

        Returns:
            Path to created file.

        Examples:
            # Simple text file
            path = tf.create_in("content", "file.txt", output_dir)

            # Pydantic model
            path = tf.create_in(user_model, "user.json", output_dir)

            # FlextResult
            result = service.get_data()
            path = tf.create_in(result, "data.json", output_dir)

        """
        manager = FlextTestsFiles(base_dir=directory)
        return manager.create(
            content,
            name,
            directory=None,
            fmt=fmt,
            enc=enc,
            indent=indent,
            delim=delim,
            headers=headers,
            readonly=readonly,
            extract_result=extract_result,
        )

    @staticmethod
    def assert_exists(
        path: Path,
        msg: str | None = None,
        *,
        is_file: bool | None = None,
        is_dir: bool | None = None,
        not_empty: bool | None = None,
        readable: bool | None = None,
        writable: bool | None = None,
    ) -> Path:
        """Generalized file existence assertion - ALL file validations in ONE method.

        Consolidates: assert_exists(), assert_file(), assert_dir(), assert_not_empty()
        into single method with optional parameters.

        Args:
            path: File or directory path to check
            msg: Custom error message
            is_file: Assert is file (True) or not file (False)
            is_dir: Assert is directory (True) or not directory (False)
            not_empty: Assert file/dir is not empty (True) or empty (False)
            readable: Assert is readable (True)
            writable: Assert is writable (True)

        Returns:
            Path if all validations pass

        Examples:
            tf.assert_exists(path)                    # Just exists
            tf.assert_exists(path, is_file=True)      # Exists and is file
            tf.assert_exists(path, not_empty=True)    # Exists and not empty
            tf.assert_exists(path, is_dir=True, writable=True)  # Dir and writable

        """
        if not path.exists():
            error_msg = msg or c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=path)
            raise AssertionError(error_msg)

        if is_file is not None:
            if is_file and not path.is_file():
                raise AssertionError(msg or f"Path {path} is not a file")
            if not is_file and path.is_file():
                raise AssertionError(msg or f"Path {path} should not be a file")

        if is_dir is not None:
            if is_dir and not path.is_dir():
                raise AssertionError(msg or f"Path {path} is not a directory")
            if not is_dir and path.is_dir():
                raise AssertionError(msg or f"Path {path} should not be a directory")

        if not_empty is not None:
            if not_empty:
                if path.is_file() and path.stat().st_size == 0:
                    raise AssertionError(msg or f"File {path} is empty")
                if path.is_dir() and not any(path.iterdir()):
                    raise AssertionError(msg or f"Directory {path} is empty")
            else:
                if path.is_file() and path.stat().st_size > 0:
                    raise AssertionError(msg or f"File {path} is not empty")
                if path.is_dir() and any(path.iterdir()):
                    raise AssertionError(msg or f"Directory {path} is not empty")

        if (
            readable is not None
            and readable
            and path.is_file()
            and not os.access(path, os.R_OK)
        ):
            raise AssertionError(msg or f"File {path} is not readable")

        if writable is not None and writable:
            if path.is_file() and not os.access(path, os.W_OK):
                raise AssertionError(msg or f"File {path} is not writable")
            if path.is_dir() and not os.access(path, os.W_OK):
                raise AssertionError(msg or f"Directory {path} is not writable")

        return path

    # =========================================================================
    # CORE PUBLIC API - 4 Methods + cleanup
    # =========================================================================

    def create(
        self,
        content: (
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel
            | r[str]
            | r[bytes]
            | r[t.Types.ConfigurationMapping]
            | r[Sequence[Sequence[str]]]
            | r[BaseModel]
        ),
        name: str = c.Tests.Files.DEFAULT_FILENAME,
        directory: Path | None = None,
        *,
        fmt: c.Tests.Files.FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        indent: int = c.Tests.Files.DEFAULT_JSON_INDENT,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        headers: list[str] | None = None,
        readonly: bool = False,
        extract_result: bool = True,
    ) -> Path:
        r"""Create file with auto-detection or explicit format.

        Supports FlextResult, Pydantic models, lists, dicts, and raw content.

        Args:
            content: Content - type determines default format:
                - str: text file
                - bytes: binary file
                - dict/Mapping: JSON file
                - list[list[str]]: CSV file
                - BaseModel: JSON file (via model_dump())
                - r[T]: Extracts value if success (if extract_result=True)
            name: Filename (extension hints format)
            directory: Directory (uses base_dir or temp if None)
            fmt: Format override ("auto", "text", "bin", "json", "yaml", "csv")
            enc: Encoding (default: utf-8)
            indent: JSON/YAML indent (default: 2)
            delim: CSV delimiter (default: ",")
            headers: CSV headers (default: None)
            readonly: Create as read-only (default: False)
            extract_result: Auto-extract FlextResult value (default: True)

        Returns:
            Path to created file.

        Raises:
            ValueError: If FlextResult is failure and extract_result=True

        Examples:
            # Text file
            path = tf().create("hello", "test.txt")

            # JSON file (auto-detected from dict)
            path = tf().create({"key": "value"}, "config.json")

            # Pydantic model (auto-detected as JSON)
            path = tf().create(user_model, "user.json")

            # FlextResult with auto-extraction
            result = service.get_config()  # r[dict]
            path = tf().create(result, "config.json")

            # CSV file (auto-detected from list[list])
            path = tf().create([["a", "b"], ["1", "2"]], "data.csv",
                              headers=["col1", "col2"])

            # Binary file
            path = tf().create(b"\x00\x01", "data.bin", fmt="bin")

        """
        # Extract from FlextResult BEFORE validation if extract_result=True
        # This ensures CreateParams receives the unwrapped content, not the FlextResult
        content_to_validate = content
        if extract_result and u.is_type(content, "result"):
            content_result = cast("r[t.Tests.Files.FileContent]", content)
            if content_result.is_failure:
                error_msg = content_result.error or "FlextResult failure"
                raise ValueError(
                    f"Cannot create file from failed FlextResult: {error_msg}",
                )
            content_to_validate = content_result.unwrap()

        # Validate and compute parameters using CreateParams model with u.Model.from_kwargs()
        # All parameters validated via Pydantic 2 Field constraints (ge, min_length, max_length) - no manual validation
        # Pydantic 2 field_validators handle type conversions automatically (str → Path, etc.)
        params_result = u.Model.from_kwargs(
            m.Tests.Files.CreateParams,
            content=content_to_validate,
            name=name,
            directory=directory,
            fmt=fmt,
            enc=enc,
            indent=indent,
            delim=delim,
            headers=headers,
            readonly=readonly,
            extract_result=extract_result,
        )
        if params_result.is_failure:
            error_msg = f"Invalid parameters for file creation: {params_result.error}"
            raise ValueError(error_msg) from None
        params = params_result.value

        target_dir = self._resolve_directory(params.directory)
        file_path = target_dir / params.name

        # Content already extracted if needed - use validated content
        actual_content = params.content

        # Convert Pydantic model to dict using u.Model.to_dict()
        if isinstance(actual_content, BaseModel):
            actual_content = u.Model.to_dict(actual_content)

        # Auto-detect format using utilities
        # Convert Sequence[Sequence[str]] to list[list[str]] if needed for type compatibility
        content_for_detect: str | bytes | t.Types.ConfigurationMapping | list[list[str]]
        if u.is_type(actual_content, "sequence") and not isinstance(
            actual_content,
            (str, bytes),
        ):
            # Convert nested Sequence to list[list[str]] for type compatibility
            # Runtime check needed to distinguish nested sequences from flat sequences
            if all(u.is_type(row, "sequence") for row in actual_content):
                content_for_detect = [list(row) for row in actual_content]
            else:
                # Not a nested sequence - cast to expected type for detection
                # detect_format accepts various types including Sequence
                content_for_detect = cast(
                    "str | bytes | t.Types.ConfigurationMapping | list[list[str]]",
                    actual_content,
                )
        elif u.is_type(actual_content, "mapping") or isinstance(actual_content, dict):
            # Mapping content - cast to ConfigurationMapping
            content_for_detect = cast(
                "str | bytes | t.Types.ConfigurationMapping | list[list[str]]",
                actual_content,
            )
        elif isinstance(actual_content, (str, bytes)):
            # String/bytes content
            content_for_detect = actual_content
        elif isinstance(actual_content, BaseModel):
            # BaseModel - convert to dict first for detection
            content_for_detect = cast(
                "str | bytes | t.Types.ConfigurationMapping | list[list[str]]",
                u.Model.dump(actual_content),
            )
        else:
            # Fallback - cast to expected union type
            content_for_detect = cast(
                "str | bytes | t.Types.ConfigurationMapping | list[list[str]]",
                actual_content,
            )
        actual_fmt = u.Tests.Files.detect_format(
            content_for_detect,
            params.name,
            params.fmt,
        )

        # Create based on format using validated params
        if actual_fmt == c.Tests.Files.Format.BIN:
            if isinstance(actual_content, bytes):
                _ = file_path.write_bytes(actual_content)
            else:
                _ = file_path.write_bytes(str(actual_content).encode(params.enc))
        elif actual_fmt == c.Tests.Files.Format.JSON:
            # Convert Mapping to dict if needed using u.Mapper.to_dict()
            # Only call to_dict if it's a Mapping but not a dict
            if u.is_type(actual_content, "mapping") and not isinstance(
                actual_content,
                dict,
            ):
                data = u.Mapper.to_dict(
                    cast("Mapping[str, t.GeneralValueType]", actual_content),
                )
            elif isinstance(actual_content, dict):
                data = actual_content
            else:
                # Fallback for other types (should not happen for JSON)
                data = cast("dict[str, t.GeneralValueType]", actual_content)
            _ = file_path.write_text(
                json.dumps(data, indent=params.indent, ensure_ascii=False),
                encoding=params.enc,
            )
        elif actual_fmt == c.Tests.Files.Format.YAML:
            # Convert Mapping to dict if needed using u.Mapper.to_dict()
            # Only call to_dict if it's a Mapping but not a dict
            if u.is_type(actual_content, "mapping") and not isinstance(
                actual_content,
                dict,
            ):
                data = u.Mapper.to_dict(
                    cast("Mapping[str, t.GeneralValueType]", actual_content),
                )
            elif isinstance(actual_content, dict):
                data = actual_content
            else:
                # Fallback for other types (should not happen for YAML)
                data = cast("dict[str, t.GeneralValueType]", actual_content)
            _ = file_path.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding=params.enc,
            )
        elif actual_fmt == c.Tests.Files.Format.CSV:
            # Convert Sequence[Sequence[str]] to list[list[str]] for write_csv
            csv_content: list[list[str]]
            if u.is_type(actual_content, "sequence") and not isinstance(
                actual_content,
                (str, bytes),
            ):
                # Check if nested sequence and convert to list[list[str]]
                # Runtime check needed to distinguish nested sequences from flat sequences
                if all(
                    u.is_type(row, "sequence") and not isinstance(row, (str, bytes))
                    for row in actual_content
                ):
                    csv_content = [list(row) for row in actual_content]
                else:
                    # Not a nested sequence - wrap in list (single column CSV)
                    csv_content = [[str(item)] for item in actual_content]
            else:
                # Not a sequence - convert to CSV format (single column)
                csv_content = [[str(actual_content)]]
            u.Tests.Files.write_csv(
                file_path,
                csv_content,
                params.headers,
                params.delim,
                params.enc,
            )
        else:  # text
            _ = file_path.write_text(str(actual_content), encoding=params.enc)

        # Set permissions using validated params
        if params.readonly:
            file_path.chmod(c.Tests.Files.PERMISSION_READONLY_FILE)

        self.created_files.append(file_path)
        return file_path

    @overload
    def read(
        self,
        path: Path,
        *,
        model_cls: None = None,
        fmt: c.Tests.Files.FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]]: ...

    @overload
    def read(
        self,
        path: Path,
        *,
        model_cls: type[TModel],
        fmt: c.Tests.Files.FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[TModel]: ...

    def read(
        self,
        path: Path,
        *,
        model_cls: type[TModel] | None = None,
        fmt: c.Tests.Files.FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]] | r[TModel]:
        """Read file with auto-detection or explicit format.

        Supports loading directly into Pydantic models when model_cls is provided.

        Args:
            path: File path
            model_cls: Optional Pydantic model class to deserialize into
            fmt: Format ("auto" detects from extension)
            enc: Encoding (default: utf-8)
            delim: CSV delimiter (default: ",")
            has_headers: CSV has headers (default: True)

        Returns:
            FlextResult with content or model instance.

        Examples:
            # Read text
            result = tf().read(path)
            if result.is_success:
                text = result.unwrap()

            # Read JSON
            result = tf().read(Path("config.json"))
            data = result.unwrap()  # dict

            # Read JSON into Pydantic model
            result = tf().read(Path("user.json"), model_cls=UserModel)
            user = result.unwrap()  # UserModel instance

            # Read YAML into Pydantic model
            result = tf().read(Path("config.yaml"), model_cls=ConfigModel)
            config = result.unwrap()  # ConfigModel instance

            # Read CSV
            result = tf().read(Path("data.csv"))
            rows = result.unwrap()  # list[list[str]]

        """
        # Validate and compute parameters using ReadParams model with u.Model.from_kwargs()
        # All parameters validated via Pydantic 2 Field constraints - no manual validation needed
        # Pydantic 2 field_validators handle type conversions automatically (str → Path, etc.)
        # Pass all parameters as kwargs to avoid "already assigned" error
        params_result = u.Model.from_kwargs(
            m.Tests.Files.ReadParams,
            path=path,
            fmt=fmt,
            enc=enc,
            delim=delim,
            has_headers=has_headers,
            **({"model_cls": model_cls} if model_cls is not None else {}),
        )
        if params_result.is_failure:
            error_msg = f"Invalid parameters for file read: {params_result.error}"
            if model_cls is not None:
                return r[TModel].fail(error_msg)
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                error_msg,
            )
        params = params_result.value

        if not params.path.exists():
            if params.model_cls is not None:
                return r[TModel].fail(
                    c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.path),
                )
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.path),
            )

        actual_fmt = u.Tests.Files.detect_format_from_path(params.path, params.fmt)

        try:
            if actual_fmt == c.Tests.Files.Format.BIN:
                content: (
                    str | bytes | t.Types.ConfigurationMapping | list[list[str]]
                ) = params.path.read_bytes()
            elif actual_fmt == c.Tests.Files.Format.JSON:
                text = params.path.read_text(encoding=params.enc)
                content = json.loads(text)
            elif actual_fmt == c.Tests.Files.Format.YAML:
                text = params.path.read_text(encoding=params.enc)
                content = yaml.safe_load(text)
            elif actual_fmt == c.Tests.Files.Format.CSV:
                content = u.Tests.Files.read_csv(
                    params.path,
                    params.delim,
                    params.enc,
                    has_headers=params.has_headers,
                )
            else:  # text
                content = params.path.read_text(encoding=params.enc)

            # If model_cls provided, use u.Model.load() for validation
            # u.Model.load() accepts ConfigurationMapping (more generic than from_dict)
            if params.model_cls is not None:
                # Type narrowing: check if content is dict/mapping
                if not (u.is_type(content, "dict") or u.is_type(content, "mapping")):
                    return r[TModel].fail(
                        f"Cannot load model from non-dict content: {type(content)}",
                    )
                # Convert to dict if needed, then to ConfigurationMapping
                if isinstance(content, dict):
                    content_dict = content
                elif u.is_type(content, "mapping"):
                    content_dict = u.Mapper.to_dict(
                        cast("Mapping[str, t.GeneralValueType]", content),
                    )
                else:
                    # Should not reach here due to check above, but satisfy type checker
                    return r[TModel].fail(
                        f"Cannot convert content to dict: {type(content)}",
                    )
                # Use u.Model.load() which accepts ConfigurationMapping and returns r[T_Model]
                content_mapping = cast("t.Types.ConfigurationMapping", content_dict)
                result = u.Model.load(params.model_cls, content_mapping)
                # Type assertion needed - load() returns r[T_Model] but type checker needs r[TModel]
                return cast("r[TModel]", result)

            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].ok(
                content,
            )
        except json.JSONDecodeError as e:
            if params.model_cls is not None:
                return r[TModel].fail(c.Tests.Files.ERROR_INVALID_JSON.format(error=e))
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                c.Tests.Files.ERROR_INVALID_JSON.format(error=e),
            )
        except yaml.YAMLError as e:
            if params.model_cls is not None:
                return r[TModel].fail(c.Tests.Files.ERROR_INVALID_YAML.format(error=e))
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                c.Tests.Files.ERROR_INVALID_YAML.format(error=e),
            )
        except UnicodeDecodeError as e:
            if params.model_cls is not None:
                return r[TModel].fail(c.Tests.Files.ERROR_ENCODING.format(error=e))
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                c.Tests.Files.ERROR_ENCODING.format(error=e),
            )
        except OSError as e:
            if params.model_cls is not None:
                return r[TModel].fail(c.Tests.Files.ERROR_READ.format(error=e))
            return r[str | bytes | t.Types.ConfigurationMapping | list[list[str]]].fail(
                c.Tests.Files.ERROR_READ.format(error=e),
            )

    def compare(
        self,
        file1: Path,
        file2: Path,
        *,
        mode: c.Tests.Files.CompareModeLiteral = "content",
        ignore_ws: bool = False,
        ignore_case: bool = False,
        pattern: str | None = None,
        deep: bool = True,
        keys: list[str] | None = None,
        exclude_keys: list[str] | None = None,
    ) -> r[bool]:
        """Compare two files.

        Args:
            file1: First file
            file2: Second file
            mode: "content" | "size" | "hash" | "lines"
            ignore_ws: Ignore whitespace
            ignore_case: Case-insensitive
            pattern: Check if both contain pattern
            keys: Only compare these keys (for dict/JSON content)
            exclude_keys: Exclude these keys from comparison (for dict/JSON content)
            deep: Use deep comparison for nested structures (default: True)

        Returns:
            FlextResult[bool] - True if match.

        Examples:
            # Content comparison
            result = tf().compare(file1, file2)

            # Hash comparison (faster for large files)
            result = tf().compare(file1, file2, mode="hash")

            # Check if both contain pattern
            result = tf().compare(file1, file2, pattern="ERROR")

            # Deep comparison with key filtering (for JSON/YAML)
            result = tf().compare(file1, file2, keys=["name", "email"])
            result = tf().compare(file1, file2, exclude_keys=["timestamp"])

        """
        # Validate and compute parameters using CompareParams model with u.Model.from_kwargs()
        # All parameters validated via Pydantic 2 Field constraints - no manual validation needed
        # Pydantic 2 field_validators handle type conversions automatically (str → Path, etc.)
        params_result = u.Model.from_kwargs(
            m.Tests.Files.CompareParams,
            file1=file1,
            file2=file2,
            mode=mode,
            ignore_ws=ignore_ws,
            ignore_case=ignore_case,
            pattern=pattern,
            deep=deep,
            keys=keys,
            exclude_keys=exclude_keys,
        )
        if params_result.is_failure:
            return r[bool].fail(
                f"Invalid parameters for file comparison: {params_result.error}",
            )
        params = params_result.value

        if not params.file1.exists():
            return r[bool].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.file1),
            )
        if not params.file2.exists():
            return r[bool].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.file2),
            )

        try:
            # Pattern matching - check if both files contain pattern
            if params.pattern is not None:
                text1 = params.file1.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
                text2 = params.file2.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
                return r[bool].ok(params.pattern in text1 and params.pattern in text2)

            # Mode-based comparison using match/case (Python 3.10+)
            match params.mode:
                case "size":
                    return r[bool].ok(
                        params.file1.stat().st_size == params.file2.stat().st_size,
                    )
                case "hash":
                    hash1 = u.Tests.Files.compute_hash(params.file1)
                    hash2 = u.Tests.Files.compute_hash(params.file2)
                    return r[bool].ok(hash1 == hash2)
                case "lines":
                    return self._compare_lines(params)
                case _:
                    return self._compare_content(params)
        except OSError as e:
            return r[bool].fail(c.Tests.Files.ERROR_COMPARE.format(error=e))

    def _compare_lines(self, params: m.Tests.Files.CompareParams) -> r[bool]:
        """Compare files line by line with optional normalization."""
        lines1 = params.file1.read_text(
            encoding=c.Tests.Files.DEFAULT_ENCODING,
        ).splitlines()
        lines2 = params.file2.read_text(
            encoding=c.Tests.Files.DEFAULT_ENCODING,
        ).splitlines()
        if params.ignore_ws:
            lines1 = [line.strip() for line in lines1]
            lines2 = [line.strip() for line in lines2]
        if params.ignore_case:
            lines1 = [line.lower() for line in lines1]
            lines2 = [line.lower() for line in lines2]
        return r[bool].ok(lines1 == lines2)

    def _compare_content(self, params: m.Tests.Files.CompareParams) -> r[bool]:
        """Compare file content with optional deep/structured comparison."""
        content1_raw = params.file1.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
        content2_raw = params.file2.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)

        # Attempt deep comparison for JSON/YAML when deep=True
        if params.deep:
            deep_result = self._try_deep_compare(
                content1_raw,
                content2_raw,
                params.keys,
                params.exclude_keys,
            )
            if deep_result is not None:
                return deep_result

        # String comparison (fallback or if deep=False)
        content1 = (
            re.sub(r"\s+", "", content1_raw) if params.ignore_ws else content1_raw
        )
        content2 = (
            re.sub(r"\s+", "", content2_raw) if params.ignore_ws else content2_raw
        )
        if params.ignore_case:
            content1 = content1.lower()
            content2 = content2.lower()
        return r[bool].ok(content1 == content2)

    def _try_deep_compare(
        self,
        content1_raw: str,
        content2_raw: str,
        keys: list[str] | None,
        exclude_keys: list[str] | None,
    ) -> r[bool] | None:
        """Try to parse and deeply compare content as JSON or YAML.

        Returns None if content cannot be parsed as structured data.
        """
        # Try JSON first (faster)
        parsed = self._try_parse_both(content1_raw, content2_raw, "json")
        if parsed is None:
            # Try YAML as fallback
            parsed = self._try_parse_both(content1_raw, content2_raw, "yaml")
        if parsed is None:
            return None

        dict1, dict2 = parsed
        # Apply key filtering if specified
        dict1, dict2 = self._apply_key_filtering(dict1, dict2, keys, exclude_keys)
        return r[bool].ok(u.Mapper.deep_eq(dict1, dict2))

    def _try_parse_both(
        self,
        content1: str,
        content2: str,
        fmt: str,
    ) -> tuple[dict[str, t.GeneralValueType], dict[str, t.GeneralValueType]] | None:
        """Try to parse both contents as dicts in given format."""
        try:
            match fmt:
                case "json":
                    dict1 = json.loads(content1)
                    dict2 = json.loads(content2)
                case "yaml":
                    dict1 = yaml.safe_load(content1)
                    dict2 = yaml.safe_load(content2)
                case _:
                    return None
            if u.is_type(dict1, "dict") and u.is_type(dict2, "dict"):
                return (dict1, dict2)
        except (json.JSONDecodeError, yaml.YAMLError, ValueError, TypeError):
            pass
        return None

    def _apply_key_filtering(
        self,
        dict1: dict[str, t.GeneralValueType],
        dict2: dict[str, t.GeneralValueType],
        keys: list[str] | None,
        exclude_keys: list[str] | None,
    ) -> tuple[dict[str, t.GeneralValueType], dict[str, t.GeneralValueType]]:
        """Apply key filtering to both dicts if specified."""
        if keys is None and exclude_keys is None:
            return dict1, dict2

        filter_keys_set = set(keys) if keys is not None else None
        exclude_keys_set = set(exclude_keys) if exclude_keys is not None else None

        result1 = u.Mapper.transform(
            dict1,
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        result2 = u.Mapper.transform(
            dict2,
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )

        if result1.is_success and result2.is_success:
            return result1.value, result2.value
        return dict1, dict2

    def info(
        self,
        path: Path,
        *,
        compute_hash: bool = False,
        detect_fmt: bool = True,
        parse_content: bool = False,
        validate_model: type[BaseModel] | None = None,
    ) -> r[m.Tests.Files.FileInfo]:
        """Get comprehensive file information.

        Args:
            path: File path
            compute_hash: Compute SHA256 (default: False)
            detect_fmt: Auto-detect format (default: True)
            parse_content: Parse content and include metadata (default: False)
            validate_model: Pydantic model to validate content against (default: None)

        Returns:
            FlextResult[FileInfo] with info or error.

        Examples:
            result = tf().info(path)
            if result.is_success:
                info = result.unwrap()
                print(f"Size: {info.size_human}")
                print(f"Format: {info.fmt}")

            # With content parsing
            result = tf().info(path, parse_content=True)
            if result.is_success and result.value.content_meta:
                print(f"Keys: {result.value.content_meta.key_count}")

            # With model validation
            result = tf().info(path, validate_model=UserModel)
            if result.is_success and result.value.content_meta:
                print(f"Valid: {result.value.content_meta.model_valid}")

        """
        # Validate and compute parameters using InfoParams model with u.Model.from_kwargs()
        # All parameters validated via Pydantic 2 Field constraints - no manual validation needed
        # Pydantic 2 field_validators handle type conversions automatically (str → Path, etc.)
        params_result = u.Model.from_kwargs(
            m.Tests.Files.InfoParams,
            path=path,
            compute_hash=compute_hash,
            detect_fmt=detect_fmt,
            parse_content=parse_content,
            validate_model=validate_model,
        )
        if params_result.is_failure:
            return r[m.Tests.Files.FileInfo].fail(
                f"Invalid parameters for file info: {params_result.error}",
            )
        params = params_result.value

        if not params.path.exists():
            return r[m.Tests.Files.FileInfo].ok(
                m.Tests.Files.FileInfo(exists=False, path=params.path),
            )

        try:
            # Use validated params throughout
            stat = params.path.stat()
            size = stat.st_size
            size_human = u.Tests.Files.format_size(size)

            # Read content for analysis
            try:
                text = params.path.read_text(
                    encoding=c.Tests.Files.DEFAULT_ENCODING,
                    errors="replace",
                )
                lines = text.count("\n") + 1 if text else 0
                is_empty = len(text.strip()) == 0
                first_line = text.split("\n")[0] if text else ""
                encoding = c.Tests.Files.DEFAULT_ENCODING
            except UnicodeDecodeError:
                # Binary file
                text = ""
                lines = 0
                is_empty = size == 0
                first_line = ""
                encoding = c.Tests.Files.DEFAULT_BINARY_ENCODING

            # Format detection
            fmt: c.Tests.Files.FormatLiteral = "unknown"
            if params.detect_fmt:
                fmt = cast(
                    "c.Tests.Files.FormatLiteral",
                    u.Tests.Files.detect_format_from_path(params.path, "auto"),
                )

            # Permissions
            permissions = stat.st_mode
            is_readonly = not (permissions & 0o200)

            # Hash
            sha256 = (
                u.Tests.Files.compute_hash(params.path) if params.compute_hash else None
            )

            # Content metadata parsing
            content_meta: m.Tests.Files.ContentMeta | None = None
            if params.parse_content or params.validate_model:
                content_meta = self._parse_content_metadata(
                    path=params.path,
                    text=text,
                    fmt=fmt,
                    validate_model=params.validate_model,
                )

            return r[m.Tests.Files.FileInfo].ok(
                m.Tests.Files.FileInfo(
                    exists=True,
                    path=params.path,
                    size=size,
                    size_human=size_human,
                    lines=lines,
                    encoding=encoding,
                    is_empty=is_empty,
                    first_line=first_line,
                    fmt=fmt,
                    is_valid=True,
                    modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    permissions=permissions,
                    is_readonly=is_readonly,
                    sha256=sha256,
                    content_meta=content_meta,
                ),
            )
        except OSError as e:
            return r[m.Tests.Files.FileInfo].fail(
                c.Tests.Files.ERROR_INFO.format(error=e),
            )

    def batch[TModel: BaseModel](
        self,
        files: t.Tests.Files.BatchFiles,
        *,
        directory: Path | None = None,
        operation: c.Tests.Files.OperationLiteral = "create",
        model: type[TModel] | None = None,
        on_error: c.Tests.Files.ErrorModeLiteral = "collect",
        parallel: bool = False,
    ) -> r[m.Tests.Files.BatchResult]:
        """Batch file operations.

        Uses u.Collection.batch() for batch processing with error handling.

        Args:
            files: Mapping[str, FileContent] or Sequence[tuple[str, FileContent]]
            directory: Target directory for create operations
            operation: "create", "read", or "delete"
            model: Optional model class for read operations
            on_error: Error handling mode ("stop", "skip", "collect")
            parallel: Run operations in parallel (not implemented yet)

        Returns:
            r[m.Tests.Files.BatchResult] with results and errors

        Examples:
            # Batch create
            result = tf().batch({
                "file1.txt": "content1",
                "file2.json": {"key": "value"},
                "file3.yaml": config_model,
            }, directory=tmp_path)

            # Batch read with model
            file_paths = {"user1.json": Path("user1.json"), ...}
            result = tf().batch(
                file_paths,
                operation="read",
                model=UserModel,
            )

        """
        # Validate and compute parameters using BatchParams model with u.Model.from_kwargs()
        # All parameters validated via Pydantic 2 Field constraints - no manual validation needed
        # Pydantic 2 field_validators handle type conversions automatically (str → Path, etc.)
        params_result = u.Model.from_kwargs(
            m.Tests.Files.BatchParams,
            files=files,
            directory=directory,
            operation=operation,
            model=model,
            on_error=on_error,
            parallel=parallel,
        )
        if params_result.is_failure:
            return r[m.Tests.Files.BatchResult].fail(
                f"Invalid parameters for batch operation: {params_result.error}",
            )
        params = params_result.value

        # Convert BatchFiles to dict - BatchFiles can be Mapping or Sequence
        files_dict: dict[str, t.Tests.Files.FileContent]
        if isinstance(params.files, Mapping):
            # Already a Mapping - convert to dict if needed
            files_dict = (
                u.Mapper.to_dict(params.files)
                if not isinstance(params.files, dict)
                else params.files
            )
        elif isinstance(params.files, Sequence):
            # BatchFiles is Sequence[tuple[str, FileContent]] - convert to dict
            files_dict = dict(params.files)
        else:
            # Invalid type - should not happen due to BatchParams validation
            return r[m.Tests.Files.BatchResult].fail(
                f"Invalid BatchFiles type: {type(params.files)}",
            )

        # Convert error mode from ErrorModeLiteral to string for u.Collection.batch()
        error_mode_str = "collect" if params.on_error == "collect" else "fail"

        def process_one(
            name_and_content: tuple[str, t.Tests.Files.FileContent],
        ) -> r[Path]:
            """Process single file operation."""
            name, content = name_and_content
            match params.operation:
                case "create":
                    try:
                        path = self.create(content, name, params.directory)
                        return r[Path].ok(path)
                    except Exception as e:
                        return r[Path].fail(f"Failed to create {name}: {e}")
                case "read":
                    # For read, content should be Path or str - wrap in Path() for type safety
                    path = (
                        Path(content)
                        if isinstance(content, (Path, str))
                        else Path(name)
                    )
                    read_result = self.read(path, model_cls=params.model)
                    if read_result.is_success:
                        # Return the path, not the content (BatchResult expects Path)
                        return r[Path].ok(path)
                    return r[Path].fail(read_result.error or f"Failed to read {name}")
                case "delete":
                    # For delete, content should be Path or str - wrap in Path() for type safety
                    path = (
                        Path(content)
                        if isinstance(content, (Path, str))
                        else Path(name)
                    )
                    try:
                        Path(path).unlink(missing_ok=True)
                        return r[Path].ok(Path(path))
                    except Exception as e:
                        return r[Path].fail(f"Failed to delete {name}: {e}")
                case _:
                    return r[Path].fail(f"Unknown operation: {params.operation}")

        # Use u.Collection.batch() for batch processing
        # u.Collection.batch() handles Result unwrapping automatically
        # Returns r[t.Types.BatchResultDict] with results as direct values (not Results)
        items_list = list(files_dict.items())
        # Explicit type annotation helps mypy infer the generic R parameter
        operation_fn: Callable[
            [tuple[str, t.Tests.Files.FileContent]], Path | r[Path]
        ] = process_one
        batch_result_dict = u.Collection.batch(
            items_list,
            operation_fn,
            on_error=error_mode_str,
            _parallel=params.parallel,
        )

        # Handle batch failure (on_error="fail" mode)
        if batch_result_dict.is_failure:
            return r[m.Tests.Files.BatchResult].fail(
                batch_result_dict.error or "Batch operation failed",
            )

        # Extract results from batch result dict
        # u.Collection.batch() returns BatchResultDict with:
        # - results: list[R] (direct values, not Results - unwrapped automatically)
        # - errors: list[tuple[int, str]] (index, error_message)
        # - total, success_count, error_count
        batch_data = batch_result_dict.unwrap()
        results = batch_data.get("results", [])
        errors = batch_data.get("errors", [])
        total = batch_data.get("total", len(files_dict))
        success_count = batch_data.get("success_count", 0)
        error_count = batch_data.get("error_count", 0)

        # Convert results to Path list and errors to dict
        # u.Collection.batch() already unwraps Results, so results contains Path directly
        succeeded: list[Path] = []
        failed: dict[str, str] = {}

        # Process successful results (direct Path values, not wrapped in Result)
        for result in results:
            if isinstance(result, Path):
                succeeded.append(result)
            # Should not happen - u.Collection.batch() unwraps Results
            # But handle gracefully if it does
            elif u.is_type(result, "result"):
                result_typed = cast("r[Path]", result)
                if result_typed.is_success:
                    succeeded.append(result_typed.unwrap())

        # Process errors from batch result (indexed errors)
        for idx, error_msg in errors:
            if idx < len(items_list):
                name, _ = items_list[idx]
                failed[name] = error_msg

        return r[m.Tests.Files.BatchResult].ok(
            m.Tests.Files.BatchResult(
                succeeded=succeeded,
                failed=failed,
                total=total,
                success_count=success_count,
                failure_count=error_count,
            ),
        )

    def cleanup(self) -> None:
        """Clean up all created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                # Restore permissions if needed
                try:
                    file_path.chmod(c.Tests.Files.PERMISSION_WRITABLE_FILE)
                except OSError:
                    pass
                try:
                    file_path.unlink()
                except OSError:
                    pass

        for dir_path in self.created_dirs:
            if dir_path.exists():
                try:
                    # Restore permissions recursively
                    for item in dir_path.rglob("*"):
                        try:
                            perm = (
                                c.Tests.Files.PERMISSION_WRITABLE_DIR
                                if item.is_dir()
                                else c.Tests.Files.PERMISSION_WRITABLE_FILE
                            )
                            item.chmod(perm)
                        except OSError:
                            pass
                    dir_path.chmod(c.Tests.Files.PERMISSION_WRITABLE_DIR)
                    shutil.rmtree(dir_path)
                except OSError:
                    pass

        self.created_files.clear()
        self.created_dirs.clear()

    # =========================================================================
    # CLASS-LEVEL CONTEXT MANAGER
    # =========================================================================

    @classmethod
    @contextmanager
    def files(
        cls,
        content: dict[
            str,
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ],
        *,
        directory: Path | None = None,
        ext: str | None = None,
        extract_result: bool = True,
        **kwargs: object,
    ) -> Generator[dict[str, Path]]:
        """Create temporary files with auto-cleanup.

        Supports Pydantic models, dicts, lists, and raw content.

        Args:
            content: Dict mapping names to content (str, bytes, dict, list, BaseModel)
            directory: Base directory (temp if None)
            ext: Default extension if not in name
            extract_result: Auto-extract FlextResult values (default: True)
            **kwargs: Passed to create()

        Yields:
            Dict mapping names to paths.

        Examples:
            # Basic usage
            with tf.files({"a": "text", "b": {"key": 1}}) as paths:
                assert paths["a"].exists()
                assert paths["b"].suffix == ".json"  # auto-detected

            # With Pydantic models
            with tf.files({"user": user_model, "config": config_model}) as paths:
                assert paths["user"].suffix == ".json"
                assert paths["config"].suffix == ".json"

        """
        manager = cls()
        if directory is not None:
            object.__setattr__(manager, "_base_dir", directory)
        with manager:
            paths: dict[str, Path] = {}
            default_ext = ext or c.Tests.Files.DEFAULT_EXTENSION
            for name, data in content.items():
                filename = name if "." in name else f"{name}{default_ext}"
                # Determine if we need to adjust extension based on content type
                if "." not in name and (
                    u.is_type(data, "dict") or isinstance(data, BaseModel)
                ):
                    filename = f"{name}.json"
                else:
                    # Check if data is a nested sequence (CSV format)
                    is_nested_sequence = (
                        "." not in name
                        and u.is_type(data, "sequence")
                        and not isinstance(data, (str, bytes))
                        # Runtime check needed to distinguish nested sequences from flat sequences
                        and all(u.is_type(row, "sequence") for row in data)
                        # Check length safely - only if data has __len__ (Sized protocol)
                        and isinstance(data, Sized)
                        and len(data) > 0
                    )
                    if is_nested_sequence:
                        filename = f"{name}.csv"
                # Validate kwargs using CreateKwargsParams model before passing to create()
                # This ensures type safety and follows FLEXT patterns (Pydantic validation)
                # Always validate - if validation fails, use defaults from CreateKwargsParams
                kwargs_result = u.Model.from_kwargs(
                    m.Tests.Files.CreateKwargsParams,
                    **kwargs,
                )
                # Use validated parameters or defaults
                if kwargs_result.is_success:
                    validated_kwargs = kwargs_result.value
                else:
                    # If validation fails, use default values (CreateKwargsParams has defaults)
                    default_result = u.Model.from_kwargs(
                        m.Tests.Files.CreateKwargsParams,
                    )
                    if default_result.is_success:
                        validated_kwargs = default_result.value
                    else:
                        # This should never happen, but handle gracefully
                        raise ValueError(
                            f"Failed to create default kwargs: {default_result.error}",
                        )
                # Pass validated parameters explicitly to create() for type safety
                path = manager.create(
                    data,
                    filename,
                    directory=validated_kwargs.directory,
                    fmt=validated_kwargs.fmt,
                    enc=validated_kwargs.enc,
                    indent=validated_kwargs.indent,
                    delim=validated_kwargs.delim,
                    headers=validated_kwargs.headers,
                    readonly=validated_kwargs.readonly,
                    extract_result=extract_result,
                )
                paths[name] = path
            yield paths

    # =========================================================================
    # INSTANCE CONTEXT MANAGER
    # =========================================================================

    @contextmanager
    def temporary_directory(self) -> Generator[Path]:
        """Create and manage a temporary directory.

        Yields:
            Path to temporary directory that is automatically cleaned up.

        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    # =========================================================================
    # DIRECTORY UTILITIES (kept for permission testing)
    # =========================================================================

    def create_readonly_directory(
        self,
        name: str = c.Tests.Files.DEFAULT_READONLY_DIR_NAME,
        directory: Path | None = None,
    ) -> Path:
        """Create a read-only directory for testing permission scenarios.

        Args:
            name: Name for the read-only directory
            directory: Parent directory (uses base_dir or temp if None)

        Returns:
            Path to the created read-only directory

        """
        target_dir = self._resolve_directory(directory)
        read_only_dir = target_dir / name
        read_only_dir.mkdir(parents=True, exist_ok=True)
        read_only_dir.chmod(c.Tests.Files.PERMISSION_READONLY_DIR)
        self.created_dirs.append(read_only_dir)
        return read_only_dir

    def restore_directory_permissions(self, directory: Path) -> None:
        """Restore directory permissions to writable (0o755).

        Args:
            directory: Directory to restore permissions for

        """
        directory.chmod(c.Tests.Files.PERMISSION_WRITABLE_DIR)

    # =========================================================================
    # DEPRECATED METHODS (backward compatibility)
    # =========================================================================

    def create_text_file(
        self,
        content: str,
        filename: str = c.Tests.Files.DEFAULT_TEXT_FILENAME,
        directory: Path | None = None,
        encoding: str = c.Tests.Files.DEFAULT_ENCODING,
    ) -> Path:
        """Create text file. DEPRECATED: Use create() instead."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_CREATE_TEXT,
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(content, filename, directory, enc=encoding)

    def create_binary_file(
        self,
        content: bytes,
        filename: str = c.Tests.Files.DEFAULT_BINARY_FILENAME,
        directory: Path | None = None,
    ) -> Path:
        """Create binary file. DEPRECATED: Use create() instead."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_CREATE_BINARY,
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(content, filename, directory, fmt="bin")

    def create_empty_file(
        self,
        filename: str = c.Tests.Files.DEFAULT_EMPTY_FILENAME,
        directory: Path | None = None,
    ) -> Path:
        """Create empty file. DEPRECATED: Use create('', name) instead."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_CREATE_EMPTY,
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create("", filename, directory)

    def create_config_file(
        self,
        content: str | t.Types.ConfigurationMapping,
        filename: str = c.Tests.Files.DEFAULT_CONFIG_FILENAME,
        directory: Path | None = None,
    ) -> Path:
        """Create config file. DEPRECATED: Use create() instead."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_CREATE_CONFIG,
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(content, filename, directory)

    def create_file_set(
        self,
        files: t.Types.StringDict,
        directory: Path | None = None,
        extension: str = c.Tests.Files.DEFAULT_EXTENSION,
    ) -> t.Types.StringPathDict:
        """Create multiple files. DEPRECATED: Use files() context manager."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_CREATE_FILE_SET,
            DeprecationWarning,
            stacklevel=2,
        )
        target_dir = self._resolve_directory(directory)
        created: t.Types.StringPathDict = {}
        for name, content in files.items():
            filename = name if "." in name else f"{name}{extension}"
            path = self.create(content, filename, target_dir)
            created[name] = path
        return created

    def get_file_info(self, file_path: Path) -> m.Tests.Files.FileInfo:
        """Get file info. DEPRECATED: Use info() instead (returns r[FileInfo])."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_GET_FILE_INFO,
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.info(file_path)
        if result.is_success:
            return result.unwrap()
        return m.Tests.Files.FileInfo(exists=False)

    @classmethod
    @contextmanager
    def temporary_files(
        cls,
        files: t.Types.StringDict,
        extension: str = c.Tests.Files.DEFAULT_EXTENSION,
    ) -> Generator[t.Types.StringPathDict]:
        """Create temporary files. DEPRECATED: Use tf.files() instead."""
        warnings.warn(
            c.Tests.Files.DEPRECATION_TEMPORARY_FILES,
            DeprecationWarning,
            stacklevel=2,
        )
        # Convert StringDict to broader type for files() compatibility using Mapping
        # Use Mapping to avoid dict invariant type error
        content_mapping: ABCMapping[
            str,
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = files
        # Convert to dict for files() method which expects dict
        content_dict: dict[
            str,
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = dict(content_mapping)
        with cls.files(content_dict, ext=extension) as created:
            yield created

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _resolve_directory(self, directory: Path | None) -> Path:
        """Resolve target directory for file creation."""
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            return self.base_dir
        temp_dir = Path(tempfile.mkdtemp())
        self.created_dirs.append(temp_dir)
        return temp_dir

    def _extract_content(
        self,
        content: (
            str
            | bytes
            | t.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel
            | r[str]
            | r[bytes]
            | r[t.Types.ConfigurationMapping]
            | r[Sequence[Sequence[str]]]
            | r[BaseModel]
        ),
        extract_result: bool,
    ) -> (
        str | bytes | t.Types.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel
    ):
        """Extract actual content from FlextResult or return as-is.

        Uses u.is_type(content, "result") for type checking and u.val() for extraction.

        Args:
            content: Content that may be wrapped in FlextResult
            extract_result: Whether to extract from FlextResult

        Returns:
            Extracted content or original value

        Raises:
            ValueError: If FlextResult is failure and extraction is enabled

        """
        # Use u.is_type() for type-safe checking with proper type narrowing
        if extract_result and u.is_type(content, "result"):
            # Type narrowing: content is r[T] here
            content_result = cast("r[t.Tests.Files.FileContent]", content)
            if content_result.is_failure:
                error_msg = content_result.error or "FlextResult failure"
                raise ValueError(
                    f"Cannot create file from failed FlextResult: {error_msg}",
                )
            # Use unwrap() for extraction - type is narrowed
            # The return type matches FileContent union type
            # unwrapped is already FileContent type from r[FileContent]
            return content_result.unwrap()
        # Content is already FileContent type (not wrapped in Result)
        return cast("t.Tests.Files.FileContent", content)

    def _parse_content_metadata(
        self,
        path: Path,
        text: str,
        fmt: c.Tests.Files.FormatLiteral,
        validate_model: type[BaseModel] | None = None,
    ) -> m.Tests.Files.ContentMeta:
        """Parse file content and extract metadata.

        Uses u.Model.load() for model validation and format-specific parsing
        to extract content statistics (key_count, item_count, row_count, etc.).

        Args:
            path: File path
            text: File text content
            fmt: Detected file format
            validate_model: Pydantic model to validate content against

        Returns:
            ContentMeta with extracted statistics

        """
        key_count: int | None = None
        item_count: int | None = None
        row_count: int | None = None
        column_count: int | None = None
        model_valid: bool | None = None
        model_name: str | None = None

        # Parse based on format
        parsed_content: (
            t.Types.ConfigurationMapping | list[t.GeneralValueType] | None
        ) = None

        if fmt in {"json", "yaml"}:
            try:
                if fmt == "json":
                    parsed_raw = json.loads(text) if text.strip() else {}
                else:
                    # YAML parsing
                    parsed_raw = yaml.safe_load(text) if text.strip() else {}

                # Type narrowing and assignment
                if u.is_type(parsed_raw, "dict"):
                    parsed_content = cast("t.Types.ConfigurationMapping", parsed_raw)
                    key_count = len(parsed_content)
                elif u.is_type(parsed_raw, "sequence"):
                    parsed_content = cast("list[t.GeneralValueType]", parsed_raw)
                    item_count = len(parsed_content)
            except (json.JSONDecodeError, yaml.YAMLError):
                # Invalid content, leave metadata as None
                pass

        elif fmt == "csv":
            # Parse CSV for row/column counts
            try:
                rows = list(csv.reader(text.splitlines()))
                if rows:
                    row_count = len(rows)
                    column_count = len(rows[0]) if rows[0] else 0
            except csv.Error:
                # Invalid CSV, leave metadata as None
                pass

        # Model validation if requested
        if validate_model is not None:
            model_name = validate_model.__name__
            if (
                parsed_content is not None
                and u.is_type(parsed_content, "dict")
                and isinstance(parsed_content, Mapping)
            ):
                # Use u.Model.load() for Pydantic validation
                # Convert to dict if needed
                content_dict = (
                    parsed_content
                    if isinstance(parsed_content, dict)
                    else dict(parsed_content)
                )
                validation_result = u.Model.load(
                    validate_model,
                    cast("t.Types.ConfigurationMapping", content_dict),
                )
                model_valid = validation_result.is_success
            elif fmt in {"json", "yaml"} and text.strip():
                # Content exists but couldn't be parsed or isn't dict
                model_valid = False
            else:
                # No content to validate
                model_valid = None

        return m.Tests.Files.ContentMeta(
            key_count=key_count,
            item_count=item_count,
            row_count=row_count,
            column_count=column_count,
            model_valid=model_valid,
            model_name=model_name,
        )


# Short alias for convenient test usage
tf = FlextTestsFiles

__all__ = ["FlextTestsFiles", "tf"]
