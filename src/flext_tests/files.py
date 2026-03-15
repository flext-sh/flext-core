"""File management utilities for FLEXT ecosystem tests.

Provides comprehensive file operations for testing across the FLEXT ecosystem
with a simplified API using generalist methods with powerful optional parameters.

Supports:
- r: Automatically extracts value before serialization
- Pydantic models: Serializes to JSON/YAML via model_dump()
- Lists, dicts, Mappings: Proper JSON/YAML serialization
- Generic type loading: Load files directly into Pydantic models

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Literal, Self, TypeGuard, TypeVar, overload

from pydantic import BaseModel, TypeAdapter, ValidationError
from yaml import YAMLError, dump as yaml_dump, safe_load as yaml_safe_load

from flext_core import FlextModelsContainers, FlextResult, FlextRuntime, r
from flext_core.runtime import RuntimeData
from flext_tests import c, m, s, t, u

TModel = TypeVar("TModel", bound=BaseModel)
_FormatLiteral = Literal["auto", "text", "bin", "json", "yaml", "csv"]
_CompareModeLiteral = Literal["content", "size", "hash", "lines"]
_OperationLiteral = Literal["create", "read", "delete"]
_ErrorModeLiteral = Literal["stop", "skip", "collect"]
TestsFileContent = t.Tests.FileContent
_YAMLError = YAMLError
_OBJECT_LIST_ADAPTER = TypeAdapter(list[t.Tests.object])
_OBJECT_DICT_ADAPTER = TypeAdapter(dict[str, t.Tests.object])


_SCALAR_PATH: tuple[type, ...] = (str, int, float, bool, datetime, Path)


def _to_runtime_data(value: t.Tests.object) -> RuntimeData:
    """Narrow t.Tests.Testobject to RuntimeData for normalize_to_general_value calls.

    Converts bytes to str and ensures the value is RuntimeData-compatible.
    The key difference between t.Tests.Testobject and RuntimeData is that t.Tests.Testobject
    includes `bytes` which is not in RuntimeData.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, datetime, Path)):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(k): _to_normalized_or_model(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized_leaf(item) for item in value]
    return str(value)


def _to_normalized_or_model(value: t.Tests.object) -> t.NormalizedValue | BaseModel:
    """Convert t.Tests.Testobject to NormalizedValue | BaseModel for Mapping values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, datetime, Path)):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return FlextModelsContainers.ConfigMap(
            root={str(k): _to_normalized_or_model(v) for k, v in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return [_to_normalized_leaf(item) for item in value]
    return str(value)


def _to_normalized_leaf(value: t.Tests.object) -> t.NormalizedValue:
    """Convert t.Tests.Testobject to NormalizedValue (no BaseModel) for list contexts."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, datetime, Path)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseModel):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_normalized_leaf(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized_leaf(item) for item in value]
    return str(value)


def _yaml_safe_load(
    raw: str,
) -> dict[str, t.Tests.object] | list[t.Tests.object] | None:
    return yaml_safe_load(raw)


def _yaml_dump(value: dict[str, t.Tests.object], *, indent: int) -> str:
    return str(
        yaml_dump(value, default_flow_style=False, allow_unicode=True, indent=indent)
    )


def _is_batch_content(content_raw: object) -> TypeGuard[t.Tests.object]:
    try:
        _ = m.Tests.CreateParams.model_validate({
            "content": content_raw,
            "name": c.Tests.Files.DEFAULT_FILENAME,
        })
        return True
    except (TypeError, ValueError, AttributeError):
        return False


class FlextTestsFiles(s[t.Tests.TestResultValue]):
    """Manages test files for FLEXT ecosystem testing.

    Extends FlextTestsServiceBase for consistent service patterns.

    Provides generalist file operations with powerful optional parameters:
    - `create()`: Create any file type with auto-detection
    - `read()`: Read any file type with r
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

    FileInfo: ClassVar[type[m.Tests.FileInfo]] = m.Tests.FileInfo

    @staticmethod
    def _validate_model_content[TModelRead: BaseModel](
        model_cls: type[TModelRead],
        content: str | bytes | m.ConfigMap | list[list[str]],
    ) -> r[TModelRead]:
        try:
            model_instance: TModelRead = model_cls.model_validate(content)
            return r[TModelRead].ok(model_instance)
        except (TypeError, ValueError, AttributeError) as ex:
            return r[TModelRead].fail(f"Failed to validate model: {ex}")

    _base_dir: Path | None = None
    _created_files: list[Path] | None = None
    _created_dirs: list[Path] | None = None

    def __init__(self, base_dir: Path | None = None, **data: t.Tests.object) -> None:
        """Initialize file manager with optional base directory.

        Args:
            base_dir: Optional base directory for file operations.
                     If not provided, temporary directories are used.
            **data: Additional data passed to parent service.

        """
        super().__init__()
        self._base_dir = base_dir
        self._created_files = []
        self._created_dirs = []

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

    @property
    def base_dir(self) -> Path | None:
        """Get base directory."""
        return self._base_dir

    @property
    def created_dirs(self) -> list[Path]:
        """Get list of created directories."""
        return self._created_dirs or []

    @property
    def created_files(self) -> list[Path]:
        """Get list of created files."""
        return self._created_files or []

    @classmethod
    @contextmanager
    def files(
        cls,
        content: Mapping[
            str, str | bytes | m.ConfigMap | Sequence[Sequence[str]] | BaseModel
        ],
        *,
        directory: Path | None = None,
        ext: str | None = None,
        extract_result: bool = True,
        **kwargs: t.Tests.object,
    ) -> Generator[Mapping[str, Path]]:
        """Create temporary files with auto-cleanup.

        Supports Pydantic models, dicts, lists, and raw content.

        Args:
            content: Dict mapping names to content (str, bytes, dict, list, BaseModel)
            directory: Base directory (temp if None)
            ext: Default extension if not in name
            extract_result: Auto-extract r values (default: True)
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
            manager._base_dir = directory
        with manager:
            paths: dict[str, Path] = {}
            default_ext = ext or c.Tests.Files.DEFAULT_EXTENSION
            for name, data_raw in content.items():
                data: t.Tests.object = data_raw
                filename = name if "." in name else f"{name}{default_ext}"
                if "." not in name and (isinstance(data, (Mapping, BaseModel))):
                    filename = f"{name}.json"
                else:
                    is_nested_sequence = "." not in name and manager._is_nested_rows(
                        data
                    )
                    if is_nested_sequence:
                        filename = f"{name}.csv"
                try:
                    validated_kwargs = m.Tests.CreateKwargsParams.model_validate(kwargs)
                except (TypeError, ValueError, ValidationError):
                    validated_kwargs = m.Tests.CreateKwargsParams(
                        directory=None,
                        fmt="auto",
                        enc="utf-8",
                        indent=2,
                        delim=",",
                        headers=None,
                        readonly=False,
                    )
                path = manager.create(
                    manager._coerce_file_content(data),
                    filename,
                    directory=validated_kwargs.directory,
                    fmt=manager._normalize_create_format(validated_kwargs.fmt),
                    enc=validated_kwargs.enc,
                    indent=validated_kwargs.indent,
                    delim=validated_kwargs.delim,
                    headers=validated_kwargs.headers,
                    readonly=validated_kwargs.readonly,
                    extract_result=extract_result,
                )
                paths[name] = path
            yield paths

    @staticmethod
    def _is_mapping(value: t.Tests.object) -> TypeGuard[Mapping[str, t.Tests.object]]:
        return isinstance(value, Mapping)

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
            _ = tf.assert_exists(path)                    # Just exists
            _ = tf.assert_exists(path, is_file=True)      # Exists and is file
            _ = tf.assert_exists(path, not_empty=True)    # Exists and not empty
            _ = tf.assert_exists(path, is_dir=True, writable=True)  # Dir and writable

        """
        if not path.exists():
            error_msg = msg or c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=path)
            raise AssertionError(error_msg)
        if is_file is not None:
            if is_file and (not path.is_file()):
                raise AssertionError(msg or f"Path {path} is not a file")
            if not is_file and path.is_file():
                raise AssertionError(msg or f"Path {path} should not be a file")
        if is_dir is not None:
            if is_dir and (not path.is_dir()):
                raise AssertionError(msg or f"Path {path} is not a directory")
            if not is_dir and path.is_dir():
                raise AssertionError(msg or f"Path {path} should not be a directory")
        if not_empty is not None:
            if not_empty:
                if path.is_file() and path.stat().st_size == 0:
                    raise AssertionError(msg or f"File {path} is empty")
                if path.is_dir() and (not any(path.iterdir())):
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
            and (not os.access(path, os.R_OK))
        ):
            raise AssertionError(msg or f"File {path} is not readable")
        if writable is not None and writable:
            if path.is_file() and (not os.access(path, os.W_OK)):
                raise AssertionError(msg or f"File {path} is not writable")
            if path.is_dir() and (not os.access(path, os.W_OK)):
                raise AssertionError(msg or f"Directory {path} is not writable")
        return path

    @staticmethod
    def create_in(
        content: str
        | bytes
        | m.ConfigMap
        | Sequence[Sequence[str]]
        | BaseModel
        | r[str]
        | r[bytes]
        | r[m.ConfigMap]
        | r[Sequence[Sequence[str]]]
        | r[BaseModel],
        name: str,
        directory: Path,
        *,
        fmt: _FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        indent: int = c.Tests.Files.DEFAULT_JSON_INDENT,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        headers: list[str] | None = None,
        readonly: bool = False,
        extract_result: bool = True,
    ) -> Path:
        """Create file directly in directory - static convenience method.

        Supports r, Pydantic models, lists, dicts, and raw content.

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
            extract_result: Auto-extract r value (default: True)

        Returns:
            Path to created file.

        Examples:
            # Simple text file
            path = tf.create_in("content", "file.txt", output_dir)

            # Pydantic model
            path = tf.create_in(user_model, "user.json", output_dir)

            # r
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

    def batch[TModel: BaseModel](
        self,
        files: t.Tests.Files.BatchFiles,
        *,
        directory: Path | None = None,
        operation: _OperationLiteral = "create",
        model: type[TModel] | None = None,
        on_error: _ErrorModeLiteral = "collect",
        parallel: bool = False,
    ) -> r[m.Tests.BatchResult]:
        """Batch file operations.

        Uses u.batch() for batch processing with error handling.

        Args:
            files: Mapping[str, t.FileContent] or Sequence[tuple[str, t.FileContent]]
            directory: Target directory for create operations
            operation: "create", "read", or "delete"
            model: Optional model class for read operations
            on_error: Error handling mode ("stop", "skip", "collect")
            parallel: Run operations in parallel (not implemented yet)

        Returns:
            r[m.Tests.BatchResult] with results and errors

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
        try:
            params = m.Tests.BatchParams.model_validate({
                "files": files,
                "directory": directory,
                "operation": operation,
                "model": model,
                "on_error": on_error,
                "parallel": parallel,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            return r[m.Tests.BatchResult].fail(
                f"Invalid parameters for batch operation: {exc}"
            )
        files_dict: dict[str, t.Tests.object]
        if isinstance(params.files, Mapping):
            files_dict = {str(k): v for k, v in params.files.items()}
        elif not isinstance(params.files, str):
            files_dict = {}
            for item in params.files:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and _is_batch_content(item[1])
                ):
                    name = str(item[0])
                    files_dict[name] = item[1]
        else:
            return r[m.Tests.BatchResult].fail(
                f"Invalid BatchFiles type: {type(params.files)}"
            )
        error_mode_str = "collect" if params.on_error == "collect" else "fail"

        def process_one(
            name_and_content: tuple[str, t.Tests.object],
        ) -> Path | r[Path]:
            """Process single file operation."""
            name, content = name_and_content
            match params.operation:
                case "create":
                    try:
                        content_for_create = (
                            m.ConfigMap(
                                root={
                                    str(k): FlextRuntime.normalize_to_container(
                                        _to_runtime_data(self._to_config_map_value(v))
                                    )
                                    if v is None
                                    or type(v)
                                    in {str, int, float, bool, bytes, list, tuple, dict}
                                    or (
                                        hasattr(type(v), "__mro__")
                                        and BaseModel in type(v).__mro__
                                    )
                                    or (hasattr(v, "keys") and hasattr(v, "items"))
                                    or (
                                        hasattr(v, "__len__")
                                        and hasattr(v, "__getitem__")
                                        and (type(v) not in {str, bytes})
                                    )
                                    else str(v)
                                    for k, v in content.items()
                                }
                            )
                            if isinstance(content, Mapping)
                            else content
                        )
                        normalized_content = self._coerce_file_content(
                            content_for_create
                        )
                        return self.create(normalized_content, name, params.directory)
                    except (OSError, TypeError, ValueError, AttributeError) as e:
                        return r[Path].fail(f"Failed to create {name}: {e}")
                case "read":
                    path = (
                        Path(content)
                        if isinstance(content, (Path, str))
                        else Path(name)
                    )
                    read_result = self.read(path, model_cls=None)
                    if read_result.is_success:
                        return path
                    return r[Path].fail(read_result.error or f"Failed to read {name}")
                case "delete":
                    path = (
                        Path(content)
                        if isinstance(content, (Path, str))
                        else Path(name)
                    )
                    try:
                        Path(path).unlink(missing_ok=True)
                        return Path(path)
                    except OSError as e:
                        return r[Path].fail(f"Failed to delete {name}: {e}")
                case _:
                    return r[Path].fail(f"Unknown operation: {params.operation}")

        items_list: list[tuple[str, t.Tests.object]] = list(files_dict.items())
        results: list[Path | t.Tests.object] = []
        errors: list[tuple[int, str]] = []
        total = len(items_list)
        for index, item in enumerate(items_list):
            operation_result = process_one(item)
            if isinstance(operation_result, r):
                if operation_result.is_success:
                    success_value: Path | t.Tests.object = operation_result.value
                    results.append(success_value)
                    continue
                error_message = operation_result.error or "Unknown error"
                if error_mode_str == "fail":
                    return r[m.Tests.BatchResult].fail(
                        f"Batch operation failed: {error_message}"
                    )
                errors.append((index, str(error_message)))
                continue
            results.append(operation_result)
        results_dict: dict[str, r[Path | t.Tests.object]] = {}
        failed_dict: dict[str, str] = {}
        for idx, result in enumerate(results):
            if idx < len(items_list):
                name, _ = items_list[idx]
                if isinstance(result, Path):
                    results_dict[name] = r[Path | t.Tests.object].ok(result)
                else:
                    results_dict[name] = r[Path | t.Tests.object].ok(
                        self._to_payload_value(result)
                    )
        for idx, error_msg in errors:
            if idx < len(items_list):
                name, _ = items_list[idx]
                failed_dict[name] = error_msg
        succeeded_count = len(results_dict)
        failed_count = len(failed_dict)
        return r[m.Tests.BatchResult].ok(
            m.Tests.BatchResult(
                succeeded=succeeded_count,
                failed=failed_count,
                total=total,
                results=results_dict,
                errors=failed_dict,
            )
        )

    def cleanup(self) -> None:
        """Clean up all created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
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

    def compare(
        self,
        file1: Path,
        file2: Path,
        *,
        mode: _CompareModeLiteral = "content",
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
            r[bool] - True if match.

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
        try:
            params = m.Tests.CompareParams.model_validate({
                "file1": file1,
                "file2": file2,
                "mode": mode,
                "ignore_ws": ignore_ws,
                "ignore_case": ignore_case,
                "pattern": pattern,
                "deep": deep,
                "keys": keys,
                "exclude_keys": exclude_keys,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            return r[bool].fail(f"Invalid parameters for file comparison: {exc}")
        if not params.file1.exists():
            return r[bool].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.file1)
            )
        if not params.file2.exists():
            return r[bool].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.file2)
            )
        try:
            if params.pattern is not None:
                text1 = params.file1.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
                text2 = params.file2.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
                return r[bool].ok(params.pattern in text1 and params.pattern in text2)
            match params.mode:
                case "size":
                    return r[bool].ok(
                        params.file1.stat().st_size == params.file2.stat().st_size
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

    def create(
        self,
        content: str
        | bytes
        | m.ConfigMap
        | Sequence[Sequence[str]]
        | BaseModel
        | r[str]
        | r[bytes]
        | r[m.ConfigMap]
        | r[Sequence[Sequence[str]]]
        | r[BaseModel],
        name: str = c.Tests.Files.DEFAULT_FILENAME,
        directory: Path | None = None,
        *,
        fmt: _FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        indent: int = c.Tests.Files.DEFAULT_JSON_INDENT,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        headers: list[str] | None = None,
        readonly: bool = False,
        extract_result: bool = True,
    ) -> Path:
        r"""Create file with auto-detection or explicit format.

        Supports r, Pydantic models, lists, dicts, and raw content.

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
            extract_result: Auto-extract r value (default: True)

        Returns:
            Path to created file.

        Raises:
            ValueError: If r is failure and extract_result=True

        Examples:
            # Text file
            path = tf().create("hello", "test.txt")

            # JSON file (auto-detected from dict)
            path = tf().create({"key": "value"}, "config.json")

            # Pydantic model (auto-detected as JSON)
            path = tf().create(user_model, "user.json")

            # r with auto-extraction
            result = service.get_config()  # r[dict]
            path = tf().create(result, "config.json")

            # CSV file (auto-detected from list[list])
            path = tf().create([["a", "b"], ["1", "2"]], "data.csv",
                              headers=["col1", "col2"])

            # Binary file
            path = tf().create(b"\\x00\\x01", "data.bin", fmt="bin")

        """
        content_to_validate = self._extract_content(content, extract_result)
        try:
            params = m.Tests.CreateParams.model_validate({
                "content": content_to_validate,
                "name": name,
                "directory": directory,
                "fmt": fmt,
                "enc": enc,
                "indent": indent,
                "delim": delim,
                "headers": headers,
                "readonly": readonly,
                "extract_result": extract_result,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid parameters for file creation: {exc}") from None
        target_dir = self._resolve_directory(params.directory)
        name_str = str(params.name)
        file_path: Path = target_dir / name_str
        actual_content: (
            str
            | bytes
            | m.ConfigMap
            | Sequence[Sequence[str]]
            | BaseModel
            | Mapping[str, t.Tests.object]
        ) = self._coerce_file_content(params.content)
        if isinstance(actual_content, BaseModel):
            actual_content = self._mapping_to_payload(u.dump(actual_content))
        content_for_detect: str | bytes | Mapping[str, t.Tests.object] | list[list[str]]
        if isinstance(actual_content, str | bytes):
            content_for_detect = actual_content
        elif isinstance(actual_content, Mapping):
            content_for_detect = self._mapping_to_payload(actual_content)
        elif isinstance(actual_content, list):
            if self._is_nested_rows(actual_content):
                content_for_detect = [
                    [str(cell) for cell in row]
                    for row in actual_content
                    if isinstance(row, (list, tuple))
                ]
            else:
                content_for_detect = str(actual_content)
        elif isinstance(actual_content, tuple):
            content_for_detect = str(actual_content)
        else:
            content_for_detect = str(actual_content)
        actual_fmt = u.Tests.Files.detect_format(
            content_for_detect, params.name, params.fmt
        )
        if actual_fmt == c.Tests.Files.Format.BIN:
            if isinstance(actual_content, bytes):
                _ = file_path.write_bytes(actual_content)
            else:
                _ = file_path.write_bytes(str(actual_content).encode(params.enc))
        elif actual_fmt == c.Tests.Files.Format.JSON:
            if isinstance(actual_content, Mapping):
                data: dict[str, t.Tests.object] = {
                    str(key): value for key, value in actual_content.items()
                }
            else:
                empty_data: dict[str, t.Tests.object] = {}
                data = {"value": actual_content} if actual_content else empty_data
            json_str = (
                TypeAdapter(dict[str, t.Tests.object])
                .dump_json(data, indent=params.indent)
                .decode()
            )
            _ = file_path.write_text(json_str, encoding=params.enc)
        elif actual_fmt == c.Tests.Files.Format.YAML:
            if isinstance(actual_content, Mapping):
                data_yaml: dict[str, t.Tests.object] = {
                    str(key): value for key, value in actual_content.items()
                }
            else:
                empty_data_y: dict[str, t.Tests.object] = {}
                data_yaml = (
                    {"value": actual_content} if actual_content else empty_data_y
                )
            yaml_result = _yaml_dump(data_yaml, indent=params.indent)
            _ = file_path.write_text(yaml_result, encoding=params.enc)
        elif actual_fmt == c.Tests.Files.Format.CSV:
            csv_content: list[list[str]]
            if isinstance(actual_content, Sequence) and (
                not isinstance(actual_content, str | bytes)
            ):
                if self._is_nested_rows(actual_content):
                    csv_content = [
                        [str(cell) for cell in row]
                        for row in actual_content
                        if not isinstance(row, str | bytes)
                    ]
                else:
                    csv_content = [[str(item)] for item in actual_content]
            else:
                csv_content = [[str(actual_content)]]
            u.Tests.Files.write_csv(
                file_path, csv_content, params.headers, params.delim, params.enc
            )
        else:
            _ = file_path.write_text(str(actual_content), encoding=params.enc)
        if params.readonly:
            file_path.chmod(c.Tests.Files.PERMISSION_READONLY_FILE)
        if self._created_files is None:
            self._created_files = []
        self._created_files.append(file_path)
        return file_path

    def execute(self) -> r[t.Tests.TestResultValue]:
        """Execute service - returns success for file manager.

        FlextTestsFiles is a utility service that doesn't have a specific
        execution result. Returns success by default.
        """
        return r[t.Tests.TestResultValue].ok(None)

    def info(
        self,
        path: Path,
        *,
        compute_hash: bool = False,
        detect_fmt: bool = True,
        parse_content: bool = False,
        validate_model: type[BaseModel] | None = None,
    ) -> r[m.Tests.FileInfo]:
        """Get comprehensive file information.

        Args:
            path: File path
            compute_hash: Compute SHA256 (default: False)
            detect_fmt: Auto-detect format (default: True)
            parse_content: Parse content and include metadata (default: False)
            validate_model: Pydantic model to validate content against (default: None)

        Returns:
            r[FileInfo] with info or error.

        Examples:
            result = tf().info(path)
            if result.is_success:
                info = result.value
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
        try:
            params = m.Tests.InfoParams.model_validate({
                "path": path,
                "compute_hash": compute_hash,
                "detect_fmt": detect_fmt,
                "parse_content": parse_content,
                "validate_model": validate_model,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            return r[m.Tests.FileInfo].fail(f"Invalid parameters for file info: {exc}")
        if not params.path.exists():
            return r[m.Tests.FileInfo].ok(
                m.Tests.FileInfo(exists=False, path=params.path)
            )
        try:
            stat = params.path.stat()
            size = stat.st_size
            size_human = u.Tests.Files.format_size(size)
            try:
                text = params.path.read_text(
                    encoding=c.Tests.Files.DEFAULT_ENCODING, errors="replace"
                )
                lines = text.count("\n") + 1 if text else 0
                is_empty = len(text.strip()) == 0
                first_line = text.split("\n")[0] if text else ""
                encoding = c.Tests.Files.DEFAULT_ENCODING
            except UnicodeDecodeError:
                text = ""
                lines = 0
                is_empty = size == 0
                first_line = ""
                encoding = c.Tests.Files.DEFAULT_BINARY_ENCODING
            fmt: str = "unknown"
            if params.detect_fmt:
                detected = u.Tests.Files.detect_format_from_path(params.path, "auto")
                match detected:
                    case "auto":
                        fmt = "auto"
                    case "text":
                        fmt = "text"
                    case "bin":
                        fmt = "bin"
                    case "json":
                        fmt = "json"
                    case "yaml":
                        fmt = "yaml"
                    case "csv":
                        fmt = "csv"
                    case _:
                        fmt = "unknown"
            permissions = stat.st_mode
            is_readonly = not permissions & 128
            sha256 = (
                u.Tests.Files.compute_hash(params.path) if params.compute_hash else None
            )
            content_meta: m.Tests.ContentMeta | None = None
            if params.parse_content or params.validate_model:
                content_meta = self._parse_content_metadata(
                    path=params.path,
                    text=text,
                    fmt=fmt,
                    validate_model=params.validate_model,
                )
            return r[m.Tests.FileInfo].ok(
                m.Tests.FileInfo(
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
                )
            )
        except OSError as e:
            return r[m.Tests.FileInfo].fail(c.Tests.Files.ERROR_INFO.format(error=e))

    @overload
    def read(
        self,
        path: Path,
        *,
        model_cls: None = None,
        fmt: _FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[str | bytes | m.ConfigMap | list[list[str]]]: ...

    @overload
    def read(
        self,
        path: Path,
        *,
        model_cls: type[TModel],
        fmt: _FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[TModel]: ...

    def read(
        self,
        path: Path,
        *,
        model_cls: type[TModel] | None = None,
        fmt: _FormatLiteral = "auto",
        enc: str = c.Tests.Files.DEFAULT_ENCODING,
        delim: str = c.Tests.Files.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> r[str | bytes | m.ConfigMap | list[list[str]]] | r[TModel]:
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
            r with content or model instance.

        Examples:
            # Read text
            result = tf().read(path)
            if result.is_success:
                text = result.value

            # Read JSON
            result = tf().read(Path("config.json"))
            data = result.value  # dict

            # Read JSON into Pydantic model
            result = tf().read(Path("user.json"), model_cls=UserModel)
            user = result.value  # UserModel instance

            # Read YAML into Pydantic model
            result = tf().read(Path("config.yaml"), model_cls=ConfigModel)
            config = result.value  # ConfigModel instance

            # Read CSV
            result = tf().read(Path("data.csv"))
            rows = result.value  # list[list[str]]

        """
        try:
            params = m.Tests.ReadParams.model_validate({
                "path": path,
                "fmt": fmt,
                "enc": enc,
                "delim": delim,
                "has_headers": has_headers,
                "model_cls": model_cls,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            error_msg = f"Invalid parameters for file read: {exc}"
            if model_cls is not None:
                invalid_params_result: r[TModel] = r[TModel].fail(error_msg)
                return invalid_params_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(error_msg)
        if not params.path.exists():
            if model_cls is not None:
                file_not_found_result: r[TModel] = r[TModel].fail(
                    c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.path)
                )
                return file_not_found_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(
                c.Tests.Files.ERROR_FILE_NOT_FOUND.format(path=params.path)
            )
        actual_fmt = u.Tests.Files.detect_format_from_path(params.path, params.fmt)
        try:
            if actual_fmt == c.Tests.Files.Format.BIN:
                content: str | bytes | m.ConfigMap | list[list[str]] = (
                    params.path.read_bytes()
                )
            elif actual_fmt == c.Tests.Files.Format.JSON:
                text = params.path.read_text(encoding=params.enc)
                parsed_json = TypeAdapter(dict[str, t.Tests.object]).validate_json(
                    text.encode()
                )
                content = self._coerce_read_content(parsed_json)
            elif actual_fmt == c.Tests.Files.Format.YAML:
                text = params.path.read_text(encoding=params.enc)
                parsed_yaml = _yaml_safe_load(text)
                content = self._coerce_read_content(
                    parsed_yaml if isinstance(parsed_yaml, dict) else None
                )
            elif actual_fmt == c.Tests.Files.Format.CSV:
                content = u.Tests.Files.read_csv(
                    params.path,
                    params.delim,
                    params.enc,
                    has_headers=params.has_headers,
                )
            else:
                content = params.path.read_text(encoding=params.enc)
            if model_cls is not None:
                return self._validate_model_content(model_cls, content)
            return r[str | bytes | m.ConfigMap | list[list[str]]].ok(content)
        except UnicodeDecodeError as e:
            if model_cls is not None:
                invalid_encoding_result: r[TModel] = r[TModel].fail(
                    c.Tests.Files.ERROR_ENCODING.format(error=e)
                )
                return invalid_encoding_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(
                c.Tests.Files.ERROR_ENCODING.format(error=e)
            )
        except ValueError as e:
            if model_cls is not None:
                invalid_json_result: r[TModel] = r[TModel].fail(
                    c.Tests.Files.ERROR_INVALID_JSON.format(error=e)
                )
                return invalid_json_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(
                c.Tests.Files.ERROR_INVALID_JSON.format(error=e)
            )
        except _YAMLError as e:
            if model_cls is not None:
                invalid_yaml_result: r[TModel] = r[TModel].fail(
                    c.Tests.Files.ERROR_INVALID_YAML.format(error=e)
                )
                return invalid_yaml_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(
                c.Tests.Files.ERROR_INVALID_YAML.format(error=e)
            )
        except OSError as e:
            if model_cls is not None:
                file_read_error_result: r[TModel] = r[TModel].fail(
                    c.Tests.Files.ERROR_READ.format(error=e)
                )
                return file_read_error_result
            return r[str | bytes | m.ConfigMap | list[list[str]]].fail(
                c.Tests.Files.ERROR_READ.format(error=e)
            )

    @contextmanager
    def temporary_directory(self) -> Generator[Path]:
        """Create and manage a temporary directory.

        Yields:
            Path to temporary directory that is automatically cleaned up.

        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def _apply_key_filtering(
        self,
        dict1: Mapping[str, t.Tests.object],
        dict2: Mapping[str, t.Tests.object],
        keys: list[str] | None,
        exclude_keys: list[str] | None,
    ) -> tuple[Mapping[str, t.Tests.object], Mapping[str, t.Tests.object]]:
        """Apply key filtering to both dicts if specified."""
        if keys is None and exclude_keys is None:
            return (dict1, dict2)
        filter_keys_set = set(keys) if keys is not None else None
        exclude_keys_set = set(exclude_keys) if exclude_keys is not None else None
        config_root1: dict[str, t.NormalizedValue | BaseModel] = {
            str(k): FlextRuntime.normalize_to_container(
                _to_runtime_data(self._to_config_map_value(self._to_payload_value(v)))
            )
            for k, v in dict1.items()
        }
        config_root2: dict[str, t.NormalizedValue | BaseModel] = {
            str(k): FlextRuntime.normalize_to_container(
                _to_runtime_data(self._to_config_map_value(self._to_payload_value(v)))
            )
            for k, v in dict2.items()
        }
        result1 = u.transform(
            m.ConfigMap(root=config_root1),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        result2 = u.transform(
            m.ConfigMap(root=config_root2),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        if result1.is_success and result2.is_success:
            filtered1: Mapping[str, t.Tests.object] = {
                str(k): self._to_payload_value(v) for k, v in result1.value.items()
            }
            filtered2: Mapping[str, t.Tests.object] = {
                str(k): self._to_payload_value(v) for k, v in result2.value.items()
            }
            return (filtered1, filtered2)
        return (dict1, dict2)

    def _coerce_file_content(
        self,
        value: BaseModel
        | FlextModelsContainers.ConfigMap
        | FlextResult[BaseModel]
        | FlextResult[FlextModelsContainers.ConfigMap]
        | FlextResult[Sequence[Sequence[str]]]
        | FlextResult[bytes]
        | FlextResult[str]
        | Mapping[str, t.Tests.Testobject]
        | Path
        | Sequence[Sequence[str]]
        | Sequence[t.Tests.Testobject]
        | bool
        | bytes
        | datetime
        | float
        | str
        | None,
    ) -> str | bytes | m.ConfigMap | Sequence[Sequence[str]] | BaseModel:
        if isinstance(value, str | bytes):
            return value
        if isinstance(value, BaseModel):
            return value
        if self._is_mapping(value):
            coerce_root: dict[str, t.NormalizedValue | BaseModel] = {
                str(key): FlextRuntime.normalize_to_container(
                    _to_runtime_data(
                        self._to_config_map_value(self._to_payload_value(item))
                    )
                )
                for key, item in value.items()
            }
            return m.ConfigMap(root=coerce_root)
        if self._is_nested_rows(value):
            rows: list[list[str]] = []
            sequence_value: Sequence[t.Tests.object] = (
                value if isinstance(value, (list, tuple)) else ()
            )
            rows.extend(
                [str(cell) for cell in row]
                for row in sequence_value
                if isinstance(row, (list, tuple))
            )
            return rows
        return str(value)

    def _coerce_read_content(
        self, value: dict[str, t.Tests.object] | None
    ) -> str | bytes | m.ConfigMap | list[list[str]]:
        if isinstance(value, str | bytes):
            return value
        if self._is_mapping(value):
            read_root: dict[str, t.NormalizedValue | BaseModel] = {
                str(key): FlextRuntime.normalize_to_container(
                    _to_runtime_data(
                        self._to_config_map_value(self._to_payload_value(item))
                    )
                )
                for key, item in value.items()
            }
            return m.ConfigMap(root=read_root)
        if self._is_nested_rows(value):
            sequence_value: Sequence[t.Tests.object] = (
                value if isinstance(value, (list, tuple)) else ()
            )
            return [
                [str(cell) for cell in row]
                for row in sequence_value
                if isinstance(row, (list, tuple))
            ]
        return str(value)

    def _compare_content(self, params: m.Tests.CompareParams) -> r[bool]:
        """Compare file content with optional deep/structured comparison."""
        content1_raw = params.file1.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
        content2_raw = params.file2.read_text(encoding=c.Tests.Files.DEFAULT_ENCODING)
        if params.deep:
            deep_result = self._try_deep_compare(
                content1_raw, content2_raw, params.keys, params.exclude_keys
            )
            if deep_result is not None:
                return deep_result
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

    def _compare_lines(self, params: m.Tests.CompareParams) -> r[bool]:
        """Compare files line by line with optional normalization."""
        lines1 = params.file1.read_text(
            encoding=c.Tests.Files.DEFAULT_ENCODING
        ).splitlines()
        lines2 = params.file2.read_text(
            encoding=c.Tests.Files.DEFAULT_ENCODING
        ).splitlines()
        if params.ignore_ws:
            lines1 = [line.strip() for line in lines1]
            lines2 = [line.strip() for line in lines2]
        if params.ignore_case:
            lines1 = [line.lower() for line in lines1]
            lines2 = [line.lower() for line in lines2]
        return r[bool].ok(lines1 == lines2)

    def _extract_content(
        self,
        content: str
        | bytes
        | m.ConfigMap
        | Sequence[Sequence[str]]
        | BaseModel
        | r[str]
        | r[bytes]
        | r[m.ConfigMap]
        | r[Sequence[Sequence[str]]]
        | r[BaseModel],
        extract_result: bool,
    ) -> str | bytes | m.ConfigMap | Sequence[Sequence[str]] | BaseModel:
        """Extract actual content from r or return as-is.

        Uses u.is_type(content, "result") for type checking and u.val() for extraction.

        Args:
            content: Content that may be wrapped in r
            extract_result: Whether to extract from r

        Returns:
            Extracted content or original value

        Raises:
            ValueError: If r is failure and extraction is enabled

        """
        if extract_result and u.is_result_like(content):
            if content.is_failure:
                error_msg = content.error or "r failure"
                raise ValueError(f"Cannot create file from failed r: {error_msg}")
            resolved_value: t.Tests.object = getattr(content, "value", None)
            return self._coerce_file_content(resolved_value)
        return self._coerce_file_content(content)

    def _is_nested_rows(
        self, value: t.Tests.object
    ) -> TypeGuard[Sequence[Sequence[str]]]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return False
        try:
            sequence_value = _OBJECT_LIST_ADAPTER.validate_python(value)
        except ValidationError:
            return False
        if len(sequence_value) == 0:
            return False
        for row_raw in sequence_value:
            if not isinstance(row_raw, Sequence) or isinstance(row_raw, str | bytes):
                return False
        return True

    def _mapping_to_payload(
        self, mapping: Mapping[str, object]
    ) -> Mapping[str, t.Tests.object]:
        payload: dict[str, t.Tests.object] = {}
        for key, value in mapping.items():
            payload[str(key)] = self._to_payload_value(value)
        return payload

    def _normalize_create_format(self, fmt: str) -> _FormatLiteral:
        if fmt in {"txt", "md"}:
            return "text"
        match fmt:
            case "auto":
                return "auto"
            case "text":
                return "text"
            case "bin":
                return "bin"
            case "json":
                return "json"
            case "yaml":
                return "yaml"
            case "csv":
                return "csv"
            case _:
                return "auto"

    def _parse_content_metadata(
        self,
        path: Path,
        text: str,
        fmt: str,
        validate_model: type[BaseModel] | None = None,
    ) -> m.Tests.ContentMeta:
        """Parse file content and extract metadata.

        Uses u.load() for model validation and format-specific parsing
        to extract content statistics (key_count, item_count, row_count, etc.).

        Args:
            path: File path
            text: File text content
            fmt: Detected file format
            validate_model: Pydantic model to validate content against

        Returns:
            ContentMeta with extracted statistics

        """
        _ = path
        key_count: int | None = None
        item_count: int | None = None
        row_count: int | None = None
        column_count: int | None = None
        model_valid: bool | None = None
        model_name: str | None = None
        parsed_content: m.ConfigMap | list[t.Tests.object] | None = None
        if fmt in {"json", "yaml"}:
            try:
                if fmt == "json":
                    if text.strip():
                        try:
                            parsed_raw: (
                                dict[str, t.Tests.object] | list[t.Tests.object] | None
                            ) = TypeAdapter(dict[str, t.Tests.object]).validate_json(
                                text.encode()
                            )
                        except ValidationError:
                            parsed_raw = TypeAdapter(
                                list[t.Tests.object]
                            ).validate_json(text.encode())
                    else:
                        parsed_raw = dict(m.ConfigMap(root={}).root)
                else:
                    parsed_raw = (
                        _yaml_safe_load(text)
                        if text.strip()
                        else dict(m.ConfigMap(root={}).root)
                    )
                if self._is_mapping(parsed_raw):
                    parse_root: dict[str, t.NormalizedValue | BaseModel] = {
                        str(key): FlextRuntime.normalize_to_container(
                            _to_runtime_data(
                                self._to_config_map_value(self._to_payload_value(v))
                            )
                        )
                        for key, v in parsed_raw.items()
                    }
                    parsed_content = m.ConfigMap(root=parse_root)
                    key_count = len(parsed_content.root)
                elif isinstance(parsed_raw, list):
                    parsed_list = _OBJECT_LIST_ADAPTER.validate_python(parsed_raw)
                    parsed_content = [
                        self._to_payload_value(item) for item in parsed_list
                    ]
                    item_count = len(parsed_content)
            except (ValueError, _YAMLError):
                pass
        elif fmt == "csv":
            try:
                rows = list(csv.reader(text.splitlines()))
                if rows:
                    row_count = len(rows)
                    column_count = len(rows[0]) if rows[0] else 0
            except csv.Error:
                pass
        if validate_model is not None:
            model_name = validate_model.__name__
            if isinstance(parsed_content, m.ConfigMap):
                try:
                    _ = validate_model.model_validate(parsed_content.root)
                    model_valid = True
                except (TypeError, ValueError, AttributeError):
                    model_valid = False
            elif fmt in {"json", "yaml"} and text.strip():
                model_valid = False
            else:
                model_valid = None
        return m.Tests.ContentMeta(
            key_count=key_count,
            item_count=item_count,
            row_count=row_count,
            column_count=column_count,
            model_valid=model_valid,
            model_name=model_name,
        )

    def _resolve_directory(self, directory: Path | None) -> Path:
        """Resolve target directory for file creation."""
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            return self.base_dir
        temp_dir = Path(tempfile.mkdtemp())
        if self._created_dirs is None:
            self._created_dirs = []
        self._created_dirs.append(temp_dir)
        return temp_dir

    def _to_config_map_value(self, value: t.Tests.object) -> t.Tests.object:
        if value is None or isinstance(value, (*t.PRIMITIVES_TYPES, BaseModel, Path)):
            return value
        if isinstance(value, bytes):
            return value.decode(c.Tests.Files.DEFAULT_ENCODING, errors="replace")
        if isinstance(value, Mapping):
            return {
                str(k): self._to_config_map_value(self._to_payload_value(v))
                for k, v in value.items()
            }
        if isinstance(value, Sequence) and (not isinstance(value, str | bytes)):
            try:
                sequence_value = _OBJECT_LIST_ADAPTER.validate_python(value)
            except ValidationError:
                empty_sequence: list[t.Tests.object] = []
                return empty_sequence
            return [
                self._to_config_map_value(self._to_payload_value(item))
                for item in sequence_value
            ]
        return str(value)

    def _to_payload_value(self, value: object) -> t.Tests.object:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if isinstance(value, bool):
            return value
        if isinstance(value, bytes):
            return value
        if isinstance(value, BaseModel):
            return value
        if isinstance(value, Path | datetime):
            return str(value)
        if isinstance(value, Mapping):
            mapping_obj: Mapping[str, t.Tests.object] = (
                _OBJECT_DICT_ADAPTER.validate_python(value)
            )
            return self._mapping_to_payload(mapping_obj)
        if isinstance(value, Sequence) and (not isinstance(value, str | bytes)):
            try:
                sequence_value = _OBJECT_LIST_ADAPTER.validate_python(value)
            except ValidationError:
                return []
            payload_items: list[t.Tests.object] = [
                self._to_payload_value(item_raw) for item_raw in sequence_value
            ]
            return payload_items
        return str(value)

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
        parsed = self._try_parse_both(content1_raw, content2_raw, "json")
        if parsed is None:
            parsed = self._try_parse_both(content1_raw, content2_raw, "yaml")
        if parsed is None:
            return None
        dict1, dict2 = parsed
        filter_keys_set = set(keys) if keys is not None else None
        exclude_keys_set = set(exclude_keys) if exclude_keys is not None else None
        left_root: dict[str, t.NormalizedValue | BaseModel] = {
            str(k): FlextRuntime.normalize_to_container(
                _to_runtime_data(self._to_config_map_value(self._to_payload_value(v)))
            )
            for k, v in dict1.items()
        }
        right_root: dict[str, t.NormalizedValue | BaseModel] = {
            str(k): FlextRuntime.normalize_to_container(
                _to_runtime_data(self._to_config_map_value(self._to_payload_value(v)))
            )
            for k, v in dict2.items()
        }
        left_result = u.transform(
            m.ConfigMap(root=left_root),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        right_result = u.transform(
            m.ConfigMap(root=right_root),
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
        )
        if left_result.is_failure or right_result.is_failure:
            return r[bool].ok(False)
        return r[bool].ok(u.deep_eq(left_result.value, right_result.value))

    def _try_parse_both(
        self, content1: str, content2: str, fmt: str
    ) -> tuple[Mapping[str, t.Tests.object], Mapping[str, t.Tests.object]] | None:
        """Try to parse both contents as dicts in given format."""
        try:
            match fmt:
                case "json":
                    adapter = TypeAdapter(dict[str, t.Tests.object])
                    dict1_raw = adapter.validate_json(content1.encode())
                    dict2_raw = adapter.validate_json(content2.encode())
                case "yaml":
                    dict1_raw = _yaml_safe_load(content1)
                    dict2_raw = _yaml_safe_load(content2)
                case _:
                    return None
            if self._is_mapping(dict1_raw) and self._is_mapping(dict2_raw):
                dict1 = {
                    str(key): self._to_config_map_value(self._to_payload_value(value))
                    for key, value in dict1_raw.items()
                }
                dict2 = {
                    str(key): self._to_config_map_value(self._to_payload_value(value))
                    for key, value in dict2_raw.items()
                }
                return (dict1, dict2)
        except (ValueError, _YAMLError, TypeError):
            pass
        return None


tf = FlextTestsFiles
__all__ = ["FlextTestsFiles", "tf"]
