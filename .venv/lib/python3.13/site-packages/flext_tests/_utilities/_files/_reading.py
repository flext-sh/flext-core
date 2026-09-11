"""File-reading helpers for FlextTestsFiles.

Read content by format and optional Pydantic model validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import overload

from flext_cli import u
from flext_tests import c, m, p, r, t
from flext_tests._utilities._files._creation import FlextTestsFilesCreationMixin
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsFilesReadingMixin(FlextTestsFilesCreationMixin):
    """Read test files with format detection and model loading."""

    @staticmethod
    def _validate_model_content[TModelRead: m.BaseModel](
        model_cls: type[TModelRead], content: t.Tests.FileContentPlain
    ) -> p.Result[TModelRead]:
        try:
            model_instance: TModelRead = model_cls.model_validate(content)
            return r[TModelRead].ok(model_instance)
        except c.EXC_BASIC_TYPE as ex:
            return r[TModelRead].fail(f"Failed to validate model: {ex}")

    @staticmethod
    def _read_fail[TModelRead: m.BaseModel](
        error: str, model_cls: type[TModelRead] | None
    ) -> p.Result[t.Tests.ReadContent] | p.Result[TModelRead]:
        """Dispatch a single read-failure message to the correct result type."""
        if model_cls is not None:
            return r[TModelRead].fail(error)
        return r[t.Tests.ReadContent].fail(error)

    @overload
    def read(
        self,
        path: Path,
        *,
        model_cls: None = None,
        fmt: c.Tests.FileFormat = c.Tests.FILE_FORMAT_AUTO,
        enc: str = c.Tests.DEFAULT_ENCODING,
        delim: str = c.Tests.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> p.Result[t.Tests.ReadContent]: ...

    @overload
    def read[TModelRead: m.BaseModel](
        self,
        path: Path,
        *,
        model_cls: type[TModelRead],
        fmt: c.Tests.FileFormat = c.Tests.FILE_FORMAT_AUTO,
        enc: str = c.Tests.DEFAULT_ENCODING,
        delim: str = c.Tests.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> p.Result[TModelRead]: ...

    def read[TModelRead: m.BaseModel](
        self,
        path: Path,
        *,
        model_cls: type[TModelRead] | None = None,
        fmt: c.Tests.FileFormat = c.Tests.FILE_FORMAT_AUTO,
        enc: str = c.Tests.DEFAULT_ENCODING,
        delim: str = c.Tests.DEFAULT_CSV_DELIMITER,
        has_headers: bool = True,
    ) -> p.Result[t.Tests.ReadContent] | p.Result[TModelRead]:
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

        """
        result: p.Result[t.Tests.ReadContent] | p.Result[TModelRead]
        try:
            params = m.Tests.ReadParams.model_validate({
                "path": path,
                "fmt": fmt,
                "enc": enc,
                "delim": delim,
                "has_headers": has_headers,
                "model_cls": model_cls,
            })
        except c.EXC_BASIC_TYPE as exc:
            result = self._read_fail(
                f"Invalid parameters for file read: {exc}", model_cls
            )
        else:
            if not params.path.exists():
                result = self._read_fail(
                    c.Tests.ERROR_FILE_NOT_FOUND.format(path=params.path), model_cls
                )
            else:
                actual_fmt = u.Cli.files_detect_format_from_path(
                    params.path, params.fmt
                )
                try:
                    content = self._read_content_by_format(
                        params.path, actual_fmt, params
                    )
                except UnicodeDecodeError as e:
                    result = self._read_fail(
                        c.Tests.ERROR_ENCODING.format(error=e), model_cls
                    )
                except ValueError as e:
                    result = self._read_fail(
                        c.Tests.ERROR_INVALID_JSON.format(error=e), model_cls
                    )
                except c.Cli.YamlParseError as e:
                    result = self._read_fail(
                        c.Tests.ERROR_INVALID_YAML.format(error=e), model_cls
                    )
                except OSError as e:
                    result = self._read_fail(
                        c.Tests.ERROR_READ.format(error=e), model_cls
                    )
                else:
                    if model_cls is not None:
                        result = self._validate_model_content(model_cls, content)
                    else:
                        result = r[t.Tests.ReadContent].ok(content)
        return result

    def _read_content_by_format(
        self, path: Path, actual_fmt: str, params: m.Tests.ReadParams
    ) -> t.Tests.ReadContent:
        """Read file content using format-specific parsing."""
        content: t.Tests.ReadContent
        match actual_fmt:
            case _ if actual_fmt == c.Tests.FILE_FORMAT_BIN:
                content = path.read_bytes()
            case _ if actual_fmt == c.Tests.FILE_FORMAT_JSON:
                text = path.read_text(encoding=params.enc)
                parsed_json = t.Tests.TESTOBJECT_MAPPING_ADAPTER.validate_json(
                    text.encode()
                )
                content = (
                    FlextTestsPayloadUtilities.to_config_map(parsed_json)
                    if FlextTestsFilesCreationMixin.is_mapping(parsed_json)
                    else text
                )
            case _ if actual_fmt == c.Tests.FILE_FORMAT_YAML:
                text = path.read_text(encoding=params.enc)
                parsed_yaml_result = u.Cli.yaml_parse(text)
                parsed_yaml = (
                    parsed_yaml_result.value if parsed_yaml_result.success else None
                )
                content = (
                    FlextTestsPayloadUtilities.to_config_map(parsed_yaml)
                    if isinstance(parsed_yaml, dict)
                    else text
                )
            case _ if actual_fmt == c.Tests.FILE_FORMAT_CSV:
                csv_result = u.Cli.files_read_csv_with_headers(path)
                csv_rows_m = csv_result.value if csv_result.success else ()
                if csv_rows_m:
                    keys = list(csv_rows_m[0].keys())
                    data_rows = [[row.get(k, "") for k in keys] for row in csv_rows_m]
                    content = data_rows if params.has_headers else [keys, *data_rows]
                else:
                    content = []
            case _:
                content = path.read_text(encoding=params.enc)
        return content


__all__: list[str] = ["FlextTestsFilesReadingMixin"]
