"""File creation utilities for flext-tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeIs, cast

from flext_tests import c, m, p, t, u
from flext_tests._utilities._files._lifecycle import FlextTestsFilesLifecycleMixin
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsFilesCreationMixin(FlextTestsFilesLifecycleMixin):
    """Create test files and directory fixtures."""

    @staticmethod
    def _is_file_result(
        value: t.Tests.FileContentPlain | p.Result[t.Tests.FileContentPlain],
    ) -> TypeIs[p.Result[t.Tests.FileContentPlain]]:
        """Narrow file input to a result-like wrapper."""
        return isinstance(value, p.Result)

    @staticmethod
    def is_mapping(
        value: t.Tests.FileContentPlain | t.Tests.TestobjectSerializable | None,
    ) -> TypeIs[Mapping[str, t.Tests.TestobjectSerializable]]:
        return isinstance(value, Mapping)

    @staticmethod
    def to_payload_mapping(
        value: t.MappingKV[str, t.Tests.TestobjectSerializable],
    ) -> t.MappingKV[str, t.Tests.TestobjectSerializable]:
        return {
            key: FlextTestsPayloadUtilities.to_payload(item)
            for key, item in value.items()
        }

    @staticmethod
    def _to_string_rows(
        value: t.SequenceOf[t.SequenceOf[t.Tests.TestobjectSerializable]],
    ) -> t.SequenceOf[t.StrSequence]:
        """Convert rows already validated as nested sequences to strings."""
        return [[str(cell) for cell in row] for row in value]

    def _coerce_file_content(
        self,
        value: t.Tests.FileContentPlain
        | p.Result[t.Tests.FileContentPlain]
        | t.Tests.TestobjectSerializable,
    ) -> t.Tests.FileContentPlain:
        unwrapped: t.Tests.FileContentPlain | t.Tests.TestobjectSerializable = (
            value.unwrap_or(c.DEFAULT_EMPTY_STRING)
            if isinstance(value, p.Result)
            else value
        )
        match unwrapped:
            case str() | bytes():
                return unwrapped
            case m.ConfigMap():
                return FlextTestsPayloadUtilities.to_config_map(unwrapped.root)
            case m.BaseModel():
                return unwrapped
            case _ if self.is_mapping(unwrapped):
                return FlextTestsPayloadUtilities.to_config_map(unwrapped)
            case _ if self._is_nested_rows(unwrapped):
                # NOTE (multi-agent): retain the validated nested-row contract.
                return self._to_string_rows(unwrapped)
            case _:
                return str(unwrapped)

    def _extract_content[ContentT: t.Tests.FileContentPlain](
        self, content: ContentT | p.Result[ContentT], *, extract_result: bool
    ) -> t.Tests.FileContentPlain:
        """Extract actual content from a result-like wrapper or return as-is."""
        # ContentT <: FileContentPlain; invariant p.Result needs an explicit adapter.
        file_content = cast(
            "t.Tests.FileContentPlain | p.Result[t.Tests.FileContentPlain]", content
        )
        if not extract_result:
            return self._coerce_file_content(file_content)
        if isinstance(content, bytes):
            return content
        if self._is_file_result(file_content):
            if file_content.failure:
                error_msg = file_content.error or "r failure"
                msg = f"Cannot create file from failed r: {error_msg}"
                raise ValueError(msg)
            return self._coerce_file_content(file_content.value)
        return self._coerce_file_content(file_content)

    def _is_nested_rows(
        self, value: t.Tests.FileContentPlain | t.Tests.TestobjectSerializable
    ) -> TypeIs[Sequence[Sequence[t.Tests.TestobjectSerializable]]]:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            return False
        try:
            sequence_value = t.Tests.TESTOBJECT_SEQUENCE_ADAPTER.validate_python(value)
        except c.ValidationError:
            return False
        if not sequence_value:
            return False
        for row_raw in sequence_value:
            if not isinstance(row_raw, Sequence) or isinstance(row_raw, str | bytes):
                return False
        return True

    def _write_content_by_format(
        self,
        *,
        file_path: Path,
        actual_content: (
            str
            | bytes
            | m.ConfigMap
            | t.SequenceOf[t.StrSequence]
            | m.BaseModel
            | t.MappingKV[str, t.Tests.TestobjectSerializable]
        ),
        actual_fmt: str,
        params: m.Tests.CreateParams,
    ) -> None:
        """Write normalized content using the selected output format."""
        match actual_fmt:
            case c.Tests.FILE_FORMAT_BIN:
                _ = file_path.write_bytes(
                    actual_content
                    if isinstance(actual_content, bytes)
                    else str(actual_content).encode(params.enc)
                )
            case c.Tests.FILE_FORMAT_JSON | c.Tests.FILE_FORMAT_YAML:
                json_payload = self._build_json_payload(actual_content)
                if actual_fmt == c.Tests.FILE_FORMAT_JSON:
                    u.Cli.json_write(
                        file_path,
                        json_payload,
                        m.Cli.JsonWriteOptions(indent=params.indent),
                    )
                else:
                    u.Cli.yaml_dump(file_path, json_payload, indent=params.indent)
            case c.Tests.FILE_FORMAT_CSV:
                u.Cli.files_write_csv(
                    file_path,
                    self._build_csv_rows(
                        actual_content=actual_content, headers=params.headers
                    ),
                )
            case _:
                _ = file_path.write_text(str(actual_content), encoding=params.enc)

    @staticmethod
    def _build_json_payload(
        actual_content: (
            str
            | bytes
            | m.ConfigMap
            | t.SequenceOf[t.StrSequence]
            | m.BaseModel
            | t.MappingKV[str, t.Tests.TestobjectSerializable]
        ),
    ) -> t.JsonValue:
        """Build normalized JSON payload from arbitrary file content."""
        mapping_content: (
            t.MappingKV[str, t.Tests.TestobjectSerializable]
            | t.MappingKV[str, t.JsonPayload]
            | t.JsonMapping
            | None
        ) = (
            actual_content.root
            if isinstance(actual_content, (m.ConfigMap, m.Dict))
            else actual_content
            if isinstance(actual_content, Mapping)
            else None
        )
        fallback_value = FlextTestsPayloadUtilities.to_normalized_value(
            FlextTestsPayloadUtilities.to_payload(actual_content)
        )
        raw_payload: t.JsonValue = (
            {
                k: FlextTestsPayloadUtilities.to_normalized_value(
                    FlextTestsPayloadUtilities.to_payload(v)
                )
                for k, v in mapping_content.items()
            }
            if mapping_content is not None
            else {"value": fallback_value}
            if actual_content
            else {}
        )
        return t.json_value_adapter().validate_python(raw_payload)

    @staticmethod
    def _build_csv_rows(
        *,
        actual_content: (
            str
            | bytes
            | m.ConfigMap
            | t.SequenceOf[t.StrSequence]
            | m.BaseModel
            | t.MappingKV[str, t.Tests.TestobjectSerializable]
        ),
        headers: t.StrSequence | None,
    ) -> list[t.StrSequence]:
        """Build CSV rows from normalized content and optional headers."""
        csv_rows: list[t.StrSequence] = []
        if headers:
            csv_rows.append(list(headers))
        if isinstance(actual_content, Sequence) and not isinstance(
            actual_content, t.STR_BYTES_TYPES
        ):
            csv_rows.extend(
                list(row)
                for row in actual_content
                if isinstance(row, Sequence) and not isinstance(row, t.STR_BYTES_TYPES)
            )
        else:
            csv_rows.append([str(actual_content)])
        return csv_rows

    def create[ContentT: t.Tests.FileContentPlain](
        self,
        content: ContentT | p.Result[ContentT],
        name: str = c.Tests.DEFAULT_FILENAME,
        directory: Path | None = None,
        *,
        fmt: c.Tests.FileFormat = c.Tests.FILE_FORMAT_AUTO,
        enc: str = c.Tests.DEFAULT_ENCODING,
        indent: int = c.Tests.DEFAULT_JSON_INDENT,
        delim: str = c.Tests.DEFAULT_CSV_DELIMITER,
        headers: t.StrSequence | None = None,
        readonly: bool = False,
        extract_result: bool = True,
    ) -> Path:
        """Create file with auto-detection or explicit format."""
        # mro-j47u: result extraction is an explicit file-boundary choice.
        content_to_validate = self._extract_content(
            content, extract_result=extract_result
        )
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
        except c.EXC_BASIC_TYPE as exc:
            msg = f"Invalid parameters for file creation: {exc}"
            raise ValueError(msg) from None
        target_dir = self._resolve_directory(params.directory)
        file_path: Path = target_dir / params.name
        actual_content: (
            str
            | bytes
            | m.ConfigMap
            | t.SequenceOf[t.StrSequence]
            | m.BaseModel
            | t.MappingKV[str, t.Tests.TestobjectSerializable]
        ) = self._coerce_file_content(params.content)
        if isinstance(actual_content, m.ConfigMap):
            actual_content = {
                key: FlextTestsPayloadUtilities.to_payload(value)
                for key, value in actual_content.root.items()
            }
        if isinstance(actual_content, m.BaseModel):
            actual_content = t.Tests.TESTOBJECT_MAPPING_ADAPTER.validate_python(
                FlextTestsPayloadUtilities.to_payload(
                    actual_content.model_dump(mode="json")
                )
            )
        actual_fmt = u.Cli.files_detect_format_from_content(
            actual_content
            if isinstance(actual_content, (str, bytes, Mapping, list))
            else str(actual_content),
            params.name,
            params.fmt,
        )
        self._write_content_by_format(
            file_path=file_path,
            actual_content=actual_content,
            actual_fmt=actual_fmt,
            params=params,
        )
        if params.readonly:
            file_path.chmod(c.Tests.PERMISSION_READONLY_FILE)
        self._created_files.append(file_path)
        return file_path


__all__: list[str] = ["FlextTestsFilesCreationMixin"]
