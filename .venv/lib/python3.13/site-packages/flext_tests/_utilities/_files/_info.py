"""File-info helpers for FlextTestsFiles.

Comprehensive metadata extraction and optional content parsing.
"""

from __future__ import annotations

from pathlib import Path

from flext_cli import u
from flext_tests import c, m, p, r, t
from flext_tests._utilities._files._assertions import FlextTestsFilesAssertionsMixin
from flext_tests._utilities.files import FlextTestsFilesUtilitiesMixin


class FlextTestsFilesInfoMixin(FlextTestsFilesAssertionsMixin):
    """Get comprehensive file information and content metadata."""

    def info(
        self,
        path: Path,
        *,
        compute_hash: bool = False,
        detect_fmt: bool = True,
        parse_content: bool = False,
        validate_model: type[m.BaseModel] | None = None,
    ) -> p.Result[m.Tests.FileInfo]:
        """Get comprehensive file information.

        Args:
            path: File path
            compute_hash: Compute SHA256 (default: False)
            detect_fmt: Auto-detect format (default: True)
            parse_content: Parse content and include metadata (default: False)
            validate_model: Pydantic model to validate content against (default: None)

        Returns:
            r[FileInfo] with info or error.

        """
        try:
            params = m.Tests.InfoParams.model_validate({
                "path": path,
                "compute_hash": compute_hash,
                "detect_fmt": detect_fmt,
                "parse_content": parse_content,
                "validate_model": validate_model,
            })
        except c.EXC_BASIC_TYPE as exc:
            return r[m.Tests.FileInfo].fail(f"Invalid parameters for file info: {exc}")
        if not params.path.exists():
            return r[m.Tests.FileInfo].ok(
                m.Tests.FileInfo(exists=False, path=params.path)
            )
        try:
            return r[m.Tests.FileInfo].ok(self._build_file_info(params))
        except OSError as e:
            return r[m.Tests.FileInfo].fail(c.Tests.ERROR_INFO.format(error=e))

    def _build_file_info(self, params: m.Tests.InfoParams) -> m.Tests.FileInfo:
        """Build a ``FileInfo`` model for an existing path."""
        stat = params.path.stat()
        size = stat.st_size
        size_human = FlextTestsFilesUtilitiesMixin.format_size(size)
        text, lines, is_empty, first_line, encoding = self._read_info_text(
            params.path, size
        )
        fmt: str = "unknown"
        if params.detect_fmt:
            detected = u.Cli.files_detect_format_from_path(params.path, "auto")
            fmt = detected if detected in c.Tests.KNOWN_FORMATS else "unknown"
        permissions = stat.st_mode
        is_readonly = not permissions & 128
        sha256 = u.Cli.sha256_file(params.path) if params.compute_hash else None
        content_meta: m.Tests.ContentMeta | None = None
        if params.parse_content or params.validate_model:
            content_meta = self._parse_content_metadata(
                text=text, fmt=fmt, validate_model=params.validate_model
            )
        return m.Tests.FileInfo(
            exists=True,
            path=params.path,
            size=size,
            size_human=size_human,
            lines=lines,
            encoding=encoding,
            is_empty=is_empty,
            first_line=first_line,
            fmt=fmt,
            valid=True,
            modified=u.from_timestamp(stat.st_mtime),
            permissions=permissions,
            is_readonly=is_readonly,
            sha256=sha256,
            content_meta=content_meta,
        )

    def _read_info_text(self, path: Path, size: int) -> tuple[str, int, bool, str, str]:
        """Read text metadata for a file, falling back to binary defaults."""
        try:
            text = path.read_text(encoding=c.Tests.DEFAULT_ENCODING, errors="replace")
            lines = text.count("\n") + 1 if text else 0
            is_empty = not text.strip()
            first_line = text.split("\n")[0] if text else ""
        except UnicodeDecodeError:
            return ("", 0, size == 0, "", c.Tests.DEFAULT_BINARY_ENCODING)
        else:
            return (text, lines, is_empty, first_line, c.Tests.DEFAULT_ENCODING)

    def _parse_content_metadata(
        self, text: str, fmt: str, validate_model: type[m.BaseModel] | None = None
    ) -> m.Tests.ContentMeta:
        """Parse file content and extract metadata.

        Uses Pydantic model_validate for validation and format-specific parsing
        to extract content statistics (key_count, item_count, row_count, etc.).

        Args:
            text: File text content
            fmt: Detected file format
            validate_model: Pydantic model to validate content against

        Returns:
            ContentMeta with extracted statistics

        """
        key_count = item_count = row_count = column_count = None
        model_valid: bool | None = None
        model_name = validate_model.__name__ if validate_model else None
        parsed_mapping: t.MappingKV[str, t.Tests.TestobjectSerializable] | None = None
        match fmt:
            case "json" | "yaml" if text.strip():
                parse_result = (
                    u.Cli.json_parse(text) if fmt == "json" else u.Cli.yaml_parse(text)
                )
                parsed_value = parse_result.value if parse_result.success else None
                match parsed_value:
                    case dict() as parsed_dict:
                        parsed_mapping = parsed_dict
                        key_count = len(parsed_dict)
                    case list() as parsed_list:
                        item_count = len(parsed_list)
                    case _:
                        pass
            case "csv":
                csv_outcome = u.Cli.csv_loads(text)
                rows: list[list[str]] = csv_outcome.value if csv_outcome.success else []
                if rows:
                    row_count = len(rows)
                    column_count = len(rows[0]) if rows[0] else 0
            case _:
                pass
        if validate_model is not None:
            if parsed_mapping is not None:
                # mro-j47u: consume the composed reading capability through self.
                model_valid = self._validate_model_content(
                    validate_model, parsed_mapping
                ).success
            elif fmt in {"json", "yaml"} and text.strip():
                model_valid = False
        return m.Tests.ContentMeta.model_validate({
            "key_count": key_count,
            "item_count": item_count,
            "row_count": row_count,
            "column_count": column_count,
            "model_valid": model_valid,
            "model_name": model_name,
        })


__all__: list[str] = ["FlextTestsFilesInfoMixin"]
