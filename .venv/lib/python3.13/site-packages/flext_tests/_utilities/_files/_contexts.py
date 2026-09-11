"""File-context helpers for FlextTestsFiles.

Temporary file bundles and temporary directory context managers.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path

from flext_tests import c, m, t
from flext_tests._utilities._files._reading import FlextTestsFilesReadingMixin


class FlextTestsFilesContextsMixin(FlextTestsFilesReadingMixin):
    """Context managers for temporary files and directories."""

    @classmethod
    @contextmanager
    def files(
        cls,
        content: t.MappingKV[str, t.Tests.FileContentPlain],
        *,
        directory: Path | None = None,
        ext: str | None = None,
        extract_result: bool = True,
        **kwargs: t.Tests.TestobjectSerializable,
    ) -> Generator[Mapping[str, Path]]:
        """Create temporary files with auto-cleanup.

        Supports Pydantic models, dicts, lists, and raw content.

        Args:
            content: Dict mapping names to content (str, bytes, dict, list, m.BaseModel)
            directory: Base directory (temp if None)
            ext: Default extension if not in name
            extract_result: Auto-extract r values (default: True)
            **kwargs: Passed to create()

        Yields:
            Dict mapping names to paths.

        """
        manager = cls._create_file_manager(directory)
        with manager:
            paths: MutableMapping[str, Path] = {}
            default_ext = ext or c.Tests.DEFAULT_EXTENSION
            for name, data_raw in content.items():
                data: t.Tests.FileContentPlain = data_raw
                filename = name if "." in name else f"{name}{default_ext}"
                if "." not in name and isinstance(
                    data, (Mapping, m.BaseModel, m.ConfigMap, m.Dict)
                ):
                    filename = f"{name}.json"
                else:
                    is_nested_sequence = "." not in name and cls._is_nested_rows(
                        manager, data
                    )
                    if is_nested_sequence:
                        filename = f"{name}.csv"
                try:
                    validated_kwargs = m.Tests.CreateKwargsParams.model_validate(kwargs)
                except c.EXC_VALIDATION_TYPE_VALUE:
                    validated_kwargs = m.Tests.CreateKwargsParams()
                path = manager.create(
                    cls._coerce_file_content(manager, data),
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

    @contextmanager
    def temporary_directory(self) -> Generator[Path]:
        """Create and manage a temporary directory.

        Yields:
            Path to temporary directory that is automatically cleaned up.

        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


__all__: list[str] = ["FlextTestsFilesContextsMixin"]
