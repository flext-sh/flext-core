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

from pathlib import Path
from typing import Annotated, ClassVar, Self, override

from flext_tests import m, p, r, s, t, u
from flext_tests._utilities._files._comparison import FlextTestsFilesComparisonMixin
from flext_tests._utilities._files._info import FlextTestsFilesInfoMixin


class FlextTestsFiles(s, FlextTestsFilesInfoMixin, FlextTestsFilesComparisonMixin):
    """Manages test files for FLEXT ecosystem testing."""

    FileInfo: ClassVar[type[m.Tests.FileInfo]] = m.Tests.FileInfo
    base_dir: Annotated[
        Path | None,
        m.BeforeValidator(
            lambda value: Path(value) if isinstance(value, str) else value
        ),
    ] = u.Field(default=None, description="Base directory used for file operations.")

    @override
    def model_post_init(self, __context: t.JsonValue | None, /) -> None:
        """Initialize private file-manager lifecycle state."""
        super().model_post_init(__context)
        self._initialize_file_lifecycle()

    @classmethod
    @override
    def _create_file_manager(cls, base_dir: Path | None) -> Self:
        """Construct a validated file manager for class-level contexts."""
        return cls(base_dir=base_dir)

    @override
    def execute(self) -> p.Result[p.Base]:
        """Execute is not the file-manager API surface.

        FlextTestsFiles is a utility service whose real API is its file
        methods (create, compare, info, ...); execute has no domain result.
        """
        return r[p.Base].fail(
            "Use specific file methods: create, compare, read, info, ..."
        )


tf = FlextTestsFiles

__all__: list[str] = ["FlextTestsFiles", "tf"]
