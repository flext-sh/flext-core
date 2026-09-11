"""Models extraction for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from flext_infra import m, u
from flext_tests import c, p, t


class FlextTestsBatchModelsMixin:
    class BatchParams(m.Value):
        """Parameters for FlextTestsFiles.batch() method."""

        files: Annotated[
            (
                t.MappingKV[str, t.Tests.TestobjectSerializable]
                | t.SequenceOf[tuple[str, t.Tests.TestobjectSerializable]]
            ),
            u.Field(description="Mapping or Sequence of files to process"),
        ]
        directory: Annotated[
            Path | None, u.Field(description="Target directory for create operations")
        ] = None
        operation: Annotated[
            c.Tests.Operation,
            m.BeforeValidator(
                lambda v: c.Tests.Operation(v) if isinstance(v, str) else v
            ),
            u.Field(
                default=c.Tests.Operation.CREATE,
                description="Operation type: create, read, or delete",
            ),
        ]
        model: Annotated[
            type[m.BaseModel] | None,
            u.Field(description="Optional model class for read operations"),
        ] = None
        on_error: Annotated[
            c.Tests.ErrorMode,
            m.BeforeValidator(
                lambda v: c.Tests.ErrorMode(v) if isinstance(v, str) else v
            ),
            u.Field(
                default=c.Tests.ErrorMode.COLLECT,
                description="Error handling mode: stop, skip, or collect",
            ),
        ]
        parallel: Annotated[bool, u.Field(description="Run operations in parallel")] = (
            False
        )

    class BatchResult(m.Value):
        """Result of batch file operations."""

        succeeded: Annotated[
            int, u.Field(ge=0, description="Number of successful operations")
        ]
        failed: Annotated[
            t.NonNegativeInt, u.Field(description="Number of failed operations")
        ]
        total: Annotated[
            t.NonNegativeInt, u.Field(description="Total number of operations")
        ]
        results: Annotated[
            t.MappingKV[str, p.Result[t.Tests.TestResultValue]],
            u.Field(description="Mapping of file names to operation results"),
        ] = u.Field(
            default_factory=lambda: MappingProxyType(
                dict[str, p.Result[t.Tests.TestResultValue]]()
            )
        )
        errors: Annotated[
            t.StrMapping, u.Field(description="Mapping of file names to error messages")
        ] = u.Field(default_factory=lambda: MappingProxyType(dict[str, str]()))

        @u.computed_field
        @property
        def failure_count(self) -> int:
            """Alias for failed count."""
            failed_count: int = self.failed
            return failed_count

        @u.computed_field
        @property
        def success_count(self) -> int:
            """Alias for succeeded count."""
            succeeded_count: int = self.succeeded
            return succeeded_count

        @u.computed_field
        @property
        def success_rate(self) -> float:
            """Compute success rate as percentage."""
            total: int = self.total
            if total == 0:
                return 0.0
            succeeded: int = self.succeeded
            return (succeeded / total) * 100.0
