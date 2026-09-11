"""Canonical streamed process runner exposed through ``u.Cli``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import p, t
from flext_cli._utilities._runtime_process_execution import (
    FlextCliUtilitiesRuntimeProcessExecutionMixin,
)


class FlextCliUtilitiesRuntimeRunToFileMixin(
    FlextCliUtilitiesRuntimeProcessExecutionMixin
):
    """Validate and dispatch one portable streamed process lifecycle."""

    if TYPE_CHECKING:

        @staticmethod
        def _resolved_env(
            env: t.StrMapping | None, remove_env_keys: t.StrSequence = ()
        ) -> dict[str, str] | None: ...

    @classmethod
    def run_to_file(
        cls,
        cmd: t.StrSequence,
        output_file: t.Cli.TextPath,
        cwd: t.Cli.TextPath | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
        input_data: str | bytes | None = None,
        *,
        live: bool = False,
        deadline: p.Cli.ProcessDeadline | None = None,
    ) -> p.Result[int]:
        """Stream combined bytes live and durably under one absolute deadline.

        Containment owns the inherited POSIX process group or Windows Job
        Object. Trusted project tools remain inside that boundary; deliberate
        POSIX ``setsid()`` escape is outside this contract. The deadline covers
        child execution, termination, reaping, stream drain, and durable flush.
        An outer caller wall remains responsible for an OS syscall that becomes
        uninterruptible.
        """
        return cls._execute_streamed_process(
            cmd,
            Path(output_file),
            cwd,
            cls._resolved_env(env, remove_env_keys),
            input_data,
            capture_output=False,
            live=live,
            timeout=timeout,
            deadline=deadline,
        ).map(lambda output: output.exit_code)


__all__: list[str] = ["FlextCliUtilitiesRuntimeRunToFileMixin"]
