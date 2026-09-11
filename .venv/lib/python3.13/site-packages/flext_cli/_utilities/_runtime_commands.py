"""Public command-result composition for ``u.Cli`` runtime."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from flext_cli import p, r, t


class FlextCliUtilitiesRuntimeCommandsMixin:
    """Compose captured command primitives without owning subprocess creation."""

    if TYPE_CHECKING:

        @classmethod
        def run_raw(
            cls,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
            *,
            capture: bool = True,
        ) -> p.Result[p.Cli.CommandOutput]: ...

    @classmethod
    def run(
        cls,
        cmd: t.StrSequence,
        cwd: t.Cli.TextPath | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
        input_data: str | bytes | None = None,
        *,
        capture: bool = True,
    ) -> p.Result[p.Cli.CommandOutput]:
        """Run a command and fail on non-zero exit status."""

        def require_zero_exit(
            output: p.Cli.CommandOutput,
        ) -> p.Result[p.Cli.CommandOutput]:
            if output.exit_code != 0:
                detail = (output.stderr or output.stdout).strip()
                return r[p.Cli.CommandOutput].fail(
                    f"failed ({output.exit_code}): {shlex.join(list(cmd))}: {detail}"
                )
            return r[p.Cli.CommandOutput].ok(output)

        return cls.run_raw(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
            remove_env_keys=remove_env_keys,
            input_data=input_data,
            capture=capture,
        ).flat_map(require_zero_exit)

    @classmethod
    def run_checked(
        cls,
        cmd: t.StrSequence,
        cwd: t.Cli.TextPath | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
        input_data: str | bytes | None = None,
        *,
        capture: bool = True,
    ) -> p.Result[bool]:
        """Run a command and return a success flag."""
        return cls.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
            remove_env_keys=remove_env_keys,
            input_data=input_data,
            capture=capture,
        ).map(lambda _: True)

    @classmethod
    def run_live(
        cls,
        cmd: t.StrSequence,
        cwd: t.Cli.TextPath | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
        input_data: str | bytes | None = None,
    ) -> p.Result[p.Cli.CommandOutput]:
        """Run a command with inherited live stdout and stderr."""
        return cls.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
            remove_env_keys=remove_env_keys,
            input_data=input_data,
            capture=False,
        )

    @classmethod
    def capture(
        cls,
        cmd: t.StrSequence,
        cwd: t.Cli.TextPath | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
        input_data: str | bytes | None = None,
    ) -> p.Result[str]:
        """Run a command and return stripped stdout."""
        return cls.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
            remove_env_keys=remove_env_keys,
            input_data=input_data,
        ).map(lambda output: output.stdout.strip())


__all__: list[str] = ["FlextCliUtilitiesRuntimeCommandsMixin"]
