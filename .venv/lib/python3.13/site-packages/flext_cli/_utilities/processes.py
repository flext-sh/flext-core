"""Managed process primitives shared through ``u.Cli``."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from flext_cli import c, p, r, t
from flext_cli._utilities.runtime import FlextCliUtilitiesRuntime


class FlextCliUtilitiesProcesses:
    """Runtime helpers for managed external processes."""

    class ManagedProcess:
        """Typed handle for a long-running child process."""

        def __init__(
            self,
            process: subprocess.Popen[str],
            *,
            cwd: t.Cli.TextPath | None,
            env: t.StrMapping | None,
        ) -> None:
            self._process = process
            self._cwd = Path(cwd) if cwd is not None else None
            self._env = dict(env) if env is not None else None
            self._stdout = ""
            self._stderr = ""
            self._communicated = False

        @property
        def pid(self) -> int:
            return self._process.pid

        @property
        def returncode(self) -> int | None:
            return self._process.returncode

        @property
        def cwd(self) -> Path | None:
            return self._cwd

        @property
        def env(self) -> Mapping[str, str] | None:
            if self._env is None:
                return None
            return MappingProxyType(self._env)

        @property
        def stdout(self) -> str:
            return self._stdout

        @property
        def stderr(self) -> str:
            return self._stderr

        def poll(self) -> int | None:
            return self._process.poll()

        def terminate(self) -> p.Result[bool]:
            if self.poll() is not None:
                return r[bool].ok(True)
            try:
                self._process.terminate()
            except c.EXC_OS_VALUE as exc:
                return r[bool].fail(f"process terminate error: {exc}")
            return r[bool].ok(True)

        def kill(self) -> p.Result[bool]:
            if self.poll() is not None:
                return r[bool].ok(True)
            try:
                self._process.kill()
            except c.EXC_OS_VALUE as exc:
                return r[bool].fail(f"process kill error: {exc}")
            return r[bool].ok(True)

        def wait(self, timeout: float | None = None) -> p.Result[int]:
            if self._communicated:
                return r[int].ok(self._process.returncode or 0)
            try:
                stdout, stderr = self._process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                return r[int].fail(f"timeout {exc.timeout}s: pid {self.pid}")
            except c.EXC_OS_VALUE as exc:
                return r[int].fail(f"process wait error: {exc}")
            self._stdout = stdout or ""
            self._stderr = stderr or ""
            self._communicated = True
            return r[int].ok(self._process.returncode or 0)

    @staticmethod
    def process_start(
        cmd: t.StrSequence,
        cwd: t.Cli.TextPath | None = None,
        env: t.StrMapping | None = None,
        remove_env_keys: t.StrSequence = (),
    ) -> p.Result[FlextCliUtilitiesProcesses.ManagedProcess]:
        """Start long-running commands; use instead of direct ``subprocess.Popen``."""
        resolved_env = None
        if env is not None or remove_env_keys:
            resolved_env = FlextCliUtilitiesRuntime.process_env(
                overrides=env, remove_keys=remove_env_keys
            )
        try:
            process = subprocess.Popen(
                list(cmd),
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=resolved_env,
            )
        except c.EXC_OS_VALUE as exc:
            return r[FlextCliUtilitiesProcesses.ManagedProcess].fail(
                f"execution error: {shlex.join(list(cmd))}: {exc}"
            )
        return r[FlextCliUtilitiesProcesses.ManagedProcess].ok(
            FlextCliUtilitiesProcesses.ManagedProcess(
                process, cwd=cwd, env=resolved_env
            )
        )


__all__: list[str] = ["FlextCliUtilitiesProcesses"]
